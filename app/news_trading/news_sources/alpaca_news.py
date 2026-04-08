"""
alpaca_news.py — Alpaca real-time news WebSocket stream.

Endpoint: wss://stream.data.alpaca.markets/v1beta1/news
Subscribes to all symbols ("*"). Auto-reconnects with exponential backoff.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Awaitable, Callable, Optional


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace for clean plain text."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

logger = logging.getLogger(__name__)

_WS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
_MAX_BACKOFF = 30.0


class AlpacaNewsStream:
    def __init__(
        self,
        api_key_id: str,
        api_secret_key: str,
        on_news: Callable[[dict], Awaitable[None]],
    ) -> None:
        self._key = api_key_id
        self._secret = api_secret_key
        self._on_news = on_news
        self._running = False
        self._ws = None

    async def start(self) -> None:
        """Connect, subscribe, and loop forever with exponential backoff reconnects."""
        self._running = True
        backoff = 1.0
        attempt = 0

        while self._running:
            try:
                await self._connect_and_run()
                backoff = 1.0
                attempt = 0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._running:
                    break
                attempt += 1
                wait = min(backoff * (2 ** (attempt - 1)), _MAX_BACKOFF)
                print(
                    f"[News] Alpaca WebSocket disconnected (attempt {attempt}): {exc}. "
                    f"Reconnecting in {wait:.0f}s...",
                    flush=True,
                )
                await asyncio.sleep(wait)

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def _connect_and_run(self) -> None:
        try:
            import websockets
        except ImportError:
            print(
                "[News] Alpaca news stream requires 'websockets'. "
                "Run: pip install websockets>=12.0",
                flush=True,
            )
            self._running = False
            return

        async with websockets.connect(_WS_URL) as ws:
            self._ws = ws
            print("[News] Alpaca news stream connected", flush=True)

            # Auth
            await ws.send(
                json.dumps({"action": "auth", "key": self._key, "secret": self._secret})
            )
            auth_resp = await ws.recv()
            self._check_auth(auth_resp)

            # Subscribe
            await ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))
            sub_resp = await ws.recv()
            self._log_subscription(sub_resp)

            print("[News] Alpaca news stream subscribed to all symbols", flush=True)

            # Message loop
            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                except asyncio.TimeoutError:
                    # Send ping to keep alive
                    await ws.ping()
                    continue

                await self._handle_message(raw)

    def _check_auth(self, raw: str) -> None:
        try:
            msgs = json.loads(raw)
            if not isinstance(msgs, list):
                msgs = [msgs]
            for m in msgs:
                if m.get("T") == "error":
                    raise ValueError(f"Alpaca auth error: {m.get('msg')}")
        except json.JSONDecodeError:
            pass  # Non-fatal; proceed

    def _log_subscription(self, raw: str) -> None:
        try:
            msgs = json.loads(raw)
            if not isinstance(msgs, list):
                msgs = [msgs]
            for m in msgs:
                if m.get("T") == "error":
                    print(f"[News] Alpaca subscription error: {m.get('msg')}", flush=True)
        except json.JSONDecodeError:
            pass

    async def _handle_message(self, raw: str) -> None:
        try:
            msgs = json.loads(raw)
            if not isinstance(msgs, list):
                msgs = [msgs]
            for msg in msgs:
                if msg.get("T") != "n":
                    continue
                item = self._normalize(msg)
                if item:
                    await self._on_news(item)
        except Exception as exc:
            print(f"[News] Alpaca message parse error: {exc}", flush=True)

    def _normalize(self, msg: dict) -> Optional[dict]:
        headline = (msg.get("headline") or "").strip()
        if not headline:
            return None
        # Strip HTML tags from content for clean plain text
        raw_content = (msg.get("content") or "").strip()
        content = _strip_html(raw_content)
        return {
            "id": str(msg.get("id", "")),
            "headline": headline,
            "summary": (msg.get("summary") or "").strip(),
            "content": content,
            "source": (msg.get("source") or "alpaca").strip(),
            "symbols": [s.upper() for s in (msg.get("symbols") or []) if s],
            "asset_class": "equity",
            "published_at": (msg.get("created_at") or msg.get("updated_at") or ""),
            "url": (msg.get("url") or "").strip(),
        }


def create_alpaca_news_stream(
    api_key_id: Optional[str],
    api_secret_key: Optional[str],
    on_news: Callable[[dict], Awaitable[None]],
) -> Optional[AlpacaNewsStream]:
    """Returns a stream instance or None if keys are missing."""
    if not api_key_id or not api_secret_key:
        print("[News] Alpaca news stream skipped (no API keys)", flush=True)
        return None
    return AlpacaNewsStream(api_key_id, api_secret_key, on_news)
