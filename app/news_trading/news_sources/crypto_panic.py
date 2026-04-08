"""
crypto_panic.py — CryptoPanic REST news polling (60-second interval).

Endpoint: https://cryptopanic.com/api/v1/posts/
Filters: hot news. Extracts crypto symbols from the currencies field.
Skipped gracefully if NEWS_CRYPTO_PANIC_API_KEY is not set.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional, Set

import httpx

_CRYPTO_PANIC_URL = "https://cryptopanic.com/api/v1/posts/"
_POLL_INTERVAL = 60.0


class CryptoPanicFeed:
    def __init__(
        self,
        api_key: str,
        on_news: Callable[[dict], Awaitable[None]],
    ) -> None:
        self._api_key = api_key
        self._on_news = on_news
        self._running = False
        self._seen_ids: Set[str] = set()

    async def start(self) -> None:
        self._running = True
        print("[News] CryptoPanic polling started (60s interval)", flush=True)
        while self._running:
            try:
                await self._poll()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[News] CryptoPanic poll error: {exc}", flush=True)
            await asyncio.sleep(_POLL_INTERVAL)

    async def stop(self) -> None:
        self._running = False

    async def _poll(self) -> None:
        params = {
            "auth_token": self._api_key,
            "filter": "hot",
            "kind": "news",
            "public": "true",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_CRYPTO_PANIC_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        posts = data.get("results") or []
        for post in reversed(posts):
            pid = str(post.get("id") or "")
            if not pid or pid in self._seen_ids:
                continue
            self._seen_ids.add(pid)
            item = self._normalize(post, pid)
            if item:
                await self._on_news(item)

        if len(self._seen_ids) > 500:
            self._seen_ids = set(list(self._seen_ids)[-500:])

    def _normalize(self, post: dict, pid: str) -> Optional[dict]:
        headline = (post.get("title") or "").strip()
        if not headline:
            return None
        # Extract crypto symbols from currencies field
        currencies = post.get("currencies") or []
        symbols = [c.get("code", "").upper() for c in currencies if c.get("code")]
        # Convert to /USD pair format used in the rest of the system
        pairs = [f"{sym}/USD" for sym in symbols if sym]
        return {
            "id": pid,
            "headline": headline,
            "summary": "",
            "source": (post.get("source", {}).get("title") or "cryptopanic").strip(),
            "symbols": pairs,
            "asset_class": "crypto",
            "published_at": (post.get("published_at") or ""),
            "url": (post.get("url") or "").strip(),
        }


def create_crypto_panic_feed(
    api_key: Optional[str],
    on_news: Callable[[dict], Awaitable[None]],
) -> Optional[CryptoPanicFeed]:
    if not api_key:
        print("[News] CryptoPanic feed skipped (NEWS_CRYPTO_PANIC_API_KEY not set)", flush=True)
        return None
    return CryptoPanicFeed(api_key, on_news)
