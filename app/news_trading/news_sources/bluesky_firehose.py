"""
bluesky_firehose.py — Real-time Bluesky post streaming via Jetstream.

Replaces polling-based monitoring.  Bluesky's Jetstream is a free,
unauthenticated WebSocket service that streams every public post within
1-2 seconds of publication — compared to 15-30 second polling latency.

Endpoint: wss://jetstream2.us-east.bsky.network/subscribe

We filter the firehose by:
  - wantedCollections=app.bsky.feed.post  (posts only, no likes/reposts)
  - wantedDids=did:plc:xxx&...             (only our target accounts)

At startup we resolve handles → DIDs via the AT Protocol getProfile API,
then open a single persistent WebSocket and process events as they arrive.
On disconnect we reconnect after a short backoff.
"""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

import httpx
import websockets

from app.news_trading.news_sources.bluesky_monitor import (
    BLUESKY_CANDIDATES,
    _extra_accounts_from_env,
)

_JETSTREAM_URL = "wss://jetstream2.us-east.bsky.network/subscribe"
_API_BASE = "https://public.api.bsky.app/xrpc"
_RESOLVE_TIMEOUT = 10.0
_RESOLVE_CONCURRENCY = 5
_RECONNECT_DELAY = 5.0


class BlueskyFirehose:
    def __init__(self, on_news: Callable[[dict], Awaitable[None]]) -> None:
        self._on_news = on_news
        self._running = False
        self._did_to_account: Dict[str, Tuple[str, str]] = {}  # did → (label, handle)
        self._seen_uris: Set[str] = set()

    async def start(self) -> None:
        self._running = True
        print(
            "[News] Bluesky Firehose starting — resolving accounts to DIDs...",
            flush=True,
        )

        candidates = list(BLUESKY_CANDIDATES) + _extra_accounts_from_env()
        resolved = await self._resolve_accounts_to_dids(candidates)

        if not resolved:
            print(
                "[News] Bluesky Firehose: no valid accounts resolved. "
                "Retrying in 5 minutes...",
                flush=True,
            )
            while self._running:
                await asyncio.sleep(300)
                resolved = await self._resolve_accounts_to_dids(candidates)
                if resolved:
                    break
            if not resolved:
                return

        self._did_to_account = {did: (label, handle) for label, handle, did in resolved}
        dids = list(self._did_to_account.keys())
        handles_preview = ", ".join(handle for _, handle, _ in resolved[:8])
        more = f" …+{len(resolved) - 8}" if len(resolved) > 8 else ""
        print(
            f"[News] Bluesky Firehose ready — {len(resolved)} accounts, "
            f"real-time via Jetstream\n"
            f"[News] Bluesky firehose accounts: {handles_preview}{more}",
            flush=True,
        )

        while self._running:
            try:
                await self._consume_firehose(dids)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                print(
                    f"[News] Bluesky Firehose disconnected ({type(exc).__name__}: "
                    f"{exc}) — reconnecting in {_RECONNECT_DELAY}s",
                    flush=True,
                )
                await asyncio.sleep(_RECONNECT_DELAY)

    async def stop(self) -> None:
        self._running = False

    # ── handle → DID resolution (one-time at startup) ──────────────────────

    async def _resolve_accounts_to_dids(
        self, candidates: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, str]]:
        """Resolve handles to DIDs via getProfile. Returns (label, handle, did)."""
        seen_handles: Set[str] = set()
        unique: List[Tuple[str, str]] = []
        for label, handle in candidates:
            h_lower = handle.lower()
            if h_lower not in seen_handles:
                seen_handles.add(h_lower)
                unique.append((label, handle))

        sem = asyncio.Semaphore(_RESOLVE_CONCURRENCY)

        async def _check(label: str, handle: str) -> Optional[Tuple[str, str, str]]:
            async with sem:
                try:
                    async with httpx.AsyncClient(timeout=_RESOLVE_TIMEOUT) as client:
                        resp = await client.get(
                            f"{_API_BASE}/app.bsky.actor.getProfile",
                            params={"actor": handle},
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            did = data.get("did")
                            followers = data.get("followersCount", 0)
                            posts = data.get("postsCount", 0)
                            if not did or posts == 0 or followers < 100:
                                return None
                            display = data.get("displayName") or label
                            return (display, handle, did)
                except Exception:
                    pass
                return None

        tasks = [_check(label, handle) for label, handle in unique]
        results = await asyncio.gather(*tasks)
        resolved = [r for r in results if r is not None]
        print(
            f"[News] Bluesky Firehose resolved {len(resolved)}/{len(unique)} "
            f"accounts to DIDs",
            flush=True,
        )
        return resolved

    # ── Jetstream WebSocket consumer ──────────────────────────────────────

    async def _consume_firehose(self, dids: List[str]) -> None:
        """Open WebSocket to Jetstream and process events as they stream in."""
        # Build URL with filters: posts only + our target DIDs
        params = ["wantedCollections=app.bsky.feed.post"]
        params.extend(f"wantedDids={did}" for did in dids)
        url = f"{_JETSTREAM_URL}?{'&'.join(params)}"

        print("[News] Bluesky Firehose connecting...", flush=True)

        # max_size=8MB just in case (typical post events are <2KB)
        async with websockets.connect(
            url,
            max_size=8 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            print(
                "[News] Bluesky Firehose connected — streaming live posts",
                flush=True,
            )
            async for message in ws:
                if not self._running:
                    break
                try:
                    await self._process_event(message)
                except Exception as exc:
                    print(
                        f"[News] Bluesky Firehose event error: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

    async def _process_event(self, raw) -> None:
        """Decode + filter + forward a single Jetstream event."""
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        event = json.loads(raw)

        # We only care about new post creates
        if event.get("kind") != "commit":
            return
        commit = event.get("commit", {})
        if commit.get("operation") != "create":
            return
        if commit.get("collection") != "app.bsky.feed.post":
            return

        did = event.get("did", "")
        record = commit.get("record", {})
        text = (record.get("text") or "").strip()
        if not text or not did:
            return

        # Skip replies — we want original content only
        if record.get("reply"):
            return

        rkey = commit.get("rkey", "")
        uri = f"at://{did}/app.bsky.feed.post/{rkey}"

        # Dedup (very rare for firehose but defend against reconnect overlap)
        if uri in self._seen_uris:
            return
        self._seen_uris.add(uri)

        label, handle = self._did_to_account.get(did, ("Unknown", did))
        created_at = record.get("createdAt") or ""
        web_url = (
            f"https://bsky.app/profile/{handle}/post/{rkey}"
            if handle and rkey
            else ""
        )

        item = {
            "id": f"bsky_{uri}",
            "headline": f"{label}: {text[:250]}",
            "summary": text,
            "content": text,
            "source": f"bluesky/@{handle}",
            "symbols": [],
            "asset_class": "equity",
            "published_at": created_at,
            "url": web_url,
        }

        # Trim seen_uris periodically
        if len(self._seen_uris) > 5000:
            self._seen_uris = set(list(self._seen_uris)[-2500:])

        await self._on_news(item)


def create_bluesky_firehose(
    on_news: Callable[[dict], Awaitable[None]],
) -> BlueskyFirehose:
    return BlueskyFirehose(on_news)
