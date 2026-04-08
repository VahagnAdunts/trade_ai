"""
polygon_news.py — Polygon.io REST news polling (30-second interval).

Endpoint: GET https://api.polygon.io/v2/reference/news
Tracks last seen article ID to avoid reprocessing.
Skipped gracefully if NEWS_POLYGON_API_KEY is not set.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional, Set

import httpx

_POLYGON_URL = "https://api.polygon.io/v2/reference/news"
_POLL_INTERVAL = 30.0


class PolygonNewsFeed:
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
        print("[News] Polygon polling started (30s interval)", flush=True)
        while self._running:
            try:
                await self._poll()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[News] Polygon poll error: {exc}", flush=True)
            await asyncio.sleep(_POLL_INTERVAL)

    async def stop(self) -> None:
        self._running = False

    async def _poll(self) -> None:
        params = {
            "limit": 10,
            "order": "desc",
            "sort": "published_utc",
            "apiKey": self._api_key,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_POLYGON_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("results") or []
        for article in reversed(articles):   # oldest first so ordering is natural
            aid = str(article.get("id") or article.get("amp_url") or "")
            if not aid or aid in self._seen_ids:
                continue
            self._seen_ids.add(aid)
            item = self._normalize(article, aid)
            if item:
                await self._on_news(item)

        # Prevent unbounded growth — keep only last 500 ids
        if len(self._seen_ids) > 500:
            self._seen_ids = set(list(self._seen_ids)[-500:])

    def _normalize(self, article: dict, aid: str) -> Optional[dict]:
        headline = (article.get("title") or "").strip()
        if not headline:
            return None
        tickers = article.get("tickers") or []
        return {
            "id": aid,
            "headline": headline,
            "summary": (article.get("description") or "").strip(),
            "source": (article.get("publisher", {}).get("name") or "polygon").strip(),
            "symbols": [t.upper() for t in tickers if t],
            "asset_class": "equity",
            "published_at": (article.get("published_utc") or ""),
            "url": (article.get("article_url") or "").strip(),
        }


def create_polygon_feed(
    api_key: Optional[str],
    on_news: Callable[[dict], Awaitable[None]],
) -> Optional[PolygonNewsFeed]:
    if not api_key:
        print("[News] Polygon feed skipped (NEWS_POLYGON_API_KEY not set)", flush=True)
        return None
    return PolygonNewsFeed(api_key, on_news)
