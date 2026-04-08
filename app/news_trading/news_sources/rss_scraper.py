"""
rss_scraper.py — free RSS fallback news feed (90-second polling interval).

Polls major financial RSS feeds, tracks seen GUIDs, and extracts equity
symbols by matching known S&P 500-era tickers mentioned in headlines.
If feedparser is not installed, logs a warning and skips silently.
"""
from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable, List, Optional, Set

_POLL_INTERVAL = 90.0

_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "https://feeds.marketwatch.com/marketwatch/topstories",
]

# Common large-cap equity tickers to match against headline text.
# This is a representative set; extend as needed.
_KNOWN_TICKERS: Set[str] = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK",
    "UNH", "JNJ", "JPM", "V", "XOM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "AVGO", "COST", "TMO", "DIS", "ACN", "MCD",
    "CSCO", "ABT", "WMT", "BAC", "CRM", "ADBE", "NKE", "NEE", "TXN", "NFLX",
    "PM", "BMY", "RTX", "HON", "IBM", "QCOM", "SBUX", "ORCL", "GS", "MS",
    "AMD", "INTC", "CAT", "BA", "MMM", "GE", "F", "GM", "T", "VZ",
    "PYPL", "SQ", "UBER", "LYFT", "SNAP", "TWTR", "PINS", "SPOT",
    "SPY", "QQQ", "IWM", "XLF", "XLK", "GLD", "SLV", "USO",
}

_TICKER_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")


def _extract_symbols(text: str) -> List[str]:
    candidates = _TICKER_PATTERN.findall(text)
    return [t for t in candidates if t in _KNOWN_TICKERS]


class RssScraper:
    def __init__(self, on_news: Callable[[dict], Awaitable[None]]) -> None:
        self._on_news = on_news
        self._running = False
        self._seen_guids: Set[str] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        try:
            import feedparser  # noqa: F401
        except ImportError:
            print(
                "[News] RSS scraper skipped — feedparser not installed. "
                "Run: pip install feedparser>=6.0.10",
                flush=True,
            )
            return

        self._running = True
        self._loop = asyncio.get_event_loop()
        print("[News] RSS scraper started (90s interval)", flush=True)
        while self._running:
            try:
                await self._poll_all()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[News] RSS scraper error: {exc}", flush=True)
            await asyncio.sleep(_POLL_INTERVAL)

    async def stop(self) -> None:
        self._running = False

    async def _poll_all(self) -> None:
        for feed_url in _RSS_FEEDS:
            try:
                await asyncio.to_thread(self._poll_feed, feed_url, self._loop)
            except Exception as exc:
                print(f"[News] RSS feed {feed_url} error: {exc}", flush=True)

    def _poll_feed(self, feed_url: str, loop: asyncio.AbstractEventLoop) -> None:
        import feedparser

        parsed = feedparser.parse(feed_url)
        entries = parsed.get("entries") or []
        for entry in reversed(entries):
            guid = str(entry.get("id") or entry.get("link") or "")
            if not guid or guid in self._seen_guids:
                continue
            self._seen_guids.add(guid)

            headline = (entry.get("title") or "").strip()
            summary = (entry.get("summary") or "").strip()
            if not headline:
                continue

            combined = f"{headline} {summary}"
            symbols = _extract_symbols(combined)

            item: dict = {
                "id": guid,
                "headline": headline,
                "summary": summary,
                "source": (parsed.feed.get("title") or "rss").strip(),
                "symbols": symbols,
                "asset_class": "equity",
                "published_at": str(entry.get("published") or ""),
                "url": str(entry.get("link") or ""),
            }
            asyncio.run_coroutine_threadsafe(self._on_news(item), loop)

        if len(self._seen_guids) > 1000:
            self._seen_guids = set(list(self._seen_guids)[-1000:])


def create_rss_scraper(
    on_news: Callable[[dict], Awaitable[None]],
) -> RssScraper:
    return RssScraper(on_news)
