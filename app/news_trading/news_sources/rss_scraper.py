"""
rss_scraper.py — primary-source RSS/Atom feeds (30-second polling interval).

Strategy: monitor sources that PUBLISH news first, before any journalist
writes about it.  Three tiers:

1. Government / Regulatory  — Fed, Treasury, White House, SEC, FDA.
   These are the original source; Bloomberg covers them 30-120s later.
   → All items forwarded (low volume, always high value).

2. Press wires               — PRNewswire, BusinessWire, GlobeNewswire.
   Companies issue earnings, M&A, FDA approvals HERE before reporters
   cover them.
   → Only items where a known ticker appears in headline/summary
     (avoids volume explosion; LLM still handles final relevance check).

3. SEC EDGAR 8-K filings     — material events filed directly with the SEC:
   mergers, earnings surprises, CEO dismissals, bankruptcy warnings.
   → Items forwarded when a known ticker or large company name is found.
"""
from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable, List, Optional, Set, Tuple

_POLL_INTERVAL = 30.0   # seconds — faster than before (was 90s), primary sources

# (label, url, ticker_filter)
# ticker_filter=True  → only forward if a known ticker/company name found
# ticker_filter=False → always forward (government/regulatory, low volume)
_RSS_FEEDS: List[Tuple[str, str, bool]] = [

    # ── Government / Regulatory — PRIMARY source, always forward ─────────
    # Posted here BEFORE Bloomberg/Reuters writes about it.
    ("Federal Reserve",
     "https://www.federalreserve.gov/feeds/press_all.xml",
     False),

    ("White House",
     "https://www.whitehouse.gov/feed/",
     False),

    ("US Treasury",
     "https://home.treasury.gov/news/press-releases/rss",
     False),

    ("SEC Press Releases",
     "https://www.sec.gov/news/pressreleases.rss",
     False),

    ("FDA",
     "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/"
     "press-releases/rss.xml",
     False),

    # ── SEC EDGAR 8-K filings — material corporate events ─────────────────
    # Mergers, earnings surprises, CEO changes, bankruptcies.
    # Filed directly; journalists write about these after.
    ("SEC EDGAR 8-K",
     "https://www.sec.gov/cgi-bin/browse-edgar"
     "?action=getcurrent&type=8-K&dateb=&owner=include"
     "&count=40&search_text=&output=atom",
     True),

    # ── Press wires — companies issue announcements here first ────────────
    # ticker_filter=True keeps volume manageable; LLM does final check.
    ("PRNewswire",
     "https://www.prnewswire.com/rss/news-releases-list.rss",
     True),

    ("BusinessWire",
     "https://feed.businesswire.com/rss/home/?rss=G7",
     True),

    ("GlobeNewswire",
     "https://www.globenewswire.com/RssFeed/subjectcode/14-Financial",
     True),
]

# ---------------------------------------------------------------------------
# Known large-cap tickers for fast symbol extraction from text
# ---------------------------------------------------------------------------
_KNOWN_TICKERS: Set[str] = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK",
    "UNH", "JNJ", "JPM", "V", "XOM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "AVGO", "COST", "TMO", "DIS", "ACN", "MCD",
    "CSCO", "ABT", "WMT", "BAC", "CRM", "ADBE", "NKE", "NEE", "TXN", "NFLX",
    "PM", "BMY", "RTX", "HON", "IBM", "QCOM", "SBUX", "ORCL", "GS", "MS",
    "AMD", "INTC", "CAT", "BA", "MMM", "GE", "F", "GM", "T", "VZ",
    "PYPL", "SQ", "UBER", "LYFT", "SNAP", "PINS", "SPOT", "COIN",
    "SPY", "QQQ", "IWM", "XLF", "XLK", "GLD", "SLV", "USO",
    "PFE", "MRNA", "BNTX", "BMY", "GILD", "BIIB", "REGN", "VRTX", "AZN",
    "C", "WFC", "AXP", "BLK", "SCHW", "CME", "ICE", "CBOE",
    "NFLX", "PARA", "WBD", "CMCSA", "CHTR",
}

# Company name fragments → ticker (for press wire / EDGAR headline matching)
# Uppercase fragment must appear as a whole word in the uppercased headline.
_COMPANY_NAME_MAP: dict = {
    "APPLE": "AAPL", "MICROSOFT": "MSFT", "AMAZON": "AMZN",
    "ALPHABET": "GOOGL", "GOOGLE": "GOOGL", "NVIDIA": "NVDA",
    "META": "META", "FACEBOOK": "META", "TESLA": "TSLA",
    "BERKSHIRE": "BRK", "UNITEDHEALTH": "UNH",
    "JOHNSON": "JNJ", "JPMORGAN": "JPM", "VISA": "V",
    "EXXON": "XOM", "PROCTER": "PG", "MASTERCARD": "MA",
    "HOME DEPOT": "HD", "CHEVRON": "CVX", "MERCK": "MRK",
    "ABBVIE": "ABBV", "ELI LILLY": "LLY", "PEPSICO": "PEP",
    "COCA-COLA": "KO", "BROADCOM": "AVGO", "COSTCO": "COST",
    "THERMO": "TMO", "DISNEY": "DIS", "ACCENTURE": "ACN",
    "MCDONALD": "MCD", "CISCO": "CSCO", "ABBOTT": "ABT",
    "WALMART": "WMT", "BANK OF AMERICA": "BAC", "SALESFORCE": "CRM",
    "ADOBE": "ADBE", "NIKE": "NKE", "NEXTERA": "NEE",
    "TEXAS INSTRUMENTS": "TXN", "NETFLIX": "NFLX",
    "PHILIP MORRIS": "PM", "BRISTOL": "BMY", "RAYTHEON": "RTX",
    "HONEYWELL": "HON", "INTEL": "INTC", "QUALCOMM": "QCOM",
    "STARBUCKS": "SBUX", "ORACLE": "ORCL",
    "GOLDMAN": "GS", "MORGAN STANLEY": "MS",
    "ADVANCED MICRO": "AMD", "CATERPILLAR": "CAT", "BOEING": "BA",
    "GENERAL ELECTRIC": "GE", "FORD": "F", "GENERAL MOTORS": "GM",
    "AT&T": "T", "VERIZON": "VZ",
    "PAYPAL": "PYPL", "UBER": "UBER",
    "PFIZER": "PFE", "MODERNA": "MRNA", "GILEAD": "GILD",
    "BIOGEN": "BIIB", "REGENERON": "REGN", "VERTEX": "VRTX",
    "CITIGROUP": "C", "WELLS FARGO": "WFC", "AMEX": "AXP",
    "BLACKROCK": "BLK", "SCHWAB": "SCHW", "COINBASE": "COIN",
}

_TICKER_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")
_WORD_BOUNDARY = re.compile(r"\b{}\b")


def _extract_symbols(text: str) -> List[str]:
    """Extract tickers from text via direct pattern match."""
    candidates = _TICKER_PATTERN.findall(text)
    return [t for t in candidates if t in _KNOWN_TICKERS]


def _has_known_company(text: str) -> List[str]:
    """
    Return list of tickers found by company name match.
    Used for press wire / EDGAR items where ticker may not appear literally.
    """
    upper = text.upper()
    found = []
    for name, ticker in _COMPANY_NAME_MAP.items():
        if name in upper:
            found.append(ticker)
    return found


def _should_forward(text: str, ticker_filter: bool) -> Tuple[bool, List[str]]:
    """
    Decide whether to forward a feed item and return extracted symbols.
    - ticker_filter=False: always forward (government/regulatory feeds)
    - ticker_filter=True:  only forward if a company/ticker is found
    """
    symbols = list(set(_extract_symbols(text) + _has_known_company(text)))
    if not ticker_filter:
        return True, symbols
    return bool(symbols), symbols


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
        feed_labels = ", ".join(label for label, _, _ in _RSS_FEEDS)
        print(
            f"[News] RSS scraper started ({_POLL_INTERVAL}s interval) — "
            f"feeds: {feed_labels}",
            flush=True,
        )
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
        for label, feed_url, ticker_filter in _RSS_FEEDS:
            try:
                await asyncio.to_thread(
                    self._poll_feed, label, feed_url, ticker_filter, self._loop
                )
            except Exception as exc:
                print(f"[News] RSS feed '{label}' error: {exc}", flush=True)

    def _poll_feed(
        self,
        label: str,
        feed_url: str,
        ticker_filter: bool,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
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
            forward, symbols = _should_forward(combined, ticker_filter)
            if not forward:
                continue

            item: dict = {
                "id": guid,
                "headline": headline,
                "summary": summary,
                "source": label,
                "symbols": symbols,
                "asset_class": "equity",
                "published_at": str(entry.get("published") or ""),
                "url": str(entry.get("link") or ""),
            }
            asyncio.run_coroutine_threadsafe(self._on_news(item), loop)

        if len(self._seen_guids) > 5000:
            self._seen_guids = set(list(self._seen_guids)[-2500:])


def create_rss_scraper(
    on_news: Callable[[dict], Awaitable[None]],
) -> RssScraper:
    return RssScraper(on_news)
