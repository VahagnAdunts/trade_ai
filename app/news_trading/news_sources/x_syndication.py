"""
x_syndication.py — Monitor X/Twitter accounts via the free syndication API.

X's syndication endpoint (used by embedded timeline widgets on websites)
returns full tweet JSON in a __NEXT_DATA__ script tag.  No API key, no
subscription, no OAuth required.

Endpoint: GET https://syndication.twitter.com/srv/timeline-profile/screen-name/{handle}

Rate limits exist (~15 req/burst, then HTTP 429) so accounts are polled in
small batches with delays between them.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

import httpx

_SYND_URL = "https://syndication.twitter.com/srv/timeline-profile/screen-name"
_POLL_INTERVAL = 120.0         # seconds between batch cycles (X syndication is aggressively limited)
_INTER_BATCH_DELAY = 20.0      # seconds between requests within a batch
_ACCOUNTS_PER_BATCH = 2        # smaller batches reduce burst 429s from shared IP
_REQUEST_TIMEOUT = 15.0
_ACCOUNT_BACKOFF = 600.0       # seconds to skip an account after 429 (per-handle)
_BATCH_429_COOLDOWN = 120.0    # extra pause when multiple 429s in one batch (shared limiter)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://platform.twitter.com/",
}

# ---------------------------------------------------------------------------
# Top X accounts with most market impact.
#
# These are accounts that routinely move stock prices with their posts.
# The syndication API works for any public X account.
# ---------------------------------------------------------------------------
MONITORED_X_ACCOUNTS: List[Tuple[str, str]] = [
    # ── Breaking news wires (fastest for market-moving headlines) ─────────
    ("DeItaone", "DeItaone"),
    ("FirstSquawk", "FirstSquawk"),
    ("LiveSquawk", "LiveSquawk"),
    ("zerohedge", "zerohedge"),
    ("Fxhedgers", "Fxhedgers"),
    ("Unusual Whales", "unusual_whales"),
    ("FinancialJuice", "financialjuice"),
    ("Newsquawk", "Newsquawk"),
    ("Tier10K", "tier10k"),
    ("MarketCurrents", "MarketCurrents"),

    # ── CEOs whose posts move stocks ─────────────────────────────────────
    ("Elon Musk", "elonmusk"),
    ("Tim Cook", "tim_cook"),
    ("Satya Nadella", "satyanadella"),
    ("Lisa Su", "LisaSu"),
    ("Jensen Huang", "JensenHuang"),

    # ── Government / Central Banks ────────────────────────────────────────
    ("White House", "WhiteHouse"),
    ("Federal Reserve", "federalreserve"),
    ("SEC", "SECGov"),
    ("US Treasury", "USTreasury"),
    ("POTUS", "POTUS"),

    # ── Investors / major market voices ───────────────────────────────────
    ("Bill Ackman", "BillAckman"),
    ("Cathie Wood", "CathieDWood"),
    ("Chamath", "chamath"),
    ("Mark Cuban", "mcuban"),
    ("Jim Cramer", "jimcramer"),
]


class XSyndicationMonitor:
    def __init__(
        self,
        on_news: Callable[[dict], Awaitable[None]],
    ) -> None:
        self._on_news = on_news
        self._running = False
        self._seen_ids: Set[str] = set()
        self._bootstrapped: Set[str] = set()
        self._account_backoff: Dict[str, float] = {}  # handle -> monotonic backoff time

    async def start(self) -> None:
        self._running = True
        accounts = MONITORED_X_ACCOUNTS
        batches = _build_batches(accounts, _ACCOUNTS_PER_BATCH)
        print(
            f"[News] X/Syndication monitor started — {len(accounts)} accounts "
            f"in {len(batches)} batches, polling every {_POLL_INTERVAL}s",
            flush=True,
        )

        batch_idx = 0
        while self._running:
            try:
                batch = batches[batch_idx]
                issues: List[str] = []

                for label, handle in batch:
                    if not self._running:
                        break
                    ok, detail = await self._poll_account(label, handle)
                    if not ok and detail and detail != "rate_backoff":
                        issues.append(f"{handle}:{detail}")
                    # Space out requests — syndication is easy to 429 by IP + burst
                    await asyncio.sleep(_INTER_BATCH_DELAY)

                if issues:
                    preview = ", ".join(issues[:4])
                    more = f" …+{len(issues) - 4}" if len(issues) > 4 else ""
                    print(
                        f"[News] X/Syndication batch {batch_idx}: "
                        f"{len(issues)}/{len(batch)} issues — {preview}{more}",
                        flush=True,
                    )
                    n429 = sum(1 for x in issues if "HTTP429" in x)
                    if n429 >= 2 or n429 == len(batch):
                        print(
                            f"[News] X/Syndication: {n429}× HTTP429 in batch — "
                            f"extra cooldown {_BATCH_429_COOLDOWN:.0f}s",
                            flush=True,
                        )
                        await asyncio.sleep(_BATCH_429_COOLDOWN)

                if batch_idx == len(batches) - 1:
                    primed = len(self._bootstrapped)
                    print(
                        f"[News] X/Syndication heartbeat: rotation complete "
                        f"({primed}/{len(accounts)} primed)",
                        flush=True,
                    )

                batch_idx = (batch_idx + 1) % len(batches)
                await asyncio.sleep(_POLL_INTERVAL)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(
                    f"[News] X/Syndication monitor loop error (retry in 30s): {exc}",
                    flush=True,
                )
                await asyncio.sleep(30.0)

    async def stop(self) -> None:
        self._running = False

    async def _poll_account(
        self, label: str, handle: str
    ) -> Tuple[bool, str]:
        """Fetch timeline for one handle. Returns (ok, detail)."""
        # Respect per-account rate-limit backoff
        now_mono = time.monotonic()
        if now_mono < self._account_backoff.get(handle, 0.0):
            return False, "rate_backoff"

        try:
            async with httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT,
                follow_redirects=True,
                trust_env=True,
            ) as client:
                resp = await client.get(
                    f"{_SYND_URL}/{handle}",
                    headers=_HEADERS,
                )

                if resp.status_code == 429:
                    wait_s = float(_ACCOUNT_BACKOFF)
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            wait_s = max(wait_s, float(ra))
                        except ValueError:
                            pass
                    self._account_backoff[handle] = time.monotonic() + wait_s
                    return False, "HTTP429"

                if resp.status_code != 200:
                    return False, f"HTTP{resp.status_code}"

                tweets = _parse_syndication_html(resp.text)
                if tweets is None:
                    return False, "no_data"

                # Bootstrap: mark existing tweets as seen
                if handle not in self._bootstrapped:
                    for tw in tweets:
                        self._seen_ids.add(tw["id"])
                    self._bootstrapped.add(handle)
                    return True, ""

                # Process new tweets (oldest first)
                for tw in reversed(tweets):
                    if tw["id"] in self._seen_ids:
                        continue
                    self._seen_ids.add(tw["id"])

                    item = _normalize_tweet(tw, label, handle)
                    if item:
                        await self._on_news(item)

                # Prevent unbounded memory
                if len(self._seen_ids) > 10_000:
                    self._seen_ids = set(list(self._seen_ids)[-5000:])

                return True, ""

        except httpx.TimeoutException:
            return False, "timeout"
        except Exception as exc:
            return False, f"exc:{type(exc).__name__}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_NEXT_DATA_RE = re.compile(
    r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL
)


def _parse_syndication_html(html: str) -> Optional[List[Dict]]:
    """Extract tweet list from syndication __NEXT_DATA__ JSON."""
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None

    entries = (
        data.get("props", {})
        .get("pageProps", {})
        .get("timeline", {})
        .get("entries", [])
    )

    tweets: List[Dict] = []
    for entry in entries:
        if entry.get("type") != "tweet":
            continue
        tw = entry.get("content", {}).get("tweet", {})
        tweet_id = entry.get("entry_id", "")  # e.g. "tweet-123456789"
        if tweet_id.startswith("tweet-"):
            tweet_id = tweet_id[6:]

        text = tw.get("full_text") or tw.get("text") or ""
        created_at = tw.get("created_at", "")
        screen_name = tw.get("user", {}).get("screen_name", "")
        display_name = tw.get("user", {}).get("name", "")

        if text and tweet_id:
            tweets.append({
                "id": tweet_id,
                "text": text,
                "created_at": created_at,
                "screen_name": screen_name,
                "display_name": display_name,
            })

    return tweets


def _normalize_tweet(
    tw: Dict, label: str, handle: str
) -> Optional[Dict]:
    """Convert parsed tweet dict to standard news item format."""
    text = tw.get("text", "").strip()
    if not text:
        return None

    # Unescape HTML entities from syndication
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )

    # Parse created_at to ISO format
    created_at_raw = tw.get("created_at", "")
    published_at = ""
    if created_at_raw:
        try:
            dt = parsedate_to_datetime(created_at_raw)
            published_at = dt.astimezone(timezone.utc).isoformat()
        except Exception:
            published_at = created_at_raw

    tweet_id = tw["id"]
    display = tw.get("display_name") or label
    screen = tw.get("screen_name") or handle

    return {
        "id": f"xsyn_{tweet_id}",
        "headline": f"{display}: {text[:250]}",
        "summary": text,
        "content": text,
        "source": f"x/@{screen}",
        "symbols": [],          # fast LLM will extract tickers
        "asset_class": "equity",
        "published_at": published_at,
        "url": f"https://x.com/{screen}/status/{tweet_id}",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_batches(
    items: List[Tuple[str, str]], size: int
) -> List[List[Tuple[str, str]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def create_x_syndication_monitor(
    on_news: Callable[[dict], Awaitable[None]],
) -> XSyndicationMonitor:
    return XSyndicationMonitor(on_news)
