"""
bluesky_monitor.py — Monitor Bluesky accounts via the free AT Protocol public API.

The AT Protocol provides completely free, unauthenticated read access to any
public account feed.  No API key, subscription, or payment required.

Replaces the defunct Nitter/X approach: Nitter instances are systematically
blocked by X (HTTP 403/404 on all public hosts).  Bluesky's open protocol
has no such restrictions.

Candidate handles are verified at startup; non-existent accounts are
silently skipped.  The list can be extended via the BLUESKY_ACCOUNTS env var.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Awaitable, Callable, List, Optional, Set, Tuple

import httpx

_API_BASE = "https://public.api.bsky.app/xrpc"
_POLL_INTERVAL = 15.0          # seconds between batches
_ACCOUNTS_PER_BATCH = 10
_RESOLVE_TIMEOUT = 10.0
_FEED_TIMEOUT = 12.0
_RESOLVE_CONCURRENCY = 5       # parallel profile lookups at startup

# ---------------------------------------------------------------------------
# Candidate accounts to monitor — (label, bluesky_handle).
#
# Handles are verified at startup via getProfile; invalid ones are skipped.
# Extend at runtime with the BLUESKY_ACCOUNTS env var (comma-separated handles).
# ---------------------------------------------------------------------------
BLUESKY_CANDIDATES: List[Tuple[str, str]] = [
    # =====================================================================
    # PRIMARY SOURCE accounts only — people/services that CREATE news,
    # not journalists or media outlets that REPORT it.
    #
    # Media outlets (Reuters, Bloomberg, CNBC, WSJ etc.) post on Bluesky
    # AFTER their articles are published — too late for trading.
    # =====================================================================

    # ── Real-time market alert services (squawk — fastest on Bluesky) ────
    # These post raw market-moving alerts, not articles.
    ("Unusual Whales", "unusualwhales.bsky.social"),      # options flow + dark pool
    ("FinancialJuice", "financialjuice.bsky.social"),     # fastest breaking alerts
    ("Newsquawk", "newsquawk.bsky.social"),               # audio squawk service
    ("Tier10K", "tier10k.bsky.social"),                   # real-time market data

    # ── CEOs / Founders (their posts ARE the news) ────────────────────────
    ("Bill Gates", "billgates.bsky.social"),
    ("Pat Gelsinger", "patgelsinger.bsky.social"),        # Intel CEO
    ("Brian Armstrong", "brianarmstrong.bsky.social"),    # Coinbase CEO

    # ── Investors / Fund Managers ────────────────────────────────────────
    # These make direct market-moving statements, not just commentary.
    ("Mark Cuban", "mcuban.bsky.social"),
    ("Mohamed El-Erian", "elerianm.bsky.social"),         # ex-PIMCO CEO, economist
    ("Liz Ann Sonders", "lizannsonders.bsky.social"),     # Schwab Chief Investment Strategist
    ("Aswath Damodaran", "aswathdamodaran.bsky.social"),  # NYU valuation expert
    ("Jason Furman", "jasonfurman.bsky.social"),          # ex-Obama CEA Chair

    # ── Political figures (policy = market-moving) ────────────────────────
    ("AOC", "aoc.bsky.social"),

    # ── Squawk / alert services that may join Bluesky ─────────────────────
    ("Fxhedgers", "fxhedgers.bsky.social"),
    ("DeItaone", "deitaone.bsky.social"),
    ("FirstSquawk", "firstsquawk.bsky.social"),
    ("LiveSquawk", "livesquawk.bsky.social"),
    ("Watcher Guru", "watcherguru.bsky.social"),
    ("IGSquawk", "igsquawk.bsky.social"),
    ("StockMKTNewz", "stockmktnewz.bsky.social"),
    ("MarketCurrents", "marketcurrents.bsky.social"),
    ("zerohedge", "zerohedge.bsky.social"),

    # ── Government / Regulatory officials ────────────────────────────────
    ("Federal Reserve", "federalreserve.bsky.social"),
    ("US Treasury", "ustreasury.bsky.social"),
    ("SEC", "secgov.bsky.social"),
    ("POTUS", "potus.bsky.social"),
    ("Speaker Johnson", "speakerjohnson.bsky.social"),

    # ── CEOs not yet confirmed on Bluesky ────────────────────────────────
    ("Tim Cook", "timcook.bsky.social"),
    ("Satya Nadella", "satyanadella.bsky.social"),
    ("Lisa Su", "lisasu.bsky.social"),
    ("Jensen Huang", "jensenhuang.bsky.social"),
    ("Sundar Pichai", "sundarpichai.bsky.social"),
    ("Jeff Bezos", "jeffbezos.bsky.social"),
    ("Elon Musk", "elonmusk.bsky.social"),

    # ── Investors / activists ─────────────────────────────────────────────
    ("Cathie Wood", "cathiewood.bsky.social"),
    ("Bill Ackman", "billackman.bsky.social"),
    ("Ray Dalio", "raydalio.bsky.social"),
    ("Carl Icahn", "carlicahn.bsky.social"),
    ("Paul Krugman", "paulkrugman.bsky.social"),
    ("Chamath", "chamath.bsky.social"),

    # ── Crypto founders (direct market impact) ────────────────────────────
    ("Vitalik Buterin", "vitalikbuterin.bsky.social"),
    ("CZ Binance", "czbinance.bsky.social"),
]


def _extra_accounts_from_env() -> List[Tuple[str, str]]:
    """Parse ``BLUESKY_ACCOUNTS`` env var — comma-separated Bluesky handles."""
    raw = (os.getenv("BLUESKY_ACCOUNTS") or "").strip()
    if not raw:
        return []
    extras: List[Tuple[str, str]] = []
    for handle in raw.split(","):
        handle = handle.strip()
        if handle:
            label = handle.split(".")[0]
            extras.append((label, handle))
    return extras


class BlueskyMonitor:
    def __init__(
        self,
        on_news: Callable[[dict], Awaitable[None]],
    ) -> None:
        self._on_news = on_news
        self._running = False
        self._seen_uris: Set[str] = set()
        self._resolved: List[Tuple[str, str]] = []  # (label, handle)
        self._bootstrapped: Set[str] = set()

    # ── public interface ──────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        print("[News] Bluesky monitor starting — resolving accounts...", flush=True)

        candidates = list(BLUESKY_CANDIDATES) + _extra_accounts_from_env()
        self._resolved = await self._resolve_accounts(candidates)

        if not self._resolved:
            print(
                "[News] Bluesky monitor: no valid accounts resolved — "
                "check BLUESKY_ACCOUNTS or built-in list.",
                flush=True,
            )
            # Stay alive and retry periodically in case of transient API issue
            while self._running:
                await asyncio.sleep(300)
                self._resolved = await self._resolve_accounts(candidates)
                if self._resolved:
                    break
            if not self._resolved:
                return

        batches = _build_batches(self._resolved, _ACCOUNTS_PER_BATCH)
        handles_preview = ", ".join(h for _, h in self._resolved[:8])
        more = f" …+{len(self._resolved) - 8}" if len(self._resolved) > 8 else ""
        print(
            f"[News] Bluesky monitor active — {len(self._resolved)} accounts "
            f"in {len(batches)} batches, polling every {_POLL_INTERVAL}s\n"
            f"[News] Bluesky accounts: {handles_preview}{more}",
            flush=True,
        )

        batch_idx = 0
        while self._running:
            try:
                batch = batches[batch_idx]
                tasks = [self._poll_account(label, handle) for label, handle in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                issues: List[str] = []
                for (label, _), res in zip(batch, results):
                    if isinstance(res, BaseException):
                        issues.append(f"{label}:{type(res).__name__}")
                    elif isinstance(res, tuple):
                        ok, detail = res
                        if not ok and detail:
                            issues.append(f"{label}:{detail}")

                if issues:
                    preview = ", ".join(issues[:5])
                    more_i = f" …+{len(issues) - 5}" if len(issues) > 5 else ""
                    print(
                        f"[News] Bluesky batch {batch_idx}: {len(issues)}/{len(batch)} "
                        f"issues — {preview}{more_i}",
                        flush=True,
                    )

                if batch_idx == len(batches) - 1:
                    primed = len(self._bootstrapped)
                    print(
                        f"[News] Bluesky heartbeat: rotation complete "
                        f"({primed}/{len(self._resolved)} primed)",
                        flush=True,
                    )

                batch_idx = (batch_idx + 1) % len(batches)
                await asyncio.sleep(_POLL_INTERVAL)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(
                    f"[News] Bluesky monitor loop error (retry in 30s): {exc}",
                    flush=True,
                )
                await asyncio.sleep(30.0)

    async def stop(self) -> None:
        self._running = False

    # ── startup account resolution ────────────────────────────────────────────

    async def _resolve_accounts(
        self, candidates: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Verify which candidate handles exist on Bluesky via getProfile."""
        resolved: List[Tuple[str, str]] = []
        seen_handles: Set[str] = set()

        # Deduplicate candidates (same handle may appear under different labels)
        unique: List[Tuple[str, str]] = []
        for label, handle in candidates:
            h_lower = handle.lower()
            if h_lower not in seen_handles:
                seen_handles.add(h_lower)
                unique.append((label, handle))

        sem = asyncio.Semaphore(_RESOLVE_CONCURRENCY)

        async def _check(label: str, handle: str) -> Optional[Tuple[str, str]]:
            async with sem:
                try:
                    async with httpx.AsyncClient(timeout=_RESOLVE_TIMEOUT) as client:
                        resp = await client.get(
                            f"{_API_BASE}/app.bsky.actor.getProfile",
                            params={"actor": handle},
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            followers = data.get("followersCount", 0)
                            posts = data.get("postsCount", 0)
                            # Skip accounts with no posts or very few followers
                            # (likely parody/placeholder accounts)
                            if posts == 0 or followers < 100:
                                return None
                            display = data.get("displayName") or label
                            return (display, handle)
                except Exception:
                    pass
                return None

        tasks = [_check(label, handle) for label, handle in unique]
        results = await asyncio.gather(*tasks)

        for r in results:
            if r is not None:
                resolved.append(r)

        found = len(resolved)
        total = len(unique)
        print(f"[News] Bluesky resolved {found}/{total} accounts", flush=True)
        if resolved:
            names = ", ".join(f"{label} (@{h})" for label, h in resolved[:12])
            more = f" …+{len(resolved) - 12}" if len(resolved) > 12 else ""
            print(f"[News] Bluesky active: {names}{more}", flush=True)
        return resolved

    # ── per-account polling ───────────────────────────────────────────────────

    async def _poll_account(
        self, label: str, handle: str
    ) -> Tuple[bool, str]:
        """
        Fetch feed for one handle.  Returns ``(ok, detail)``.
        On first successful fetch, existing posts are marked as seen (bootstrap).
        """
        try:
            async with httpx.AsyncClient(timeout=_FEED_TIMEOUT) as client:
                resp = await client.get(
                    f"{_API_BASE}/app.bsky.feed.getAuthorFeed",
                    params={
                        "actor": handle,
                        "limit": "15",
                        "filter": "posts_no_replies",
                    },
                )
                if resp.status_code == 429:
                    return False, "rate_limited"
                if resp.status_code != 200:
                    return False, f"HTTP{resp.status_code}"

                data = resp.json()
                feed = data.get("feed") or []

                # Bootstrap: mark existing posts as seen so we only alert on NEW posts
                if handle not in self._bootstrapped:
                    for item in feed:
                        post = item.get("post", {})
                        uri = post.get("uri", "")
                        if uri:
                            self._seen_uris.add(uri)
                    self._bootstrapped.add(handle)
                    return True, ""

                # Process new posts (oldest first for natural ordering)
                for item in reversed(feed):
                    post = item.get("post", {})
                    uri = post.get("uri", "")
                    if not uri or uri in self._seen_uris:
                        continue
                    self._seen_uris.add(uri)

                    # Skip reposts — we want original posts only
                    if item.get("reason"):
                        continue

                    news_item = self._normalize(post, label, handle)
                    if news_item:
                        await self._on_news(news_item)

                # Prevent unbounded memory growth
                if len(self._seen_uris) > 10_000:
                    self._seen_uris = set(list(self._seen_uris)[-5000:])

                return True, ""

        except httpx.TimeoutException:
            return False, "timeout"
        except Exception as exc:
            return False, f"exc:{type(exc).__name__}"

    # ── normalization ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(
        post: dict, label: str, handle: str
    ) -> Optional[dict]:
        record = post.get("record") or {}
        text = (record.get("text") or "").strip()
        if not text:
            return None

        author = post.get("author") or {}
        display_name = author.get("displayName") or label
        created_at = record.get("createdAt") or post.get("indexedAt") or ""
        uri = post.get("uri", "")

        # Convert AT URI → web URL
        # at://did:plc:xxx/app.bsky.feed.post/yyy → bsky.app/profile/handle/post/yyy
        web_url = ""
        if uri and "/app.bsky.feed.post/" in uri:
            post_id = uri.split("/app.bsky.feed.post/")[-1]
            web_url = f"https://bsky.app/profile/{handle}/post/{post_id}"

        return {
            "id": f"bsky_{uri}",
            "headline": f"{display_name}: {text[:250]}",
            "summary": text,
            "content": text,
            "source": f"bluesky/@{handle}",
            "symbols": [],          # fast LLM will extract tickers
            "asset_class": "equity",
            "published_at": created_at,
            "url": web_url,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_batches(
    items: List[Tuple[str, str]], size: int
) -> List[List[Tuple[str, str]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def create_bluesky_monitor(
    on_news: Callable[[dict], Awaitable[None]],
) -> BlueskyMonitor:
    return BlueskyMonitor(on_news)
