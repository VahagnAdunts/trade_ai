"""
x_monitor.py — Monitor X (Twitter) accounts via free Nitter RSS feeds.

Nitter is an open-source Twitter frontend that provides RSS for any public
account at ``{nitter_instance}/{username}/rss``. No API key or payment needed.

Accounts are split into batches and polled in rotation. Multiple Nitter
instances are tried on failure for resilience.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, List, Optional, Set, Tuple

import httpx

_POLL_INTERVAL = 20.0

# Some Nitter hosts reject default httpx / datacenter-looking clients without a UA.
_RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TredAiNewsBot/1.0; +https://github.com/VahagnAdunts/trade_ai) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
}

NITTER_INSTANCES: List[str] = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.woodland.cafe",
    "https://nitter.1d4.us",
]

MONITORED_ACCOUNTS: List[str] = [
    # ── Breaking news aggregators (fastest for market-moving headlines) ──
    "DeItaone",
    "FirstSquawk",
    "zerohedge",
    "LiveSquawk",
    "Fxhedgers",
    "unusual_whales",
    "financialjuice",
    "Newsquawk",
    "IGSquawk",
    "tier10k",
    "MarketCurrents",
    "TradeTheEvent",

    # ── Major news outlets ──
    "Reuters",
    "AP",
    "Bloomberg",
    "CNBC",
    "WSJ",
    "FT",
    "MarketWatch",
    "business",
    "YahooFinance",
    "CNBCnow",
    "ReutersBiz",
    "WSJmarkets",
    "BreakingNews",
    "BBCBreaking",

    # ── CNBC / Bloomberg journalists ──
    "jimcramer",
    "carlquintanilla",
    "JoeSquawk",
    "BeckyQuick",
    "davidfaber",
    "SaraEisen",
    "LesliePicker",
    "KateRooney",
    "MelissaLeeCNBC",
    "SquawkCNBC",
    "SonaliBasak",
    "EdLudlow",

    # ── Government / Central Banks / Regulators ──
    "federalreserve",
    "USTreasury",
    "SECGov",
    "WhiteHouse",
    "POTUS",
    "SecYellen",
    "FDIC",
    "FDArecalls",

    # ── CEOs / Founders (personal posts can move stocks) ──
    "elonmusk",
    "JeffBezos",
    "BillGates",
    "satyanadella",
    "tim_cook",
    "sundarpichai",
    "LisaSu",
    "BrianChesky",
    "realDonaldTrump",
    "BobIger",
    "MarkZuckerberg",

    # ── Investors / Fund Managers / Analysts ──
    "elerianm",
    "LizAnnSonders",
    "chamath",
    "CathieDWood",
    "carlicahn",
    "AswathDamodaran",
    "RayDalio",
    "BillAckman",
    "mcuban",
    "jimchanos",
    "NourielRoubini",
    "paulkrugman",
    "jasonfurman",
    "LarryMcDonald",

    # ── Crypto influential ──
    "VitalikButerin",
    "cz_binance",
    "brian_armstrong",
    "APompliano",

    # ── Political figures (policy moves markets) ──
    "SpeakerJohnson",
    "LeaderMcConnell",
    "SenSchumer",
    "SenWarren",

    # ── Sector / industry ──
    "OPECnews",
    "IEA",
    "GoldmanSachs",
    "MorganStanley",
    "BlackRock",
]

_ACCOUNTS_PER_BATCH = 10


def _build_batches(accounts: List[str], size: int = _ACCOUNTS_PER_BATCH) -> List[List[str]]:
    return [accounts[i:i + size] for i in range(0, len(accounts), size)]


class XNitterMonitor:
    def __init__(
        self,
        on_news: Callable[[dict], Awaitable[None]],
    ) -> None:
        self._on_news = on_news
        self._running = False
        self._seen_guids: Set[str] = set()
        self._bootstrapped_handles: Set[str] = set()
        self._batches = _build_batches(MONITORED_ACCOUNTS)
        self._instance_idx = 0

    async def start(self) -> None:
        self._running = True
        n_batches = len(self._batches)
        print(
            f"[News] X/Nitter monitor started — {len(MONITORED_ACCOUNTS)} accounts "
            f"in {n_batches} batches, polling every {_POLL_INTERVAL}s",
            flush=True,
        )
        batch_idx = 0
        while self._running:
            try:
                batch = self._batches[batch_idx]
                tasks = [self._poll_account(handle) for handle in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                issues: List[str] = []
                for h, res in zip(batch, results):
                    if isinstance(res, BaseException):
                        issues.append(f"{h}:exc:{type(res).__name__}")
                    else:
                        ok, msg = res
                        if not ok and msg:
                            issues.append(f"{h}:{msg}")
                if issues:
                    preview = ", ".join(issues[:6])
                    more = f" …+{len(issues) - 6}" if len(issues) > 6 else ""
                    print(
                        f"[News] X/Nitter batch {batch_idx}: RSS issues {len(issues)}/{len(batch)} "
                        f"— {preview}{more}",
                        flush=True,
                    )
                if batch_idx == n_batches - 1:
                    primed = len(self._bootstrapped_handles)
                    print(
                        f"[News] X/Nitter heartbeat: finished account rotation "
                        f"({primed}/{len(MONITORED_ACCOUNTS)} handles primed from RSS)",
                        flush=True,
                    )
                batch_idx = (batch_idx + 1) % n_batches
                await asyncio.sleep(_POLL_INTERVAL)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[News] X/Nitter monitor loop error (will retry in 30s): {exc}", flush=True)
                await asyncio.sleep(30.0)

    async def stop(self) -> None:
        self._running = False

    async def _poll_account(self, handle: str) -> Tuple[bool, str]:
        """
        Fetch RSS for one handle. Returns (ok, detail).
        detail is empty on success; on failure a short reason (HTTP code, parse error, etc.).
        """
        try:
            import feedparser
        except ImportError:
            print(
                "[News] X/Nitter monitor requires 'feedparser'. "
                "Run: pip install feedparser>=6.0.10",
                flush=True,
            )
            return False, "feedparser_missing"

        last_detail = "all_instances_failed"
        for attempt in range(len(NITTER_INSTANCES)):
            idx = (self._instance_idx + attempt) % len(NITTER_INSTANCES)
            base = NITTER_INSTANCES[idx].rstrip("/")
            url = f"{base}/{handle}/rss"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(url, headers=_RSS_HEADERS)
                if resp.status_code >= 400:
                    last_detail = f"HTTP{resp.status_code}"
                    continue
                feed = await asyncio.to_thread(feedparser.parse, resp.text)
                entries = feed.get("entries") or []
                if feed.get("bozo") and not entries:
                    bexc = feed.get("bozo_exception")
                    last_detail = f"parse:{type(bexc).__name__}" if bexc else "parse:bozo"
                    continue

                # First successful fetch: remember existing items so we only alert on *new* posts.
                if handle not in self._bootstrapped_handles:
                    for entry in entries:
                        guid = str(entry.get("id") or entry.get("link") or "")
                        if guid:
                            self._seen_guids.add(guid)
                    self._bootstrapped_handles.add(handle)
                    self._instance_idx = idx
                    return True, ""

                for entry in reversed(entries):
                    guid = str(entry.get("id") or entry.get("link") or "")
                    if not guid or guid in self._seen_guids:
                        continue
                    self._seen_guids.add(guid)
                    item = self._normalize(entry, handle)
                    if item:
                        await self._on_news(item)
                self._instance_idx = idx
                return True, ""
            except httpx.HTTPError as exc:
                last_detail = f"httpx:{type(exc).__name__}"
                continue
            except Exception as exc:
                last_detail = f"exc:{type(exc).__name__}"
                continue

        if len(self._seen_guids) > 10000:
            self._seen_guids = set(list(self._seen_guids)[-5000:])

        return False, last_detail

    @staticmethod
    def _normalize(entry: dict, handle: str) -> Optional[dict]:
        title = (entry.get("title") or "").strip()
        if not title:
            return None
        summary = (entry.get("summary") or "").strip()
        link = str(entry.get("link") or "")
        published_raw = entry.get("published") or ""
        published_at = ""
        if published_raw:
            try:
                dt = parsedate_to_datetime(str(published_raw))
                published_at = dt.astimezone(timezone.utc).isoformat()
            except Exception:
                published_at = str(published_raw)
        return {
            "id": f"x_{handle}_{entry.get('id', '')}",
            "headline": title[:280],
            "summary": summary,
            "content": f"{title}\n{summary}".strip(),
            "source": f"x/@{handle}",
            "symbols": [],
            "asset_class": "equity",
            "published_at": published_at,
            "url": link or f"https://x.com/{handle}",
        }


def create_x_monitor(
    on_news: Callable[[dict], Awaitable[None]],
) -> XNitterMonitor:
    return XNitterMonitor(on_news)
