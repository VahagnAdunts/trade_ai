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
from typing import Awaitable, Callable, List, Optional, Set

import httpx

_POLL_INTERVAL = 20.0

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
            batch = self._batches[batch_idx]
            tasks = [self._poll_account(handle) for handle in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            batch_idx = (batch_idx + 1) % n_batches
            await asyncio.sleep(_POLL_INTERVAL)

    async def stop(self) -> None:
        self._running = False

    async def _poll_account(self, handle: str) -> None:
        try:
            import feedparser
        except ImportError:
            print(
                "[News] X/Nitter monitor requires 'feedparser'. "
                "Run: pip install feedparser>=6.0.10",
                flush=True,
            )
            return

        for attempt in range(len(NITTER_INSTANCES)):
            idx = (self._instance_idx + attempt) % len(NITTER_INSTANCES)
            base = NITTER_INSTANCES[idx]
            url = f"{base}/{handle}/rss"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(url)
                if resp.status_code >= 400:
                    continue
                feed = await asyncio.to_thread(feedparser.parse, resp.text)
                entries = feed.get("entries") or []
                for entry in reversed(entries):
                    guid = str(entry.get("id") or entry.get("link") or "")
                    if not guid or guid in self._seen_guids:
                        continue
                    self._seen_guids.add(guid)
                    item = self._normalize(entry, handle)
                    if item:
                        await self._on_news(item)
                self._instance_idx = idx
                return
            except Exception:
                continue

        if len(self._seen_guids) > 10000:
            self._seen_guids = set(list(self._seen_guids)[-5000:])

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
