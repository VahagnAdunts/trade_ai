"""
x_monitor.py — Monitor X (Twitter) accounts via free Nitter RSS feeds.

Nitter is an open-source Twitter frontend that provides RSS for any public
account at ``{nitter_instance}/{username}/rss``. No API key or payment needed.

Accounts are split into batches and polled in rotation. Multiple Nitter
instances are tried on failure for resilience.
"""
from __future__ import annotations

import asyncio
import os
import socket
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, List, Optional, Set, Tuple
from urllib.parse import urlparse

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

# Nitter-style RSS at ``{base}/{username}/rss``. Order matters.
# rss.xcancel.com rejects normal browser UAs (400) or serves a non-feed "whitelist" stub (200).
# Several public Nitter hosts return 403 anti-bot HTML or 404; nitter.net currently serves real RSS.
DEFAULT_NITTER_INSTANCES: List[str] = [
    "https://nitter.net",
    "https://nitter.poast.org",
]


def _resolve_nitter_bases() -> List[str]:
    """Comma-separated https origins in NITTER_INSTANCES, else built-in defaults."""
    raw = (os.getenv("NITTER_INSTANCES") or "").strip()
    if not raw:
        return list(DEFAULT_NITTER_INSTANCES)
    bases = [b.strip().rstrip("/") for b in raw.split(",") if b.strip()]
    return bases if bases else list(DEFAULT_NITTER_INSTANCES)


def _rss_response_rejected(status_code: int, text: str, content_type: str) -> Optional[str]:
    """
    Some hosts return 200 HTML/placeholder instead of RSS. Return a short reason to try the next host.
    """
    if status_code != 200:
        return None
    sample = (text or "")[:3000].lower()
    ct = (content_type or "").lower()
    if "only works inside an rss client" in sample:
        return "rss_placeholder_400_style"
    if "not yet whitelisted" in sample:
        return "rss_whitelist_stub"
    if "verifying your browser" in sample:
        return "browser_challenge_html"
    if "<rss" not in sample and "application/rss" not in ct and "application/xml" not in ct:
        if "<html" in sample[:800]:
            return "HTML_not_rss"
    return None


def _short_err(exc: Exception, limit: int = 72) -> str:
    s = str(exc).strip().replace("\n", " ")
    if not s:
        return ""
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _proxy_env_active() -> bool:
    return any((os.getenv(k) or "").strip() for k in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"))


def _log_startup_network_hints(nitter_bases: List[str]) -> None:
    """One-time DNS check + proxy hint (errno -5 / 'No address' = DNS failure)."""
    if _proxy_env_active():
        print(
            "[News] X/Nitter: HTTPS_PROXY / HTTP_PROXY / ALL_PROXY is set. "
            "If you see 'No address associated with hostname', that name may be the "
            "**proxy** host (mis-typed or dead) — unset these vars when not using a proxy.",
            flush=True,
        )
    for base in nitter_bases:
        raw = base if "://" in base else f"https://{base}"
        host = urlparse(raw).hostname
        if not host:
            print(f"[News] X/Nitter DNS: invalid base URL {base!r}", flush=True)
            continue
        try:
            infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            uniq = sorted({x[4][0] for x in infos})
            prev = ", ".join(uniq[:4])
            extra = f" (+{len(uniq) - 4} more)" if len(uniq) > 4 else ""
            print(f"[News] X/Nitter DNS OK: {host} -> {prev}{extra}", flush=True)
        except OSError as exc:
            print(
                f"[News] X/Nitter DNS FAILED for {host}: {exc}. "
                f"Replace or extend NITTER_INSTANCES with hosts that resolve from this network.",
                flush=True,
            )


def print_monitored_x_account_catalog(primary_nitter_base: str) -> None:
    """Print every monitored X handle and the RSS URL shape (stdout / host logs)."""
    base = primary_nitter_base.rstrip("/")
    handles = MONITORED_ACCOUNTS
    n = len(handles)
    if n == 0:
        return
    ex = handles[0]
    print(
        f"[News] X/Nitter catalog: {n} X accounts. "
        f"RSS: {base}/<handle>/rss   Web: https://x.com/<handle>",
        flush=True,
    )
    print(f"[News] X/Nitter example: {base}/{ex}/rss", flush=True)
    per_line = 8
    for i in range(0, n, per_line):
        row = handles[i : i + per_line]
        print("[News] X/Nitter   " + "  ".join(row), flush=True)


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
        self._nitter_bases = _resolve_nitter_bases()
        self._instance_idx = 0

    async def start(self) -> None:
        self._running = True
        n_batches = len(self._batches)
        custom = bool((os.getenv("NITTER_INSTANCES") or "").strip())
        src = "NITTER_INSTANCES env" if custom else "default host list"
        proxy_note = "proxy env active (trust_env)" if _proxy_env_active() else "no proxy env"
        print(
            f"[News] X/Nitter monitor started — {len(MONITORED_ACCOUNTS)} accounts "
            f"in {n_batches} batches, polling every {_POLL_INTERVAL}s "
            f"({len(self._nitter_bases)} base URL(s) from {src}; {proxy_note})",
            flush=True,
        )
        _log_startup_network_hints(self._nitter_bases)
        if self._nitter_bases:
            print_monitored_x_account_catalog(self._nitter_bases[0])
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
        bases = self._nitter_bases
        for attempt in range(len(bases)):
            idx = (self._instance_idx + attempt) % len(bases)
            base = bases[idx].rstrip("/")
            url = f"{base}/{handle}/rss"
            try:
                timeout = httpx.Timeout(25.0, connect=20.0)
                async with httpx.AsyncClient(
                    timeout=timeout,
                    trust_env=True,
                    follow_redirects=True,
                ) as client:
                    resp = await client.get(url, headers=_RSS_HEADERS)
                if resp.status_code >= 400:
                    last_detail = f"HTTP{resp.status_code}"
                    continue
                ctype = resp.headers.get("content-type") or ""
                reject = _rss_response_rejected(resp.status_code, resp.text, ctype)
                if reject:
                    last_detail = reject
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
                tail = _short_err(exc)
                last_detail = (
                    f"httpx:{type(exc).__name__}:{tail}" if tail else f"httpx:{type(exc).__name__}"
                )
                continue
            except Exception as exc:
                tail = _short_err(exc)
                last_detail = f"exc:{type(exc).__name__}:{tail}" if tail else f"exc:{type(exc).__name__}"
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
