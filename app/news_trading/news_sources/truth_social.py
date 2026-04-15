"""
truth_social.py — Poll Truth Social for posts from key political figures.

Truth Social exposes a Mastodon-compatible public API.
Polls account timelines every 30 seconds and normalizes to NewsItem dicts.
"""
from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable, Dict, List, Optional, Set

import httpx

_BASE_URL = "https://truthsocial.com/api/v1"
_POLL_INTERVAL = 30.0

# Browser-like headers required — bare httpx UA gets HTTP 403.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://truthsocial.com/",
}

TRUTH_SOCIAL_ACCOUNTS: Dict[str, str] = {
    "realDonaldTrump": "107780257626128497",
}


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class TruthSocialMonitor:
    def __init__(
        self,
        on_news: Callable[[dict], Awaitable[None]],
    ) -> None:
        self._on_news = on_news
        self._running = False
        self._seen_ids: Set[str] = set()

    async def start(self) -> None:
        self._running = True
        names = ", ".join(TRUTH_SOCIAL_ACCOUNTS.keys())
        print(f"[News] Truth Social monitor started — accounts: {names}", flush=True)
        while self._running:
            for username, account_id in TRUTH_SOCIAL_ACCOUNTS.items():
                try:
                    await self._poll_account(username, account_id)
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    print(
                        f"[News] Truth Social poll error (@{username}): {exc}",
                        flush=True,
                    )
            await asyncio.sleep(_POLL_INTERVAL)

    async def stop(self) -> None:
        self._running = False

    async def _poll_account(self, username: str, account_id: str) -> None:
        url = f"{_BASE_URL}/accounts/{account_id}/statuses"
        params = {"limit": "5", "exclude_replies": "true", "exclude_reblogs": "true"}

        async with httpx.AsyncClient(timeout=12.0, trust_env=True) as client:
            resp = await client.get(url, params=params, headers=_HEADERS)
            if resp.status_code >= 400:
                body = (resp.text or "").strip()[:200]
                print(
                    f"[News] Truth Social API error {resp.status_code} for @{username}: {body}",
                    flush=True,
                )
                return
            statuses: List[dict] = resp.json()

        for status in reversed(statuses):
            sid = str(status.get("id", ""))
            if not sid or sid in self._seen_ids:
                continue
            self._seen_ids.add(sid)
            item = self._normalize(status, username)
            if item:
                await self._on_news(item)

        if len(self._seen_ids) > 2000:
            self._seen_ids = set(list(self._seen_ids)[-1000:])

    @staticmethod
    def _normalize(status: dict, username: str) -> Optional[dict]:
        raw_content = status.get("content") or ""
        text = _strip_html(raw_content).strip()
        if not text:
            return None
        created = status.get("created_at") or ""
        post_url = status.get("url") or status.get("uri") or ""
        return {
            "id": f"ts_{status.get('id', '')}",
            "headline": text[:280],
            "summary": text,
            "content": text,
            "source": f"truthsocial/@{username}",
            "symbols": [],
            "asset_class": "equity",
            "published_at": created,
            "url": post_url,
        }


def create_truth_social_monitor(
    on_news: Callable[[dict], Awaitable[None]],
) -> TruthSocialMonitor:
    return TruthSocialMonitor(on_news)
