"""
x_api_stream.py — Real-time X (Twitter) posts via official API v2 filtered stream.

Uses app-only Bearer token (X_BEARER_TOKEN).  This is the **reliable, low-latency**
path compared to syndication scraping (HTTP 429 from X on datacenter IPs).

Flow:
  1. DELETE existing filtered-stream rules tagged ``tredai_*`` (owned by this app).
  2. POST new ``from:user`` rules (packed up to 512 chars per rule; max 25 rules).
  3. Open long-lived GET ``/2/tweets/search/stream`` and emit ``news_item`` dicts.

Requires developer project access to **filtered stream** (not available on all tiers).
If the stream returns 401/403, disable NEWS_X_API_STREAM_ENABLED or upgrade X API access.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

import httpx

from app.news_trading.news_sources.x_syndication import MONITORED_X_ACCOUNTS

_RULES_URL = "https://api.twitter.com/2/tweets/search/stream/rules"
_STREAM_URL = "https://api.twitter.com/2/tweets/search/stream"
_RULE_TAG_PREFIX = "tredai_"
_MAX_RULE_VALUE_LEN = 500  # stay under 512 API limit
_MAX_RULES = 25
_RECONNECT_DELAY = 5.0


def _pack_from_rules(handles: List[str]) -> List[Tuple[str, str]]:
    """Return list of (rule_value, tag) with OR-chained ``from:user`` clauses (≤512 chars each)."""
    rules: List[Tuple[str, str]] = []
    parts: List[str] = []
    rule_idx = 0
    for h in handles:
        h = (h or "").strip()
        if not h:
            continue
        part = f"from:{h}"
        candidate = " OR ".join(parts + [part]) if parts else part
        if len(candidate) > _MAX_RULE_VALUE_LEN and parts:
            rules.append((" OR ".join(parts), f"{_RULE_TAG_PREFIX}{rule_idx}"))
            rule_idx += 1
            parts = [part]
        elif len(part) > _MAX_RULE_VALUE_LEN:
            continue
        else:
            parts.append(part)
    if parts:
        rules.append((" OR ".join(parts), f"{_RULE_TAG_PREFIX}{rule_idx}"))
    return rules[:_MAX_RULES]


class XFilteredStreamMonitor:
    def __init__(
        self,
        bearer_token: str,
        on_news: Callable[[dict], Awaitable[None]],
    ) -> None:
        self._bearer = bearer_token.strip()
        self._on_news = on_news
        self._running = False
        self._seen_ids: Set[str] = set()

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._bearer}"}

    async def start(self) -> None:
        self._running = True
        handles = [h for _, h in MONITORED_X_ACCOUNTS]
        print(
            f"[News] X API filtered stream — configuring up to {len(handles)} handles "
            f"({len(_pack_from_rules(handles))} rule(s))",
            flush=True,
        )
        while self._running:
            try:
                await self._sync_rules(handles)
                await self._consume_stream()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                print(
                    f"[News] X API stream error ({type(exc).__name__}: {exc}) "
                    f"— reconnecting in {_RECONNECT_DELAY}s",
                    flush=True,
                )
                await asyncio.sleep(_RECONNECT_DELAY)

    async def stop(self) -> None:
        self._running = False

    async def _sync_rules(self, handles: List[str]) -> None:
        packed = _pack_from_rules(handles)
        if not packed:
            raise RuntimeError("no X stream rules could be built (empty handle list?)")

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Delete prior tredai_* rules
            gr = await client.get(_RULES_URL, headers=self._headers())
            gr.raise_for_status()
            payload = gr.json()
            ids_to_delete: List[str] = []
            for row in payload.get("data") or []:
                tag = str(row.get("tag") or "")
                rid = row.get("id")
                if tag.startswith(_RULE_TAG_PREFIX) and rid:
                    ids_to_delete.append(str(rid))
            if ids_to_delete:
                dr = await client.post(
                    _RULES_URL,
                    headers={**self._headers(), "Content-Type": "application/json"},
                    json={"delete": {"ids": ids_to_delete}},
                )
                if dr.status_code not in (200, 201):
                    print(
                        f"[News] X API stream: rule delete returned {dr.status_code}: "
                        f"{(dr.text or '')[:200]}",
                        flush=True,
                    )

            add_body = {"add": [{"value": v, "tag": t} for v, t in packed]}
            ar = await client.post(
                _RULES_URL,
                headers={**self._headers(), "Content-Type": "application/json"},
                json=add_body,
            )
            if ar.status_code not in (200, 201):
                raise RuntimeError(
                    f"X stream rule add failed HTTP {ar.status_code}: {(ar.text or '')[:400]}"
                )
            meta = ar.json().get("meta", {})
            summary = ar.json().get("summary", {})
            print(
                f"[News] X API stream rules synced — created={summary.get('created', 0)} "
                f"not_created={summary.get('not_created', 0)} "
                f"invalid={len(meta.get('invalid', []) or [])}",
                flush=True,
            )

    async def _consume_stream(self) -> None:
        params = {
            "tweet.fields": "created_at,author_id,text",
            "expansions": "author_id",
            "user.fields": "username,name",
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "GET",
                _STREAM_URL,
                headers=self._headers(),
                params=params,
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")[:600]
                    raise RuntimeError(
                        f"X filtered stream HTTP {resp.status_code}: {body}"
                    )
                print("[News] X API filtered stream connected (real-time)", flush=True)
                async for line in resp.aiter_lines():
                    if not self._running:
                        break
                    line = (line or "").strip()
                    if not line or line == " ":  # heartbeats
                        continue
                    if not line.startswith("{"):
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "errors" in obj:
                        print(f"[News] X API stream payload error: {obj['errors']}", flush=True)
                        continue
                    tw = obj.get("data")
                    if not isinstance(tw, dict):
                        continue
                    tid = str(tw.get("id") or "")
                    if not tid or tid in self._seen_ids:
                        continue
                    self._seen_ids.add(tid)
                    if len(self._seen_ids) > 50_000:
                        self._seen_ids = set(list(self._seen_ids)[-25_000:])

                    users = {
                        u["id"]: u
                        for u in (obj.get("includes") or {}).get("users") or []
                        if isinstance(u, dict) and u.get("id")
                    }
                    aid = str(tw.get("author_id") or "")
                    umeta = users.get(aid, {})
                    uname = str(umeta.get("username") or aid)
                    dname = str(umeta.get("name") or uname)
                    item = _tweet_v2_to_news_item(tw, uname, dname)
                    if item:
                        await self._on_news(item)


def _tweet_v2_to_news_item(tweet: dict, username: str, display_name: str) -> Optional[dict]:
    text = (tweet.get("text") or "").strip()
    if not text:
        return None
    tid = str(tweet.get("id") or "")
    if not tid:
        return None
    raw = str(tweet.get("created_at") or "").strip()
    published_at = ""
    if raw:
        try:
            published_at = (
                datetime.fromisoformat(raw.replace("Z", "+00:00"))
                .astimezone(timezone.utc)
                .isoformat()
            )
        except ValueError:
            published_at = raw

    return {
        "id": f"xapi_{tid}",
        "headline": f"{display_name}: {text[:250]}",
        "summary": text,
        "content": text,
        "source": f"x/@{username}",
        "symbols": [],
        "asset_class": "equity",
        "published_at": published_at,
        "url": f"https://x.com/{username}/status/{tid}",
    }


def create_x_filtered_stream(
    bearer_token: str,
    on_news: Callable[[dict], Awaitable[None]],
) -> XFilteredStreamMonitor:
    return XFilteredStreamMonitor(bearer_token, on_news)
