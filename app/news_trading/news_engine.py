"""
news_engine.py — main orchestrator for the news-driven trading module.

Starts all enabled news sources, dispatches events, runs LLM analysis,
opens positions, and launches per-position monitors. Runs forever until cancelled.
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import httpx

_ET = ZoneInfo("America/New_York")
_NYSE_OPEN_HOUR, _NYSE_OPEN_MIN = 9, 30
_NYSE_CLOSE_HOUR, _NYSE_CLOSE_MIN = 16, 0


def _nyse_is_open() -> bool:
    """
    Returns True if the current wall-clock time is within NYSE regular hours
    (Mon–Fri 09:30–16:00 ET). Does NOT account for federal holidays.
    """
    now = datetime.now(_ET)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    open_min = _NYSE_OPEN_HOUR * 60 + _NYSE_OPEN_MIN
    close_min = _NYSE_CLOSE_HOUR * 60 + _NYSE_CLOSE_MIN
    now_min = now.hour * 60 + now.minute
    return open_min <= now_min < close_min

from app.config import AppConfig
from app.data_provider import TwelveDataMultiKeyClient
from app.features import build_feature_context
from app.llm_clients import ClaudeAnalyzer, GeminiAnalyzer, GrokAnalyzer, OpenAIAnalyzer
from app.models import OHLCVPoint
from app.regime import build_market_regime_payload, load_regime_cache
from app.telegram_notifier import TelegramConfig, send_telegram_message
from app.news_trading.event_classifier import EventClassification, classify_event
from app.news_trading.news_prompt import build_fast_classify_prompt, build_news_entry_prompt
from app.news_trading.position_monitor import OpenNewsPosition, PositionMonitor
from app.news_trading.realtime_feed import RealtimePriceFeed

UTC = timezone.utc
_HEADLINE_TTL_MINUTES = 30
_COOLDOWN_MINUTES = 30
_MAX_SYMBOLS_PER_EVENT = 3
_SEMAPHORE_LIMIT = 5
_HISTORICAL_LOOKBACK_DAYS = 30
_MAX_NEWS_AGE_MINUTES = 15


class NewsTradeEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._running = False
        self._open_positions: Dict[str, OpenNewsPosition] = {}
        self._seen_headlines: Dict[str, datetime] = {}
        self._cooldowns: Dict[str, datetime] = {}
        self._semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)

        self._telegram_cfg = TelegramConfig(
            enabled=config.telegram_enabled,
            bot_token=config.telegram_bot_token,
            chat_id=config.telegram_chat_id,
        )
        self._analyzers: List[Tuple[str, Any]] = [
            ("chatgpt", OpenAIAnalyzer(config.openai_api_key, config.openai_model)),
            ("gemini", GeminiAnalyzer(config.google_api_key, config.gemini_model)),
            ("claude", ClaudeAnalyzer(config.anthropic_api_key, config.claude_model)),
            ("grok", GrokAnalyzer(config.xai_api_key, config.grok_model)),
        ]
        self._price_feed = RealtimePriceFeed()

    async def start(self) -> None:
        self._running = True
        mode = self.config.news_source_mode
        print(f"[News] Engine starting... (source_mode={mode})", flush=True)

        sources = []

        use_alpaca = mode in ("alpaca", "both")
        use_x = mode in ("x_stream", "both")

        # ── Traditional news sources (mode=alpaca or both) ──
        if use_alpaca:
            if self.config.news_alpaca_news_enabled and self.config.alpaca_api_key_id:
                from app.news_trading.news_sources.alpaca_news import create_alpaca_news_stream
                stream = create_alpaca_news_stream(
                    self.config.alpaca_api_key_id,
                    self.config.alpaca_api_secret_key,
                    self._on_news_item,
                )
                if stream:
                    sources.append(stream.start())

            if self.config.news_polygon_api_key:
                from app.news_trading.news_sources.polygon_news import create_polygon_feed
                feed = create_polygon_feed(self.config.news_polygon_api_key, self._on_news_item)
                if feed:
                    sources.append(feed.start())

            if self.config.news_crypto_enabled and self.config.news_crypto_panic_api_key:
                from app.news_trading.news_sources.crypto_panic import create_crypto_panic_feed
                cpanic = create_crypto_panic_feed(
                    self.config.news_crypto_panic_api_key, self._on_news_item
                )
                if cpanic:
                    sources.append(cpanic.start())

            from app.news_trading.news_sources.rss_scraper import create_rss_scraper
            rss = create_rss_scraper(self._on_news_item)
            sources.append(rss.start())

        # ── X / Twitter + Truth Social sources (mode=x_stream or both) ──
        if use_x:
            from app.news_trading.news_sources.x_monitor import create_x_monitor
            x_mon = create_x_monitor(self._on_news_item)
            sources.append(x_mon.start())

            if self.config.news_truth_social_enabled:
                from app.news_trading.news_sources.truth_social import create_truth_social_monitor
                ts_mon = create_truth_social_monitor(self._on_news_item)
                sources.append(ts_mon.start())

        if not sources:
            print("[News] WARNING: No news sources configured. Check your .env.", flush=True)

        print("[News] Waiting for events...", flush=True)

        if sources:
            results = await asyncio.gather(*sources, return_exceptions=True)
            for res in results:
                if isinstance(res, BaseException):
                    print(
                        f"[News] FATAL: a news source task exited: {res!r}",
                        flush=True,
                    )
        else:
            while self._running:
                await asyncio.sleep(60)

    async def stop(self) -> None:
        self._running = False

    async def _on_news_item(self, news_item: dict) -> None:
        headline = news_item.get("headline", "")
        symbols = list(news_item.get("symbols", []))
        asset_class = news_item.get("asset_class", "equity")
        raw_published_at = str(news_item.get("published_at") or news_item.get("created_at") or "").strip()

        # Asset class gating
        if asset_class == "equity" and not self.config.news_equity_enabled:
            return
        if asset_class == "crypto" and not self.config.news_crypto_enabled:
            return

        # Deduplicate — MD5 of lowercase headline, TTL 30 minutes
        h = hashlib.md5(headline.lower().encode()).hexdigest()
        now = datetime.now(UTC)
        if h in self._seen_headlines:
            return
        self._seen_headlines[h] = now
        self._cleanup_seen_headlines(now)

        # Staleness protection based on actual publication timestamp.
        is_stale, age_minutes = _is_news_stale(news_item)
        if age_minutes >= 0:
            print(
                f"[News][TimeDebug] published_at_raw='{raw_published_at}' "
                f"age_min={age_minutes:.2f} stale={is_stale}",
                flush=True,
            )
        else:
            print(
                f"[News][TimeDebug] published_at_raw='{raw_published_at}' "
                f"age_min=parse_failed stale=True",
                flush=True,
            )
        if is_stale:
            if age_minutes >= 0:
                print(
                    f"[News] STALE ({age_minutes:.0f}min old) — skipped: {headline[:80]}",
                    flush=True,
                )
            else:
                print(f"[News] NO TIMESTAMP — skipped: {headline[:80]}", flush=True)
            return

        # Keyword classify for direction hint; no longer used as a gate
        classification = classify_event(headline, symbols, asset_class)

        # If no symbols were tagged by the source, use fast LLM to extract one
        source = (news_item.get("source") or "").strip()
        if not symbols:
            fast = await self._fast_llm_classify(headline, news_item.get("summary", ""))
            if not fast.get("is_relevant") or not fast.get("symbol"):
                return  # truly irrelevant
            symbols = [fast["symbol"].upper()]
            classification = classify_event(headline, symbols, asset_class)
            print(
                f"[News] Fast LLM identified symbol {symbols[0]} — {fast.get('reason', '')}",
                flush=True,
            )
            await self._notify_new_post(
                headline=headline,
                source=source,
                symbol=symbols[0],
                reason=fast.get("reason", ""),
                url=news_item.get("url", ""),
            )

        # Full LLM consensus and trades only during NYSE RTH for equities (crypto 24/7)
        if asset_class == "equity" and not _nyse_is_open():
            print(
                f"[News] NYSE closed — skipping analysis/trades (symbols={symbols[:3]})",
                flush=True,
            )
            return

        print(
            f"[News] Event ({classification.impact}): {headline[:100]}",
            flush=True,
        )

        for symbol in symbols[:_MAX_SYMBOLS_PER_EVENT]:
            if symbol in self._open_positions:
                print(f"[News] {symbol} already has open position — skip", flush=True)
                continue
            if self._in_cooldown(symbol):
                print(f"[News] {symbol} in cooldown — skip", flush=True)
                continue
            news_item_copy = {**news_item, "symbol": symbol, "age_minutes": age_minutes}
            asyncio.create_task(
                self._process_event_guarded(classification, news_item_copy)
            )

    def _cleanup_seen_headlines(self, now: datetime) -> None:
        cutoff = now - timedelta(minutes=_HEADLINE_TTL_MINUTES)
        self._seen_headlines = {
            k: v for k, v in self._seen_headlines.items() if v > cutoff
        }

    def _in_cooldown(self, symbol: str) -> bool:
        exp = self._cooldowns.get(symbol)
        if exp is None:
            return False
        return datetime.now(UTC) < exp

    def _set_cooldown(self, symbol: str) -> None:
        self._cooldowns[symbol] = datetime.now(UTC) + timedelta(
            minutes=_COOLDOWN_MINUTES
        )

    async def _fetch_article_content(self, news_id: str, url: str) -> str:
        """
        Fetch full article content via Alpaca REST API (include_content=true).
        Falls back to empty string on any error — never blocks the pipeline.
        """
        if not news_id or not self.config.alpaca_api_key_id:
            return ""
        try:
            import httpx
            from app.news_trading.news_sources.alpaca_news import _strip_html
            headers = {
                "APCA-API-KEY-ID": self.config.alpaca_api_key_id,
                "APCA-API-SECRET-KEY": self.config.alpaca_api_secret_key or "",
            }
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    f"https://data.alpaca.markets/v1beta1/news/{news_id}",
                    headers=headers,
                    params={"include_content": "true"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw = data.get("content") or data.get("news", [{}])[0].get("content", "")
                    return _strip_html(raw)
            return ""
        except Exception as exc:
            print(f"[News] Could not fetch article content for id={news_id}: {exc}", flush=True)
            return ""

    async def _fast_llm_classify(self, headline: str, summary: str) -> dict:
        """
        Lightweight single-model call (gpt-4o-mini) to decide if a headline
        is stock-relevant and extract the ticker. Used when source provides no symbols.
        Returns {"is_relevant": bool, "symbol": str, "reason": str}.
        """
        from app.llm_clients import _parse_json_object
        sys_prompt, user_msg = build_fast_classify_prompt(headline, summary)
        try:
            # Use the OpenAI analyzer's client directly with gpt-4o-mini
            openai_analyzer = next(a for label, a in self._analyzers if label == "chatgpt")
            text = await asyncio.to_thread(
                lambda: openai_analyzer.client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                ).output_text.strip()
            )
            return _parse_json_object(text)
        except Exception as exc:
            print(f"[News] Fast classify error: {exc}", flush=True)
            return {"is_relevant": False, "symbol": "", "reason": str(exc)}

    async def _process_event_guarded(
        self, classification: EventClassification, news_item: dict
    ) -> None:
        """Wrapper that holds the semaphore for the full duration of _process_event."""
        async with self._semaphore:
            await self._process_event(classification, news_item)

    async def _process_event(
        self, classification: EventClassification, news_item: dict
    ) -> None:
        symbol = news_item.get("symbol", "")
        asset_class = news_item.get("asset_class", "equity")
        age_minutes = float(news_item.get("age_minutes", -1.0))
        if not symbol:
            return

        try:
            # Fetch full article body if not already present (WebSocket omits content)
            content = news_item.get("content", "")
            if not content:
                content = await self._fetch_article_content(
                    news_item.get("id", ""),
                    news_item.get("url", ""),
                )

            points = await self._fetch_hourly_30d_with_fallback(symbol)
            features = build_feature_context(symbol, points)
            provider = _NewsHistoricalProvider(self)
            regime_cache = await load_regime_cache(
                provider, crypto=(asset_class == "crypto")
            )
            market_regime = build_market_regime_payload(
                symbol, points, regime_cache, crypto=(asset_class == "crypto")
            )

            # Price reaction guard: skip if likely already priced in.
            if await self._is_already_priced_in(
                symbol=symbol,
                side_hint=classification.direction_hint,
                asset_class=asset_class,
                age_minutes=age_minutes,
            ):
                return

            sys_prompt, user_msg = build_news_entry_prompt(
                symbol=symbol,
                headline=news_item.get("headline", ""),
                summary=news_item.get("summary", ""),
                content=content,
                features=features,
                market_regime=market_regime,
                direction_hint=classification.direction_hint,
                asset_class=asset_class,
            )

            # Notify immediately when analysis starts — includes article link
            await self._notify_analysis_started(symbol, news_item, classification, asset_class)
            # Save full prompt to disk for inspection
            self._save_prompt_to_disk(symbol, sys_prompt, user_msg, news_item)

            print(f"[News] Running LLM analysis for {symbol}...", flush=True)
            results = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        analyzer.analyze_news, sys_prompt, user_msg, symbol
                    )
                    for _, analyzer in self._analyzers
                ],
                return_exceptions=True,
            )

            # Log per-model results
            for (label, _), result in zip(self._analyzers, results):
                if isinstance(result, Exception):
                    print(f"[News] {label} failed for {symbol}: {result}", flush=True)
                else:
                    lc = result.get("long_confidence", "-")
                    sc = result.get("short_confidence", "-")
                    act = result.get("action", "?").upper()
                    conf = result.get("confidence", "-")
                    mh = result.get("max_hold_minutes", "?")
                    print(
                        f"[News] {label}: L={lc}% S={sc}% → {act} {conf}% (hold={mh}min)",
                        flush=True,
                    )

            consensus = self._evaluate_entry_consensus(results, symbol)
            if consensus is None:
                print(f"[News] No consensus for {symbol} — skip", flush=True)
                await self._notify_no_consensus(symbol, news_item, results)
                return

            print(
                f"[News] CONSENSUS: {consensus['side'].upper()} — "
                f"{consensus['supporter_count']}/4 agree, "
                f"min_confidence={consensus['min_confidence']}%, "
                f"max_hold={consensus['max_hold_minutes']}min",
                flush=True,
            )

            await self._notify_entry_analysis(symbol, news_item, consensus, asset_class)

            position = await self._open_position(
                symbol, consensus["side"], consensus, news_item
            )
            if position is None:
                return

            self._open_positions[symbol] = position
            self._set_cooldown(symbol)
            asyncio.create_task(self._run_monitor(position))

        except Exception as exc:
            print(f"[News] Error processing {symbol}: {exc}", flush=True)

    async def _is_already_priced_in(
        self,
        symbol: str,
        side_hint: str,
        asset_class: str,
        age_minutes: float,
    ) -> bool:
        """
        Detect if market likely already reacted to this news.
        """
        try:
            snapshot = await self._price_feed.get_snapshot(symbol, asset_class, self.config)
            move_pct = abs(snapshot.momentum_5m_pct)

            if move_pct > 5.0:
                print(
                    f"[News] {symbol} already moved {move_pct:+.2f}% — "
                    f"news likely priced in (age={age_minutes:.0f}min, hint={side_hint})",
                    flush=True,
                )
                return True

            if age_minutes > 5 and move_pct > 3.0:
                print(
                    f"[News] {symbol} moved {move_pct:+.2f}% on {age_minutes:.0f}min-old news "
                    f"(hint={side_hint}) — skip",
                    flush=True,
                )
                return True
        except Exception as exc:
            print(f"[News] Could not check price reaction for {symbol}: {exc}", flush=True)
        return False

    async def _fetch_hourly_30d_with_fallback(self, symbol: str) -> List[OHLCVPoint]:
        """
        Primary source: Alpaca historical bars.
        Fallback source: TwelveData historical bars.
        """
        try:
            return await self._fetch_hourly_30d_alpaca(symbol)
        except Exception as alpaca_exc:
            print(
                f"[News] Alpaca historical fetch failed for {symbol}: {alpaca_exc} "
                f"— using TwelveData fallback",
                flush=True,
            )
            if not self.config.stock_data_api_key:
                raise ValueError(
                    f"Alpaca fetch failed for {symbol} and TwelveData key is missing"
                ) from alpaca_exc
            provider = TwelveDataMultiKeyClient(
                *self.config.twelve_data_api_keys(), log_label="[News]"
            )
            points = await provider.fetch_hourly_30d(symbol)
            print(
                f"[News] Historical source for {symbol}: TwelveData "
                f"(fallback, bars={len(points)})",
                flush=True,
            )
            return points

    async def _fetch_hourly_30d_alpaca(self, symbol: str) -> List[OHLCVPoint]:
        if not self.config.alpaca_api_key_id or not self.config.alpaca_api_secret_key:
            raise ValueError("Alpaca API credentials are missing")

        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=_HISTORICAL_LOOKBACK_DAYS)
        headers = {
            "APCA-API-KEY-ID": self.config.alpaca_api_key_id,
            "APCA-API-SECRET-KEY": self.config.alpaca_api_secret_key,
        }
        params = {
            "timeframe": "1Hour",
            "start": start_dt.isoformat().replace("+00:00", "Z"),
            "end": end_dt.isoformat().replace("+00:00", "Z"),
            "limit": 1000,
            "adjustment": "raw",
            "sort": "asc",
        }
        alpaca_symbol = _normalize_alpaca_symbol(symbol)

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"https://data.alpaca.markets/v2/stocks/{alpaca_symbol}/bars",
                headers=headers,
                params=params,
            )
            if resp.status_code >= 400:
                body_preview = (resp.text or "").strip().replace("\n", " ")
                if len(body_preview) > 500:
                    body_preview = body_preview[:500] + "...[truncated]"
                raise ValueError(
                    f"Alpaca bars error {resp.status_code} for {alpaca_symbol}: {body_preview}"
                )
            payload = resp.json()

        rows = payload.get("bars") or []
        if not rows:
            raise ValueError(f"No Alpaca hourly bars returned for {symbol}")

        points: List[OHLCVPoint] = []
        for row in rows:
            ts = row.get("t")
            if not ts:
                continue
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            points.append(
                OHLCVPoint(
                    datetime=dt,
                    open=float(row.get("o", 0.0)),
                    high=float(row.get("h", 0.0)),
                    low=float(row.get("l", 0.0)),
                    close=float(row.get("c", 0.0)),
                    volume=float(row.get("v", 0.0)),
                )
            )

        points.sort(key=lambda x: x.datetime)
        if not points:
            raise ValueError(f"No valid Alpaca hourly bars parsed for {symbol}")
        print(
            f"[News] Historical source for {symbol}: Alpaca (bars={len(points)})",
            flush=True,
        )
        return points

    def _evaluate_entry_consensus(
        self, results: list, symbol: str
    ) -> Optional[Dict[str, Any]]:
        decisions = []
        per_model: Dict[str, Any] = {}

        for (label, _), result in zip(self._analyzers, results):
            if isinstance(result, Exception):
                per_model[label] = {"error": str(result)}
                continue
            per_model[label] = result
            decisions.append(result)

        qualified: Dict[str, list] = {"long": [], "short": []}
        for d in decisions:
            if d.get("confidence", 0) >= self.config.news_min_confidence_pct:
                action = d.get("action", "long")
                if action in qualified:
                    qualified[action].append(d)

        chosen_side = max(
            ("long", "short"),
            key=lambda s: (len(qualified[s]), 1 if s == "long" else 0),
        )
        supporters = qualified[chosen_side]

        # Hardcoded: requires exactly 3 of 4
        if len(supporters) < 3:
            return None

        min_conf = min(d["confidence"] for d in supporters)
        max_hold = min(
            (d.get("max_hold_minutes", 60) for d in supporters), default=60
        )
        best = max(supporters, key=lambda d: d["confidence"])

        return {
            "side": chosen_side,
            "min_confidence": min_conf,
            "max_hold_minutes": max_hold,
            "invalidation": best.get("invalidation", ""),
            "thinking": best.get("thinking", ""),
            "per_model": per_model,
            "supporter_count": len(supporters),
        }

    async def _notify_new_post(
        self,
        headline: str,
        source: str,
        symbol: str,
        reason: str,
        url: str,
    ) -> None:
        url_line = f"\n🔗 {url}" if url else ""
        msg = (
            f"🐦 NEW POST DETECTED\n"
            f"Source: {source}\n"
            f"Symbol: {symbol}\n"
            f'"{headline[:200]}"'
            f"\nLLM reason: {reason[:200]}"
            f"{url_line}"
        )
        ok, err = await send_telegram_message(self._telegram_cfg, msg)
        if not ok:
            print(
                f"[News] Telegram new-post notify failed: {err or 'telegram disabled'}",
                flush=True,
            )

    async def _notify_analysis_started(
        self,
        symbol: str,
        news_item: dict,
        classification: EventClassification,
        asset_class: str,
    ) -> None:
        headline = news_item.get("headline", "")
        url = (news_item.get("url") or "").strip()
        url_line = f"\n🔗 {url}" if url else ""
        summary = (news_item.get("summary") or "").strip()
        summary_line = f"\n{summary[:200]}" if summary else ""
        msg = (
            f"📡 NEWS ANALYSING\n"
            f"{symbol} [{asset_class}]\n"
            f"\n"
            f'"{headline[:140]}"'
            f"{summary_line}"
            f"{url_line}\n"
            f"\nAsking all 4 LLMs..."
        )
        await send_telegram_message(self._telegram_cfg, msg)

    def _save_prompt_to_disk(
        self,
        symbol: str,
        sys_prompt: str,
        user_msg: str,
        news_item: dict,
    ) -> None:
        """
        Saves the full prompt (system + user) to outputs/news_prompts/ as a .txt file.
        Filename: YYYYMMDD_HHMMSS_<SYMBOL>.txt
        """
        try:
            out_dir = Path("outputs/news_prompts")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            safe_sym = symbol.replace("/", "_")
            path = out_dir / f"{ts}_{safe_sym}.txt"
            content = (
                f"=== NEWS ITEM ===\n"
                f"Headline : {news_item.get('headline', '')}\n"
                f"Summary  : {news_item.get('summary', '')}\n"
                f"Source   : {news_item.get('source', '')}\n"
                f"URL      : {news_item.get('url', '')}\n"
                f"Symbols  : {news_item.get('symbols', [])}\n"
                f"Published: {news_item.get('published_at', '')}\n"
                f"\n"
                f"{'=' * 60}\n"
                f"=== SYSTEM PROMPT (sent to all 4 LLMs) ===\n"
                f"{'=' * 60}\n"
                f"{sys_prompt}\n"
                f"\n"
                f"{'=' * 60}\n"
                f"=== USER MESSAGE (sent to all 4 LLMs) ===\n"
                f"{'=' * 60}\n"
                f"{user_msg}\n"
            )
            path.write_text(content, encoding="utf-8")
            print(f"[News] Prompt saved → {path}", flush=True)
        except Exception as exc:
            print(f"[News] Could not save prompt to disk: {exc}", flush=True)

    async def _notify_llm_context(
        self,
        symbol: str,
        news_item: dict,
        features: dict,
        market_regime: dict,
        classification: EventClassification,
    ) -> None:
        """Send a readable summary of the prompt context all 4 LLMs will receive."""
        price = features.get("price") or {}
        momentum = features.get("momentum") or {}
        vol = features.get("volatility") or {}
        tf_align = features.get("timeframe_alignment") or {}
        dq = features.get("data_quality") or {}

        # Part 1 — News + classifier
        headline = news_item.get("headline", "")
        summary = (news_item.get("summary") or "").strip()
        url = (news_item.get("url") or "").strip()
        url_line = f"\n🔗 {url}" if url else ""
        part1 = (
            f"📋 LLM CONTEXT — {symbol}\n"
            f"{'─'*30}\n"
            f"📰 News:\n"
            f'"{headline}"\n'
            f"{summary[:300] if summary else '(no summary)'}"
            f"{url_line}\n\n"
            f"Category: {classification.category}"
        )
        await send_telegram_message(self._telegram_cfg, part1)

        # Part 2 — Key technicals
        rsi = momentum.get("rsi_14", "—")
        macd_h = momentum.get("macd_hist", "—")
        ret_1h = momentum.get("ret_1h_pct", "—")
        ret_24h = momentum.get("ret_24h_pct", "—")
        last_close = price.get("last_close", "—")
        ema20 = price.get("ema_20", "—")
        dist_ema = price.get("dist_to_ema20_pct", "—")
        vwap = price.get("vwap", "—")
        vwap_dev = price.get("vwap_deviation_pct", "—")
        bb_z = vol.get("bb_zscore_20", "—")
        atr_pct = vol.get("atr_14_pct_of_price", "—")
        trend_1h = tf_align.get("trend_1h", "—")
        trend_4h = tf_align.get("trend_4h", "—")
        trend_1d = tf_align.get("trend_1d", "—")
        align_score = tf_align.get("alignment_score", "—")
        warnings = dq.get("warnings") or []
        warn_line = f"\n⚠️ {'; '.join(warnings[:2])}" if warnings else ""

        part2 = (
            f"📊 Technicals — {symbol}\n"
            f"{'─'*30}\n"
            f"Price: ${last_close}  EMA20: ${ema20} ({dist_ema:+}%)\n"
            f"VWAP: ${vwap} ({vwap_dev:+}%)\n"
            f"RSI-14: {rsi}  MACD hist: {macd_h}\n"
            f"Ret 1h: {ret_1h}%  24h: {ret_24h}%\n"
            f"BB z-score: {bb_z}  ATR%: {atr_pct}%\n"
            f"Trends → 1h: {trend_1h} | 4h: {trend_4h} | 1d: {trend_1d}\n"
            f"Alignment score: {align_score}"
            f"{warn_line}"
        )
        await send_telegram_message(self._telegram_cfg, part2)

        # Part 3 — Market regime (brief)
        regime_lines = []
        for bench in ("SPY", "QQQ", "VIXY"):
            b = market_regime.get(bench)
            if b and isinstance(b, dict):
                r1h = b.get("ret_1h_pct", "—")
                r24h = b.get("ret_24h_pct", "—")
                regime_lines.append(f"{bench}: 1h={r1h}%  24h={r24h}%")
        if regime_lines:
            part3 = (
                f"🌍 Market Regime — {symbol}\n"
                f"{'─'*30}\n"
                + "\n".join(regime_lines)
            )
            await send_telegram_message(self._telegram_cfg, part3)

    async def _notify_no_consensus(
        self,
        symbol: str,
        news_item: dict,
        results: list,
    ) -> None:
        headline = news_item.get("headline", "")
        url = (news_item.get("url") or "").strip()
        url_line = f"\n🔗 {url}" if url else ""
        model_lines = []
        for (label, _), result in zip(self._analyzers, results):
            if isinstance(result, Exception):
                model_lines.append(f"{label}: ERR")
            else:
                lc = result.get("long_confidence", "-")
                sc = result.get("short_confidence", "-")
                act = str(result.get("action", "?")).upper()
                conf = result.get("confidence", "-")
                model_lines.append(f"{label}: L{lc}% S{sc}% → {act} {conf}%")
        msg = (
            f"❌ NO CONSENSUS — {symbol}\n"
            f'News: "{headline[:100]}"'
            f"{url_line}\n"
            f"\n"
            + "\n".join(model_lines)
        )
        await send_telegram_message(self._telegram_cfg, msg)

    async def _notify_entry_analysis(
        self,
        symbol: str,
        news_item: dict,
        consensus: Dict[str, Any],
        asset_class: str,
    ) -> None:
        headline = news_item.get("headline", "")
        per_model = consensus.get("per_model") or {}
        model_lines = []
        for key in ("chatgpt", "gemini", "claude", "grok"):
            item = per_model.get(key) or {}
            if item.get("error"):
                model_lines.append(f"{key}: ERR")
            else:
                lc = item.get("long_confidence", "-")
                sc = item.get("short_confidence", "-")
                act = str(item.get("action", "?")).upper()
                model_lines.append(f"{key}: L{lc}% S{sc}% → {act}")

        url = (news_item.get("url") or "").strip()
        url_line = f"\n🔗 {url}" if url else ""

        msg = (
            f"📰 NEWS TRADE OPENED\n"
            f"{symbol} {consensus['side'].upper()} [{asset_class}]\n"
            f"Confidence: {consensus['min_confidence']}% "
            f"({consensus['supporter_count']}/4 models agree)\n"
            f"Max hold: {consensus['max_hold_minutes']} min\n"
            f"\n"
            f'News: "{headline[:120]}"'
            f"{url_line}\n"
            f"\n"
            f"Per model:\n"
            + "\n".join(model_lines)
        )
        await send_telegram_message(self._telegram_cfg, msg)

    async def _open_position(
        self,
        symbol: str,
        side: str,
        consensus: Dict[str, Any],
        news_item: dict,
    ) -> Optional[OpenNewsPosition]:
        asset_class = news_item.get("asset_class", "equity")
        order_result: Dict[str, Any] = {}
        order_id: Optional[str] = None

        try:
            if asset_class == "equity":
                if not self.config.alpaca_api_key_id or not self.config.alpaca_api_secret_key:
                    print(f"[News] {symbol} order skipped: Alpaca keys not configured", flush=True)
                    return None
                from app.alpaca_trading import _make_client, _submit_market_order
                client = _make_client(self.config)
                order_result = await asyncio.to_thread(
                    _submit_market_order,
                    client,
                    symbol,
                    side,
                    self.config,
                    self.config.news_trade_dollars,
                    crypto=False,
                    stop_loss_pct=PositionMonitor.FIXED_STOP_LOSS_PCT,
                    take_profit_pct=PositionMonitor.FIXED_TAKE_PROFIT_PCT,
                )
                if order_result.get("skipped"):
                    print(
                        f"[News] {symbol} order skipped: "
                        f"{order_result.get('message', order_result.get('reason'))}",
                        flush=True,
                    )
                    return None
                order_id = order_result.get("order_id")
            else:
                from app.news_trading.binance_broker import BinanceBroker
                broker = BinanceBroker(self.config)
                order_result = await broker.open_position(
                    symbol, side, self.config.news_trade_dollars
                )
                order_id = order_result.get("order_id")

        except Exception as exc:
            if _is_trading_halt_error(exc):
                print(
                    f"[News] {symbol} order skipped: trading halt detected ({exc})",
                    flush=True,
                )
                await self._notify_order_not_opened(
                    symbol=symbol,
                    reason="trading_halt",
                    detail=str(exc),
                    news_item=news_item,
                )
                return None
            print(f"[News] {symbol} order failed: {exc}", flush=True)
            await self._notify_order_not_opened(
                symbol=symbol,
                reason="order_failed",
                detail=str(exc),
                news_item=news_item,
            )
            return None

        # Determine entry price from actual order fill whenever possible.
        entry_price = float(order_result.get("last_price_usd") or 0.0)
        if asset_class == "equity":
            fill_price = await self._get_equity_fill_price(symbol, order_id)
            if fill_price > 0:
                entry_price = fill_price
        if entry_price == 0.0 and asset_class == "equity":
            try:
                import httpx as _httpx
                _headers = {
                    "APCA-API-KEY-ID": self.config.alpaca_api_key_id or "",
                    "APCA-API-SECRET-KEY": self.config.alpaca_api_secret_key or "",
                }
                async with _httpx.AsyncClient(timeout=8.0) as _client:
                    _resp = await _client.get(
                        f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest",
                        headers=_headers,
                    )
                    if _resp.status_code == 200:
                        _q = _resp.json().get("quote") or {}
                        _bid = float(_q.get("bp") or 0)
                        _ask = float(_q.get("ap") or 0)
                        if _bid and _ask:
                            entry_price = (_bid + _ask) / 2.0
                        elif _ask:
                            entry_price = _ask
                        elif _bid:
                            entry_price = _bid
            except Exception as exc:
                print(f"[News] Could not fetch entry price for {symbol}: {exc}", flush=True)

        return OpenNewsPosition(
            id=str(uuid.uuid4()),
            symbol=symbol,
            asset_class=asset_class,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(UTC),
            size_usd=self.config.news_trade_dollars,
            original_headline=news_item.get("headline", ""),
            original_thinking=consensus.get("thinking", ""),
            invalidation_condition=consensus.get("invalidation", ""),
            max_hold_minutes=consensus.get("max_hold_minutes", 60),
            order_id=order_id,
        )

    async def _get_equity_fill_price(self, symbol: str, order_id: Optional[str]) -> float:
        """
        Best-effort fill price extraction for equity entries:
        1) order.filled_avg_price
        2) open position avg_entry_price
        Returns 0.0 if unavailable.
        """
        if not order_id:
            return 0.0
        try:
            from app.alpaca_trading import _find_open_position, _make_client

            client = _make_client(self.config)

            def _poll_fill() -> float:
                import time

                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline:
                    order = client.get_order_by_id(order_id)
                    raw_fill = getattr(order, "filled_avg_price", None)
                    if raw_fill is not None:
                        try:
                            val = float(raw_fill)
                            if val > 0:
                                return val
                        except (TypeError, ValueError):
                            pass
                    pos = _find_open_position(client, symbol)
                    if pos is not None:
                        raw_entry = getattr(pos, "avg_entry_price", None)
                        if raw_entry is not None:
                            try:
                                val = float(raw_entry)
                                if val > 0:
                                    return val
                            except (TypeError, ValueError):
                                pass
                    time.sleep(1.0)
                return 0.0

            return float(await asyncio.to_thread(_poll_fill))
        except Exception as exc:
            print(f"[News] Could not poll fill price for {symbol}: {exc}", flush=True)
            return 0.0

    async def _run_monitor(self, position: OpenNewsPosition) -> None:
        try:
            monitor = PositionMonitor(
                config=self.config,
                position=position,
                llm_clients=self._analyzers,
                telegram_cfg=self._telegram_cfg,
                price_feed=self._price_feed,
            )
            summary = await monitor.run()
            print(
                f"[News] Position closed: {position.symbol} — {summary.get('reason')} "
                f"pnl={summary.get('pnl_pct', 0):+.2f}%",
                flush=True,
            )
        except Exception as exc:
            print(f"[News] Monitor error for {position.symbol}: {exc}", flush=True)
        finally:
            self._open_positions.pop(position.symbol, None)

    async def _notify_order_not_opened(
        self, symbol: str, reason: str, detail: str, news_item: dict
    ) -> None:
        headline = (news_item.get("headline") or "").strip()
        url = (news_item.get("url") or "").strip()
        url_line = f"\n🔗 {url}" if url else ""
        msg = (
            f"⚠️ NEWS TRADE NOT OPENED\n"
            f"{symbol}\n"
            f"Reason: {reason}\n"
            f"Detail: {detail[:250]}"
            f"{url_line}\n"
            f"\nNews: \"{headline[:120]}\""
        )
        await send_telegram_message(self._telegram_cfg, msg)


def _normalize_alpaca_symbol(symbol: str) -> str:
    s = symbol.upper().strip()
    if ":" in s:
        s = s.split(":", 1)[-1]
    return s.replace("/", "-")


class _NewsHistoricalProvider:
    """Adapter for regime loading: Alpaca primary, TwelveData fallback."""

    def __init__(self, engine: NewsTradeEngine) -> None:
        self._engine = engine

    async def fetch_hourly_30d(self, symbol: str) -> List[OHLCVPoint]:
        return await self._engine._fetch_hourly_30d_with_fallback(symbol)


def _is_trading_halt_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "trading halt" in text or "halt on symbol" in text


def _is_news_stale(news_item: dict) -> tuple[bool, float]:
    """
    Returns (is_stale, age_minutes) using published_at from the payload.
    If missing/unparseable, treat as stale.
    """
    raw = news_item.get("published_at") or news_item.get("created_at") or ""
    if not raw:
        return True, -1.0
    text = str(raw).strip()

    # 1) ISO-like timestamps (Alpaca/Polygon/CryptoPanic)
    published: Optional[datetime]
    try:
        published = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        published = None

    # 2) RFC2822/RSS timestamps (e.g., "Fri, 10 Apr 2026 15:20:19 GMT")
    if published is None:
        try:
            published = parsedate_to_datetime(text)
        except Exception:
            published = None

    if published is None:
        return True, -1.0

    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    published = published.astimezone(UTC)
    now = datetime.now(UTC)
    age_minutes = (now - published).total_seconds() / 60.0
    return age_minutes > _MAX_NEWS_AGE_MINUTES, age_minutes
