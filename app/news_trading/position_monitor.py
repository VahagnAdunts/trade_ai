"""
position_monitor.py — 5-minute LLM exit-decision loop for open news positions.

MONITOR_INTERVAL_SECONDS = 300 (hardcoded, never configurable).
Consensus to close requires 3 of 4 models (hardcoded, never configurable).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.config import AppConfig
from app.news_trading.news_prompt import build_news_exit_prompt
from app.news_trading.realtime_feed import PriceSnapshot, RealtimePriceFeed
from app.telegram_notifier import TelegramConfig, send_telegram_message

UTC = timezone.utc


@dataclass
class OpenNewsPosition:
    id: str                         # uuid4
    symbol: str
    asset_class: str                # "equity" or "crypto"
    side: str                       # "long" or "short"
    entry_price: float
    entry_time: datetime
    size_usd: float
    original_headline: str
    original_thinking: str          # "thinking" field from best entry LLM
    invalidation_condition: str     # from entry LLM with highest confidence
    max_hold_minutes: int           # from entry consensus (min across models)
    order_id: Optional[str]


class PositionMonitor:
    # Hardcoded — do not make configurable
    MONITOR_INTERVAL_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        config: AppConfig,
        position: OpenNewsPosition,
        llm_clients: List[Tuple[str, Any]],
        telegram_cfg: TelegramConfig,
        price_feed: RealtimePriceFeed,
    ) -> None:
        self.config = config
        self.position = position
        self.llm_clients = llm_clients
        self.telegram_cfg = telegram_cfg
        self.price_feed = price_feed

    async def run(self) -> Dict[str, Any]:
        """Main monitor loop. Runs until position is closed. Returns close summary dict."""
        while True:
            await asyncio.sleep(self.MONITOR_INTERVAL_SECONDS)

            minutes_held = (
                datetime.now(UTC) - self.position.entry_time
            ).total_seconds() / 60.0

            # 1. Hard time limit
            if minutes_held >= self.position.max_hold_minutes:
                return await self._close_and_notify(
                    reason="timeout",
                    exit_price=None,
                    minutes_held=minutes_held,
                )

            # 2. Fetch price snapshot
            try:
                snapshot = await self.price_feed.get_snapshot(
                    self.position.symbol,
                    self.position.asset_class,
                    self.config,
                )
            except Exception as exc:
                print(
                    f"[Monitor] {self.position.symbol} price fetch failed: {exc}",
                    flush=True,
                )
                continue

            pnl_pct = self._calculate_pnl_pct(snapshot.price)

            # 3. Stop loss
            if pnl_pct <= -self.config.news_stop_loss_pct:
                # Confirm with a fresh (non-cached) snapshot to avoid false triggers.
                try:
                    confirm_snapshot = await self.price_feed.get_snapshot(
                        self.position.symbol,
                        self.position.asset_class,
                        self.config,
                        force_refresh=True,
                    )
                    confirm_pnl = self._calculate_pnl_pct(confirm_snapshot.price)
                    if confirm_pnl > -self.config.news_stop_loss_pct:
                        continue
                    snapshot = confirm_snapshot
                    pnl_pct = confirm_pnl
                except Exception as exc:
                    print(
                        f"[Monitor] {self.position.symbol} stop-loss confirm failed: {exc}",
                        flush=True,
                    )
                    # Do not close on failed confirm — first snapshot may be a bad tick.
                    continue
                return await self._close_and_notify(
                    reason="stop_loss",
                    exit_price=snapshot.price,
                    minutes_held=minutes_held,
                    pnl_pct=pnl_pct,
                )

            # 4. Take profit
            if pnl_pct >= self.config.news_take_profit_pct:
                # Confirm with a fresh (non-cached) snapshot to avoid false triggers.
                try:
                    confirm_snapshot = await self.price_feed.get_snapshot(
                        self.position.symbol,
                        self.position.asset_class,
                        self.config,
                        force_refresh=True,
                    )
                    confirm_pnl = self._calculate_pnl_pct(confirm_snapshot.price)
                    if confirm_pnl < self.config.news_take_profit_pct:
                        continue
                    snapshot = confirm_snapshot
                    pnl_pct = confirm_pnl
                except Exception as exc:
                    print(
                        f"[Monitor] {self.position.symbol} take-profit confirm failed: {exc}",
                        flush=True,
                    )
                    continue
                return await self._close_and_notify(
                    reason="take_profit",
                    exit_price=snapshot.price,
                    minutes_held=minutes_held,
                    pnl_pct=pnl_pct,
                )

            # 5. Ask LLMs whether to hold or close
            decision, vote_breakdown, sys_prompt, user_msg, raw_results = (
                await self._ask_llms_exit_decision(snapshot, pnl_pct, minutes_held)
            )

            print(
                f"[Monitor] {self.position.symbol} {self.position.side.upper()} "
                f"pnl={pnl_pct:+.2f}% held={minutes_held:.0f}min "
                f"decision={decision} votes={vote_breakdown}",
                flush=True,
            )

            # Send monitoring update to Telegram with prompt + per-model results
            await self._notify_monitor_update(
                snapshot, pnl_pct, minutes_held, decision,
                vote_breakdown, raw_results, sys_prompt, user_msg,
            )

            if decision == "close":
                return await self._close_and_notify(
                    reason="llm_exit",
                    exit_price=snapshot.price,
                    minutes_held=minutes_held,
                    pnl_pct=pnl_pct,
                    votes=vote_breakdown,
                )
            # Hold — continue to next iteration

    async def _ask_llms_exit_decision(
        self,
        snapshot: PriceSnapshot,
        pnl_pct: float,
        minutes_held: float,
    ) -> Tuple[str, Dict[str, str], str, str, list]:
        sys_prompt, user_msg = build_news_exit_prompt(
            symbol=self.position.symbol,
            side=self.position.side,
            entry_price=self.position.entry_price,
            current_price=snapshot.price,
            pnl_pct=pnl_pct,
            minutes_held=minutes_held,
            original_headline=self.position.original_headline,
            original_thinking=self.position.original_thinking,
            invalidation_condition=self.position.invalidation_condition,
            current_rsi=snapshot.rsi_current,
            price_momentum_5m=snapshot.momentum_5m_pct,
            volume_ratio=snapshot.volume_ratio,
        )

        results = await asyncio.gather(
            *[
                asyncio.to_thread(analyzer.quick_exit_decision, sys_prompt, user_msg)
                for _, analyzer in self.llm_clients
            ],
            return_exceptions=True,
        )

        close_votes = 0
        urgent_votes = 0
        hold_votes = 0
        vote_breakdown: Dict[str, str] = {}

        for (label, _), result in zip(self.llm_clients, results):
            if isinstance(result, Exception):
                vote_breakdown[label] = "error"
                continue
            decision_val = result.get("decision", "hold")
            urgency_val = result.get("urgency", "normal")
            vote_breakdown[label] = f"{decision_val}/{urgency_val}"
            if decision_val == "close":
                close_votes += 1
                if urgency_val == "urgent":
                    urgent_votes += 1
            else:
                hold_votes += 1

        # Urgent override: 2+ urgent AND 2+ close
        if urgent_votes >= 2 and close_votes >= 2:
            return "close", vote_breakdown, sys_prompt, user_msg, list(results)
        # Normal 3-of-4 consensus
        if close_votes >= 3:
            return "close", vote_breakdown, sys_prompt, user_msg, list(results)
        return "hold", vote_breakdown, sys_prompt, user_msg, list(results)

    async def _notify_monitor_update(
        self,
        snapshot: PriceSnapshot,
        pnl_pct: float,
        minutes_held: float,
        decision: str,
        vote_breakdown: Dict[str, str],
        raw_results: list,
        sys_prompt: str,
        user_msg: str,
    ) -> None:
        sym = self.position.symbol
        side = self.position.side.upper()
        decision_emoji = "✅ HOLD" if decision == "hold" else "🔴 CLOSE"

        # Per-model vote + thinking
        model_lines = []
        for (label, _), result in zip(self.llm_clients, raw_results):
            if isinstance(result, Exception):
                model_lines.append(f"{label}: ERR — {result}")
                continue
            d = result.get("decision", "hold")
            u = result.get("urgency", "normal")
            thinking = (result.get("thinking") or "").strip()[:200]
            thinking_line = f"\n  💭 {thinking}" if thinking else ""
            model_lines.append(f"{label}: {d}/{u}{thinking_line}")

        msg1 = (
            f"📊 MONITOR UPDATE — {sym} {side}\n"
            f"{'─'*30}\n"
            f"Price:  ${snapshot.price:.4f}  (entry ${self.position.entry_price:.4f})\n"
            f"P&L:    {pnl_pct:+.2f}%\n"
            f"Held:   {minutes_held:.0f} min / {self.position.max_hold_minutes} min max\n"
            f"RSI:    {snapshot.rsi_current:.1f}\n"
            f"Mom5m:  {snapshot.momentum_5m_pct:+.3f}%\n"
            f"Vol×:   {snapshot.volume_ratio:.1f}x\n"
            f"\n"
            f"Decision: {decision_emoji}\n"
            f"\nPer model:\n"
            + "\n".join(model_lines)
        )
        await send_telegram_message(self.telegram_cfg, msg1)

    def _calculate_pnl_pct(self, current_price: float) -> float:
        if self.position.entry_price == 0:
            return 0.0
        if self.position.side == "long":
            return (current_price - self.position.entry_price) / self.position.entry_price * 100.0
        # short
        return (self.position.entry_price - current_price) / self.position.entry_price * 100.0

    async def _close_and_notify(
        self,
        reason: str,
        exit_price: Optional[float],
        minutes_held: float,
        pnl_pct: Optional[float] = None,
        votes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        # Execute close order
        actual_exit_price = exit_price
        close_result: Dict[str, Any] = {}

        try:
            if self.position.asset_class == "equity":
                from app.alpaca_trading import _close_position, _make_client
                client = _make_client(self.config)
                close_result = await asyncio.to_thread(
                    _close_position, client, self.position.symbol
                )
                # Try to get filled exit price from close result
                pnl_summary = close_result.get("pnl_summary") or {}
                if pnl_summary.get("exit_avg"):
                    actual_exit_price = pnl_summary["exit_avg"]
            else:
                from app.news_trading.binance_broker import BinanceBroker
                broker = BinanceBroker(self.config)
                close_result = await broker.close_position(
                    self.position.symbol, self.position.side
                )
        except Exception as exc:
            print(
                f"[Monitor] {self.position.symbol} close order failed: {exc}",
                flush=True,
            )
            close_result = {"error": str(exc)}

        if actual_exit_price is None:
            actual_exit_price = self.position.entry_price  # fallback

        # Always recalculate P&L from the actual fill price, not the snapshot
        # that triggered the close (they differ — snapshot is a quote estimate).
        trigger_snapshot_pnl = pnl_pct
        pnl_pct = self._calculate_pnl_pct(actual_exit_price)

        pnl_usd = self.position.size_usd * pnl_pct / 100.0

        # Telegram close notification
        vote_line = ""
        if reason == "llm_exit" and votes:
            close_c = sum(1 for v in votes.values() if "close" in v)
            hold_c = sum(1 for v in votes.values() if "hold" in v)
            vote_line = f"\nLLM votes: {close_c} close, {hold_c} hold"

        diverge_line = ""
        if (
            reason in ("stop_loss", "take_profit")
            and trigger_snapshot_pnl is not None
            and abs(pnl_pct - trigger_snapshot_pnl) > 0.15
        ):
            diverge_line = (
                f"\n(Trigger snapshot P&L was {trigger_snapshot_pnl:+.2f}% "
                f"vs fill {pnl_pct:+.2f}%)"
            )

        msg = (
            f"📰 NEWS TRADE CLOSED\n"
            f"{self.position.symbol} {self.position.side.upper()} "
            f"[{self.position.asset_class}]\n"
            f"\n"
            f"Entry: ${self.position.entry_price:.4f}\n"
            f"Exit:  ${actual_exit_price:.4f}\n"
            f"P&L:   {pnl_pct:+.2f}% (${pnl_usd:+.2f})\n"
            f"Held:  {minutes_held:.0f} minutes\n"
            f"\n"
            f"Reason: {reason}"
            f"{diverge_line}"
            f"{vote_line}\n"
            f'\nNews: "{self.position.original_headline[:80]}"'
        )
        await send_telegram_message(self.telegram_cfg, msg)
        print(
            f"[Monitor] {self.position.symbol} closed — reason={reason} "
            f"pnl={pnl_pct:+.2f}% held={minutes_held:.0f}min",
            flush=True,
        )

        return {
            "symbol": self.position.symbol,
            "side": self.position.side,
            "reason": reason,
            "entry_price": self.position.entry_price,
            "exit_price": actual_exit_price,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "minutes_held": minutes_held,
            "votes": votes,
            "close_result": close_result,
        }
