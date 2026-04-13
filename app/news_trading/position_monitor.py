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
    FIXED_STOP_LOSS_PCT = 0.5
    FIXED_TAKE_PROFIT_PCT = 1.0
    FIXED_MAX_HOLD_MINUTES = 20

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
            if minutes_held >= self.FIXED_MAX_HOLD_MINUTES:
                if self.position.asset_class == "equity" and not await self._is_equity_position_open():
                    return await self._notify_already_closed(
                        minutes_held=minutes_held,
                        current_price=self.position.entry_price,
                        pnl_pct=0.0,
                    )
                return await self._close_and_notify(
                    reason="timeout",
                    exit_price=None,
                    minutes_held=minutes_held,
                )

            # 2. Fetch price snapshot (for status only)
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

            # If broker-side bracket already closed the equity position, stop monitoring.
            if self.position.asset_class == "equity" and not await self._is_equity_position_open():
                return await self._notify_already_closed(minutes_held, snapshot.price, pnl_pct)

            print(
                f"[Monitor] {self.position.symbol} {self.position.side.upper()} "
                f"pnl={pnl_pct:+.2f}% held={minutes_held:.0f}min "
                f"rule=timeout_only",
                flush=True,
            )

            await self._notify_monitor_update(
                snapshot=snapshot,
                pnl_pct=pnl_pct,
                minutes_held=minutes_held,
            )
            # Hold — continue to next iteration

    async def _notify_monitor_update(
        self,
        snapshot: PriceSnapshot,
        pnl_pct: float,
        minutes_held: float,
    ) -> None:
        sym = self.position.symbol
        side = self.position.side.upper()

        msg1 = (
            f"📊 MONITOR UPDATE — {sym} {side}\n"
            f"{'─'*30}\n"
            f"Price:  ${snapshot.price:.4f}  (entry ${self.position.entry_price:.4f})\n"
            f"P&L:    {pnl_pct:+.2f}%\n"
            f"Rules:  Broker SL/TP at entry | timeout close only\n"
            f"Held:   {minutes_held:.0f} min / {self.FIXED_MAX_HOLD_MINUTES} min max\n"
            f"RSI:    {snapshot.rsi_current:.1f}\n"
            f"Mom5m:  {snapshot.momentum_5m_pct:+.3f}%\n"
            f"Vol×:   {snapshot.volume_ratio:.1f}x\n"
        )
        await send_telegram_message(self.telegram_cfg, msg1)

    async def _is_equity_position_open(self) -> bool:
        try:
            from app.alpaca_trading import _find_open_position, _make_client
            client = _make_client(self.config)
            pos = await asyncio.to_thread(_find_open_position, client, self.position.symbol)
            return pos is not None
        except Exception as exc:
            print(f"[Monitor] {self.position.symbol} open-position check failed: {exc}", flush=True)
            # Fail-open: keep monitoring rather than accidentally stopping.
            return True

    async def _notify_already_closed(
        self,
        minutes_held: float,
        current_price: float,
        pnl_pct: float,
    ) -> Dict[str, Any]:
        msg = (
            f"📰 NEWS TRADE CLOSED\n"
            f"{self.position.symbol} {self.position.side.upper()} [{self.position.asset_class}]\n\n"
            f"Entry: ${self.position.entry_price:.4f}\n"
            f"Ref px: ${current_price:.4f}\n"
            f"P&L:   {pnl_pct:+.2f}% (reference)\n"
            f"Held:  {minutes_held:.0f} minutes\n\n"
            f"Reason: broker_tp_sl\n"
            f'\nNews: "{self.position.original_headline[:80]}"'
        )
        await send_telegram_message(self.telegram_cfg, msg)
        return {
            "symbol": self.position.symbol,
            "side": self.position.side,
            "reason": "broker_tp_sl",
            "entry_price": self.position.entry_price,
            "exit_price": current_price,
            "pnl_pct": pnl_pct,
            "pnl_usd": self.position.size_usd * pnl_pct / 100.0,
            "minutes_held": minutes_held,
            "votes": None,
            "close_result": {"broker_managed": True},
        }

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
