from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

from app.config import AppConfig
from app.data_provider import fetch_quote_close_sync_try_keys
from app.telegram_notifier import TelegramConfig, send_telegram_message


def _format_alpaca_error(exc: Exception) -> str:
    if isinstance(exc, APIError):
        try:
            parts: list[str] = []
            sc = getattr(exc, "status_code", None)
            if sc is not None:
                parts.append(str(sc))
            parts.append(str(exc.message))
            return " ".join(parts)
        except Exception:
            return str(exc)
    return str(exc)


def _alpaca_symbol(symbol: str) -> str:
    """Map internal symbols to Alpaca tickers (crypto pairs stay BTC/USD; BRK.B -> BRK-B)."""
    u = symbol.upper()
    if "/" in u:
        return u
    if u == "BRK.B":
        return "BRK-B"
    return u


def _normalize_pair_key(symbol: str) -> str:
    """Compare BTC/USD vs BTCUSD vs paper quirks: alphanumeric only, uppercased."""
    return "".join(c for c in symbol.upper() if c.isalnum())


def _find_open_position(client: TradingClient, symbol: str) -> Any:
    """
    Find the open Position for this symbol. Alpaca may use BTC/USD, BTCUSD, etc.;
    path-based close by string often 404s — we match flexibly then close by asset_id.
    """
    want = _normalize_pair_key(_alpaca_symbol(symbol))
    positions: List[Any] = client.get_all_positions()
    for p in positions:
        ps = getattr(p, "symbol", None) or ""
        if _normalize_pair_key(str(ps)) == want:
            return p
    return None


def _make_client(cfg: AppConfig) -> TradingClient:
    return TradingClient(
        api_key=cfg.alpaca_api_key_id,
        secret_key=cfg.alpaca_api_secret_key,
        paper=cfg.alpaca_paper,
    )


def log_alpaca_account_health(config: AppConfig) -> None:
    """Print account equity + shorting flags once before Alpaca round-trips."""
    if not config.alpaca_api_key_id or not config.alpaca_api_secret_key:
        return
    log_alpaca_account_snapshot(_make_client(config))


def log_alpaca_account_snapshot(client: TradingClient) -> None:
    """Print equity and short/margin flags (see Alpaca margin & short requirements)."""
    try:
        acc = client.get_account()
        eq = getattr(acc, "equity", None)
        short_ok = getattr(acc, "shorting_enabled", None)
        tblk = getattr(acc, "trading_blocked", None)
        ablk = getattr(acc, "account_blocked", None)
        st = getattr(acc, "status", None)
        print(
            f"[Alpaca] account status={st} equity={eq} shorting_enabled={short_ok} "
            f"trading_blocked={tblk} account_blocked={ablk}",
            flush=True,
        )
        if short_ok is False:
            print(
                "[Alpaca] shorting_enabled is false — short orders will not work. "
                "Alpaca requires sufficient equity for margin/short (see margin docs).",
                flush=True,
            )
    except Exception as exc:
        print(f"[Alpaca] could not load account snapshot: {exc}", flush=True)


def _order_id(order: Any) -> str | None:
    oid = getattr(order, "id", None)
    if oid is not None:
        return str(oid)
    if isinstance(order, dict):
        v = order.get("id")
        return str(v) if v is not None else None
    return None


def _order_status(order: Any) -> str | None:
    st = getattr(order, "status", None)
    if st is not None:
        return str(st)
    if isinstance(order, dict):
        v = order.get("status")
        return str(v) if v is not None else None
    return None


def _submit_market_order(
    client: TradingClient,
    symbol: str,
    side: str,
    config: AppConfig,
    notional_usd: float,
    *,
    crypto: bool = False,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Equity long: notional (fractional $), TIF day. Equity short: whole-share qty from quote.
    Crypto long: notional, TIF GTC (Alpaca crypto does not use `day`). Crypto short from flat:
    not supported on spot — skipped with reason.
    """
    alpaca_sym = _alpaca_symbol(symbol)
    order_side = OrderSide.BUY if side == "long" else OrderSide.SELL

    if crypto:
        if side == "short":
            return {
                "skipped": True,
                "reason": "crypto_spot_no_short",
                "message": (
                    "Alpaca spot crypto cannot open a short from a flat position (no borrow). "
                    "Consensus short is not executed."
                ),
                "symbol": alpaca_sym,
            }
        req = MarketOrderRequest(
            symbol=alpaca_sym,
            notional=notional_usd,
            side=order_side,
            time_in_force=TimeInForce.GTC,
        )
        order = client.submit_order(req)
        return {
            "skipped": False,
            "order_id": _order_id(order),
            "symbol": alpaca_sym,
            "notional_usd": notional_usd,
            "order_status": _order_status(order),
            "asset_class": "crypto",
        }

    if side == "long" and (stop_loss_pct is None or take_profit_pct is None):
        req = MarketOrderRequest(
            symbol=alpaca_sym,
            notional=notional_usd,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return {
            "skipped": False,
            "order_id": _order_id(order),
            "symbol": alpaca_sym,
            "notional_usd": notional_usd,
            "order_status": _order_status(order),
        }

    notional = notional_usd
    _quote_keys = [config.stock_data_api_key]
    if config.stock_data_api_key_secondary:
        _quote_keys.append(config.stock_data_api_key_secondary)
    price = fetch_quote_close_sync_try_keys(symbol, *_quote_keys)
    if price <= 0:
        raise ValueError(f"Invalid quote price for {symbol}: {price}")
    shares = int(notional // price)
    if shares < 1:
        return {
            "skipped": True,
            "reason": "order_dollars_below_one_share",
            "message": (
                f"Order notional (${notional:g}) is below one share at last quote ~${price:.2f}; "
                "short not opened."
            ),
            "notional_usd": notional,
            "last_price_usd": price,
            "symbol": alpaca_sym,
        }

    req_kwargs: Dict[str, Any] = {
        "symbol": alpaca_sym,
        "qty": float(shares),
        "side": order_side,
        "time_in_force": TimeInForce.DAY,
    }
    if stop_loss_pct is not None and take_profit_pct is not None:
        if side == "long":
            tp_price = round(price * (1.0 + take_profit_pct / 100.0), 4)
            sl_price = round(price * (1.0 - stop_loss_pct / 100.0), 4)
        else:
            tp_price = round(price * (1.0 - take_profit_pct / 100.0), 4)
            sl_price = round(price * (1.0 + stop_loss_pct / 100.0), 4)
        req_kwargs["order_class"] = OrderClass.BRACKET
        req_kwargs["take_profit"] = TakeProfitRequest(limit_price=tp_price)
        req_kwargs["stop_loss"] = StopLossRequest(stop_price=sl_price)

    req = MarketOrderRequest(**req_kwargs)
    order = client.submit_order(req)
    return {
        "skipped": False,
        "order_id": _order_id(order),
        "symbol": alpaca_sym,
        "qty": float(shares),
        "notional_budget_usd": notional,
        "last_price_usd": price,
        "order_status": _order_status(order),
        "bracket_enabled": bool(stop_loss_pct is not None and take_profit_pct is not None),
    }


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _position_side_str(pos: Any) -> str:
    s = getattr(pos, "side", None)
    if s is None:
        return ""
    return str(getattr(s, "value", s) or "").lower()


def _build_close_pnl_summary(pos: Any, close_order: Any) -> Dict[str, Any]:
    """
    Realized round-trip P/L when close fill is known: long → proceeds − cost_basis.
    If filled_avg_price / filled_qty are missing (order not filled yet), do not infer
    P/L as −cost_basis (that was the bogus −$500 case).
    """
    u = _safe_float(getattr(pos, "unrealized_pl", None))
    cost = _safe_float(getattr(pos, "cost_basis", None))
    fq = _safe_float(getattr(close_order, "filled_qty", None)) or 0.0
    exit_px = _safe_float(getattr(close_order, "filled_avg_price", None))
    has_fill = exit_px is not None and fq > 0
    proceeds: Optional[float] = (exit_px * fq) if has_fill else None
    side = _position_side_str(pos)
    pnl: Optional[float] = None
    if "long" in side and has_fill and cost is not None and proceeds is not None:
        pnl = proceeds - cost
    elif "short" in side:
        # Alpaca short cost_basis / fill math is easy to get wrong; use pre-close unrealized_pl.
        pnl = u
    elif u is not None:
        pnl = u
    st = _order_status_lower(close_order)
    return {
        "pnl_usd": pnl,
        "entry_avg": _safe_float(getattr(pos, "avg_entry_price", None)),
        "exit_avg": exit_px,
        "qty": _safe_float(getattr(pos, "qty", None)),
        "cost_basis": cost,
        "proceeds_usd": proceeds,
        "unrealized_pl_at_close": u,
        "side": side,
        "close_order_status": st,
        "close_fill_complete": has_fill and st == "filled",
    }


def _poll_close_order_filled(client: TradingClient, order: Any, timeout_sec: float = 120.0) -> Any:
    """
    close_position() often returns before the market order is fully filled; without
    filled_avg_price, P/L math shows bogus −cost_basis. Poll until filled or terminal.
    """
    oid = _order_id(order)
    if not oid:
        return order
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        o = client.get_order_by_id(oid)
        st = _order_status_lower(o)
        fap = _safe_float(getattr(o, "filled_avg_price", None))
        if st == "filled" and fap is not None:
            return o
        if st in ("canceled", "rejected", "expired"):
            return o
        time.sleep(2.0)
    return client.get_order_by_id(oid)


def _close_position(client: TradingClient, symbol: str) -> Dict[str, Any]:
    """
    Close via DELETE /v2/positions/{symbol_or_asset_id}. Using the position's
    asset_id (UUID) avoids URL/symbol-format mismatches (e.g. BTC/USD vs BTCUSD).
    """
    pos = _find_open_position(client, symbol)
    if pos is None:
        raise Exception("Not Found")
    alpaca_sym = str(getattr(pos, "symbol", "") or _alpaca_symbol(symbol))
    order = client.close_position(getattr(pos, "asset_id"))
    order = _poll_close_order_filled(client, order)
    pnl_summary = _build_close_pnl_summary(pos, order)
    return {
        "close_order_id": _order_id(order),
        "symbol": alpaca_sym,
        "pnl_summary": pnl_summary,
        "position_before_close": {
            "symbol": alpaca_sym,
            "avg_entry_price": getattr(pos, "avg_entry_price", None),
            "qty": getattr(pos, "qty", None),
            "cost_basis": getattr(pos, "cost_basis", None),
            "market_value": getattr(pos, "market_value", None),
            "unrealized_pl": getattr(pos, "unrealized_pl", None),
            "side": _position_side_str(pos),
        },
    }


def _format_alpaca_close_telegram(
    *,
    internal_symbol: str,
    alpaca_sym: str,
    crypto: bool,
    hold_seconds: int,
    paper: bool,
    pnl_summary: Dict[str, Any],
) -> str:
    asset = "Crypto" if crypto else "Equity"
    mode = "paper" if paper else "live"
    pnl = pnl_summary.get("pnl_usd")
    entry = pnl_summary.get("entry_avg")
    exit_ = pnl_summary.get("exit_avg")
    qty = pnl_summary.get("qty")
    cost = pnl_summary.get("cost_basis")
    proceeds = pnl_summary.get("proceeds_usd")
    side = pnl_summary.get("side") or "?"
    ost = pnl_summary.get("close_order_status") or ""
    fill_ok = bool(pnl_summary.get("close_fill_complete"))
    u_pre = pnl_summary.get("unrealized_pl_at_close")

    def _money(v: Any) -> str:
        if v is None:
            return "—"
        try:
            return f"${float(v):,.2f}"
        except (TypeError, ValueError):
            return str(v)

    def _qty_str(v: Any) -> str:
        if v is None:
            return "—"
        try:
            s = f"{float(v):,.6f}".rstrip("0").rstrip(".")
            return s if s else "—"
        except (TypeError, ValueError):
            return str(v)

    if pnl is not None and fill_ok and exit_ is not None:
        pnl_line = f"Realized P/L (round-trip): ${float(pnl):+,.2f}"
    elif pnl is not None:
        pnl_line = f"Realized P/L (best estimate): ${float(pnl):+,.2f}"
    elif u_pre is not None:
        pnl_line = (
            f"P/L: not computed (close fill incomplete). "
            f"Unrealized before close: {_money(u_pre)} · order status: {ost or '?'}"
        )
    else:
        pnl_line = f"P/L: unknown (close fill incomplete; order status: {ost or '?'})"

    lines = [
        f"Alpaca close ({mode})",
        f"{asset} · {internal_symbol} (broker: {alpaca_sym})",
        f"Side: {side} · Hold: {hold_seconds}s",
        f"Avg entry: {_money(entry)} → Avg exit (close fill): {_money(exit_)}",
        f"Qty (position): {_qty_str(qty)}",
        f"Cost basis (open): {_money(cost)} · Proceeds (close sell/buy fill): {_money(proceeds)}",
        pnl_line,
    ]
    return "\n".join(lines)


async def _send_alpaca_close_telegram(
    telegram_cfg: Optional[TelegramConfig],
    *,
    internal_symbol: str,
    alpaca_sym: str,
    crypto: bool,
    hold_seconds: int,
    paper: bool,
    pnl_summary: Dict[str, Any],
) -> None:
    if telegram_cfg is None:
        return
    text = _format_alpaca_close_telegram(
        internal_symbol=internal_symbol,
        alpaca_sym=alpaca_sym,
        crypto=crypto,
        hold_seconds=hold_seconds,
        paper=paper,
        pnl_summary=pnl_summary,
    )
    ok, err = await send_telegram_message(telegram_cfg, text)
    if ok:
        print(f"[Telegram] Alpaca close summary sent for {internal_symbol}", flush=True)
    elif err:
        print(f"[Telegram] Alpaca close notify failed: {err}", flush=True)


def _order_status_lower(order: Any) -> str:
    st = getattr(order, "status", None)
    if st is None:
        return ""
    if hasattr(st, "value"):
        return str(st.value).lower()
    return str(st).lower()


async def _wait_for_fill_and_visible_position(
    config: AppConfig,
    order_id: str,
    symbol: str,
    *,
    timeout_sec: float = 120.0,
    poll_sec: float = 2.0,
) -> tuple[bool, str]:
    """
    After submit_order, wait until the order is filled (or terminal failure) and
    a matching open position exists. Otherwise we may sleep the hold with no position.
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        client = _make_client(config)

        def _poll() -> tuple[Any, Any]:
            o = client.get_order_by_id(order_id)
            p = _find_open_position(client, symbol)
            return o, p

        try:
            order, pos = await asyncio.to_thread(_poll)
        except Exception as exc:
            return False, f"poll_failed: {_format_alpaca_error(exc)}"

        sv = _order_status_lower(order)
        if sv in ("filled", "partially_filled") and pos is not None:
            return True, sv
        if sv in ("canceled", "rejected", "expired"):
            return False, f"order_{sv}"

        await asyncio.sleep(poll_sec)

    return False, "timeout_waiting_fill_or_position"


async def _close_position_with_retries(
    config: AppConfig,
    symbol: str,
    *,
    hold_seconds: Optional[int] = None,
    max_attempts: int = 4,
) -> tuple[Dict[str, Any], Optional[Exception]]:
    """
    Use a fresh TradingClient per attempt. The open-time client must not be reused
    after a long asyncio.sleep — idle HTTP connections are often closed by the server
    or time out, so close_position would fail or hang.
    """
    if hold_seconds is not None:
        detail = f"after {hold_seconds}s hold"
    else:
        detail = "recovery"
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        client = _make_client(config)
        try:
            print(
                f"[Alpaca] {symbol} closing position ({detail}, attempt {attempt}/{max_attempts})...",
                flush=True,
            )
            close_res = await asyncio.to_thread(_close_position, client, symbol)
            return close_res, None
        except Exception as exc:
            last_err = exc
            err_s = _format_alpaca_error(exc)
            print(
                f"[Alpaca] {symbol} close attempt {attempt} failed: {err_s}",
                flush=True,
            )
            if attempt < max_attempts:
                await asyncio.sleep(min(2.0 * attempt, 10.0))
    assert last_err is not None
    return {}, last_err


async def alpaca_consensus_round_trip(
    config: AppConfig,
    symbol: str,
    side: str,
    *,
    order_usd: float,
    crypto: bool = False,
    telegram_cfg: Optional[TelegramConfig] = None,
) -> Dict[str, Any]:
    """
    Submit a market order aligned with consensus (long=buy notional, short=sell whole shares),
    wait hold_seconds, then close the position for that symbol.

    After a successful open, the scheduled close is recorded in ALPACA_PENDING_CLOSES_FILE
    so a process restart can still close at the right time (see reconcile_pending_closes_on_startup).
    """
    from app.alpaca_pending import (
        clear_pending_close,
        new_pending_record,
        register_pending_close,
        should_clear_stale_pending_no_position,
    )

    hold = config.alpaca_hold_seconds
    dollars = order_usd
    alpaca_sym = _alpaca_symbol(symbol)
    out: Dict[str, Any] = {
        "ok": False,
        "paper": config.alpaca_paper,
        "alpaca_symbol": alpaca_sym,
        "side": side,
        "order_dollars": dollars,
        "hold_seconds": hold,
        "crypto": crypto,
    }
    open_client = _make_client(config)
    pending_id: Optional[str] = None

    try:
        open_res = await asyncio.to_thread(
            _submit_market_order,
            open_client,
            symbol,
            side,
            config,
            dollars,
            crypto=crypto,
        )
        out["open"] = open_res
        if open_res.get("skipped"):
            out["skipped"] = True
            out["reason"] = open_res.get("message", open_res.get("reason"))
            print(f"[Alpaca] {symbol} SHORT SKIPPED: {out['reason']}", flush=True)
            return out
        st = open_res.get("order_status") or "?"
        oid = open_res.get("order_id")
        if side == "long":
            print(
                f"[Alpaca] {symbol} order accepted by API: id={oid} status={st} "
                f"side=long notional=${dollars} paper={config.alpaca_paper}"
            )
        else:
            lp = open_res.get("last_price_usd")
            lp_s = f"{float(lp):.2f}" if lp is not None else "?"
            print(
                f"[Alpaca] {symbol} order accepted by API: id={oid} status={st} "
                f"side=short qty={open_res.get('qty')} @ ~${lp_s} (budget ${dollars:g}) "
                f"paper={config.alpaca_paper}"
            )
        if not oid:
            out["error"] = "missing_order_id_after_submit"
            out["phase"] = "open"
            print(f"[Alpaca] {symbol} OPEN FAILED: no order id in response", flush=True)
            return out
        fill_ok, fill_reason = await _wait_for_fill_and_visible_position(
            config, str(oid), symbol
        )
        out["fill_wait"] = {"ok": fill_ok, "reason": fill_reason}
        if not fill_ok:
            out["error"] = fill_reason
            out["phase"] = "fill_wait"
            print(
                f"[Alpaca] {symbol} no tradable position after order (paper may be slow): {fill_reason}",
                flush=True,
            )
            return out
        print(f"[Alpaca] {symbol} order filled & position visible ({fill_reason})", flush=True)
        out["ok"] = True
        pending_id, pending_record = new_pending_record(
            symbol=symbol,
            hold_seconds=hold,
            crypto=crypto,
            side=side,
            paper=config.alpaca_paper,
        )
        try:
            await register_pending_close(config, pending_record)
            out["pending_close_id"] = pending_id
        except Exception as reg_exc:
            print(
                f"[Alpaca pending] WARN: could not persist pending close (position still open): "
                f"{reg_exc}",
                flush=True,
            )
    except Exception as exc:
        out["error"] = _format_alpaca_error(exc)
        out["phase"] = "open"
        print(f"[Alpaca] {symbol} OPEN FAILED: {out['error']}")
        return out

    await asyncio.sleep(hold)

    close_res, close_exc = await _close_position_with_retries(
        config, symbol, hold_seconds=hold
    )
    if close_exc is None:
        out["close"] = close_res
        out["closed"] = True
        pnl_summary = (close_res or {}).get("pnl_summary") or {}
        out["close_pnl"] = pnl_summary
        if pending_id:
            await clear_pending_close(config, pending_id)
        await _send_alpaca_close_telegram(
            telegram_cfg,
            internal_symbol=symbol,
            alpaca_sym=alpaca_sym,
            crypto=crypto,
            hold_seconds=hold,
            paper=config.alpaca_paper,
            pnl_summary=pnl_summary,
        )
    else:
        err = _format_alpaca_error(close_exc)
        out["close_error"] = err
        out["closed"] = False
        if pending_id and should_clear_stale_pending_no_position(close_exc, err):
            await clear_pending_close(config, pending_id)

    return out
