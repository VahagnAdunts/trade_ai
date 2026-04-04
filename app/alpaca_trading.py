from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
from urllib.parse import quote

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from app.config import AppConfig
from app.data_provider import fetch_quote_close_sync


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
    *,
    crypto: bool = False,
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
        notional_usd = config.alpaca_order_dollars
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

    if side == "long":
        notional_usd = config.alpaca_order_dollars
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

    notional = config.alpaca_order_dollars
    price = fetch_quote_close_sync(symbol, config.stock_data_api_key)
    if price <= 0:
        raise ValueError(f"Invalid quote price for {symbol}: {price}")
    shares = int(notional // price)
    if shares < 1:
        return {
            "skipped": True,
            "reason": "order_dollars_below_one_share",
            "message": (
                f"ALPACA_ORDER_DOLLARS (${notional:g}) is below one share at last quote ~${price:.2f}; "
                "short not opened."
            ),
            "notional_usd": notional,
            "last_price_usd": price,
            "symbol": alpaca_sym,
        }

    req = MarketOrderRequest(
        symbol=alpaca_sym,
        qty=float(shares),
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )
    order = client.submit_order(req)
    return {
        "skipped": False,
        "order_id": _order_id(order),
        "symbol": alpaca_sym,
        "qty": float(shares),
        "notional_budget_usd": notional,
        "last_price_usd": price,
        "order_status": _order_status(order),
    }


def _close_position(client: TradingClient, symbol: str) -> Dict[str, Any]:
    """
    Close via DELETE /v2/positions/{symbol}. Crypto pairs like BTC/USD must be
    path-encoded (BTC%2FUSD); a raw slash breaks the URL into wrong segments.
    """
    alpaca_sym = _alpaca_symbol(symbol)
    path_symbol = quote(str(alpaca_sym), safe="")
    order = client.close_position(path_symbol)
    return {"close_order_id": _order_id(order), "symbol": alpaca_sym}


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
    crypto: bool = False,
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
    dollars = config.alpaca_order_dollars
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
            _submit_market_order, open_client, symbol, side, config, crypto=crypto
        )
        out["open"] = open_res
        if open_res.get("skipped"):
            out["skipped"] = True
            out["reason"] = open_res.get("message", open_res.get("reason"))
            print(f"[Alpaca] {symbol} SHORT SKIPPED: {out['reason']}", flush=True)
            return out
        out["ok"] = True
        st = open_res.get("order_status") or "?"
        oid = open_res.get("order_id") or "?"
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
        if pending_id:
            await clear_pending_close(config, pending_id)
    else:
        err = _format_alpaca_error(close_exc)
        out["close_error"] = err
        out["closed"] = False
        if pending_id and should_clear_stale_pending_no_position(close_exc, err):
            await clear_pending_close(config, pending_id)

    return out
