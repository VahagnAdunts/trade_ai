from __future__ import annotations

import asyncio
from typing import Any, Dict

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from app.config import AppConfig


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
    """Map internal symbols to Alpaca tickers (e.g. BRK.B -> BRK-B)."""
    u = symbol.upper()
    if u == "BRK.B":
        return "BRK-B"
    return symbol


def _make_client(cfg: AppConfig) -> TradingClient:
    return TradingClient(
        api_key=cfg.alpaca_api_key_id,
        secret_key=cfg.alpaca_api_secret_key,
        paper=cfg.alpaca_paper,
    )


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
    client: TradingClient, symbol: str, side: str, notional_usd: float
) -> Dict[str, Any]:
    alpaca_sym = _alpaca_symbol(symbol)
    order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol=alpaca_sym,
        notional=notional_usd,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )
    order = client.submit_order(req)
    return {
        "order_id": _order_id(order),
        "symbol": alpaca_sym,
        "notional_usd": notional_usd,
        "order_status": _order_status(order),
    }


def _close_position(client: TradingClient, symbol: str) -> Dict[str, Any]:
    alpaca_sym = _alpaca_symbol(symbol)
    order = client.close_position(alpaca_sym)
    return {"close_order_id": _order_id(order), "symbol": alpaca_sym}


async def alpaca_consensus_round_trip(
    config: AppConfig,
    symbol: str,
    side: str,
) -> Dict[str, Any]:
    """
    Submit a market order aligned with consensus (long=buy, short=sell),
    wait hold_seconds, then close the position for that symbol.
    """
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
    }
    client = _make_client(config)

    try:
        open_res = await asyncio.to_thread(_submit_market_order, client, symbol, side, dollars)
        out["open"] = open_res
        out["ok"] = True
        st = open_res.get("order_status") or "?"
        oid = open_res.get("order_id") or "?"
        print(
            f"[Alpaca] {symbol} order accepted by API: id={oid} status={st} "
            f"side={side} notional=${dollars} paper={config.alpaca_paper}"
        )
        if side == "short":
            print(
                "[Alpaca] Short: if no position appears, check Orders for rejected/canceled "
                "(shortability/borrow) or wait for market hours for fills."
            )
    except Exception as exc:
        out["error"] = _format_alpaca_error(exc)
        out["phase"] = "open"
        print(f"[Alpaca] {symbol} OPEN FAILED: {out['error']}")
        return out

    await asyncio.sleep(hold)

    try:
        close_res = await asyncio.to_thread(_close_position, client, symbol)
        out["close"] = close_res
        out["closed"] = True
    except Exception as exc:
        out["close_error"] = _format_alpaca_error(exc)
        out["closed"] = False

    return out
