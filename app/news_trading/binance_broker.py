"""
binance_broker.py — Binance Futures broker for crypto news trades.

Uses raw httpx requests with HMAC-SHA256 signing (no python-binance dependency).
Raises ValueError on init if API keys are not configured.
All async methods raise on API error — callers must wrap in try/except.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict
from urllib.parse import urlencode

import httpx

from app.config import AppConfig


class BinanceBroker:
    def __init__(self, config: AppConfig) -> None:
        if not config.binance_api_key or not config.binance_secret_key:
            raise ValueError(
                "Binance broker requires BINANCE_API_KEY and BINANCE_SECRET_KEY in .env"
            )
        self._api_key = config.binance_api_key
        self._secret = config.binance_secret_key
        if config.binance_testnet:
            self._base = "https://testnet.binancefuture.com"
        else:
            self._base = "https://fapi.binance.com"

    def _sign_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(sorted(params.items()))
        sig = hmac.new(
            self._secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = sig
        return params

    def _headers(self) -> Dict[str, str]:
        return {"X-MBX-APIKEY": self._api_key}

    @staticmethod
    def _to_binance_symbol(symbol: str) -> str:
        """Convert 'BTC/USD' → 'BTCUSDT'."""
        return symbol.replace("/", "").replace("USD", "USDT").upper()

    async def get_current_price(self, symbol: str) -> float:
        bsym = self._to_binance_symbol(symbol)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self._base}/fapi/v1/ticker/price",
                params={"symbol": bsym},
            )
            resp.raise_for_status()
        return float(resp.json().get("price", 0))

    async def open_position(
        self,
        symbol: str,
        side: str,
        usdt_amount: float,
        leverage: int = 1,
    ) -> Dict[str, Any]:
        bsym = self._to_binance_symbol(symbol)
        order_side = "BUY" if side == "long" else "SELL"

        # Set leverage
        async with httpx.AsyncClient(timeout=10.0) as client:
            lev_resp = await client.post(
                f"{self._base}/fapi/v1/leverage",
                headers=self._headers(),
                params=self._sign_request({"symbol": bsym, "leverage": leverage}),
            )
            lev_resp.raise_for_status()

            # Get current price
            price = await self.get_current_price(symbol)
            if price <= 0:
                raise ValueError(f"Binance: invalid price for {bsym}: {price}")

            # Calculate quantity (round to 3 decimal places as a safe default)
            qty = round(usdt_amount / price, 3)

            # Submit market order
            order_params = self._sign_request({
                "symbol": bsym,
                "side": order_side,
                "type": "MARKET",
                "quantity": qty,
            })
            order_resp = await client.post(
                f"{self._base}/fapi/v1/order",
                headers=self._headers(),
                params=order_params,
            )
            order_resp.raise_for_status()
            order = order_resp.json()

        return {
            "order_id": str(order.get("orderId", "")),
            "symbol": bsym,
            "side": order_side,
            "quantity": qty,
            "status": order.get("status", ""),
        }

    async def close_position(self, symbol: str, side: str) -> Dict[str, Any]:
        bsym = self._to_binance_symbol(symbol)
        close_side = "SELL" if side == "long" else "BUY"

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get current position quantity
            pos_resp = await client.get(
                f"{self._base}/fapi/v2/positionRisk",
                headers=self._headers(),
                params=self._sign_request({"symbol": bsym}),
            )
            pos_resp.raise_for_status()
            positions = pos_resp.json()

            qty = 0.0
            realized_pnl = 0.0
            for pos in positions:
                if pos.get("symbol") == bsym:
                    qty = abs(float(pos.get("positionAmt", 0)))
                    realized_pnl = float(pos.get("unRealizedProfit", 0))
                    break

            if qty <= 0:
                return {
                    "close_order_id": "",
                    "pnl_usd": realized_pnl,
                    "message": "no_open_position",
                }

            close_params = self._sign_request({
                "symbol": bsym,
                "side": close_side,
                "type": "MARKET",
                "quantity": qty,
                "reduceOnly": "true",
            })
            close_resp = await client.post(
                f"{self._base}/fapi/v1/order",
                headers=self._headers(),
                params=close_params,
            )
            close_resp.raise_for_status()
            close_order = close_resp.json()

        return {
            "close_order_id": str(close_order.get("orderId", "")),
            "pnl_usd": realized_pnl,
        }
