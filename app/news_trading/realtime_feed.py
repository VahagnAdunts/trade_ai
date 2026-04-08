"""
realtime_feed.py — near-real-time price snapshots for open positions.

Fetches latest quote + 1-minute bars to compute momentum, RSI, and volume
ratio. Caches per symbol for 60 seconds to avoid hammering APIs.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Dict, Optional

import httpx

from app.config import AppConfig

_CACHE_TTL = 60.0  # seconds


@dataclass
class PriceSnapshot:
    symbol: str
    price: float
    bid: Optional[float]
    ask: Optional[float]
    price_5min_ago: float
    momentum_5m_pct: float      # (price - price_5min_ago) / price_5min_ago * 100
    rsi_current: float          # RSI-14 using last 14 one-minute bars
    volume_ratio: float         # current bar volume vs 20-bar average
    timestamp: datetime


def _rsi14(closes: list) -> float:
    """Standard RSI-14 from a list of close prices."""
    period = 14
    if len(closes) <= period:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = mean(gains) if gains else 0.0
    avg_loss = mean(losses) if losses else 0.0
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = ((avg_gain * (period - 1)) + max(d, 0.0)) / period
        avg_loss = ((avg_loss * (period - 1)) + max(-d, 0.0)) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class RealtimePriceFeed:
    def __init__(self) -> None:
        self._cache: Dict[str, tuple] = {}  # symbol → (snapshot, fetched_at)

    async def get_snapshot(
        self,
        symbol: str,
        asset_class: str,
        config: AppConfig,
    ) -> PriceSnapshot:
        now = datetime.now(timezone.utc).timestamp()
        cached = self._cache.get(symbol)
        if cached:
            snap, fetched_at = cached
            if now - fetched_at < _CACHE_TTL:
                return snap

        try:
            if asset_class == "equity":
                snap = await self._fetch_equity(symbol, config)
            else:
                snap = await self._fetch_crypto(symbol, config)
            self._cache[symbol] = (snap, now)
            return snap
        except Exception:
            if cached:
                # Return stale cache rather than crashing
                return cached[0]
            raise

    async def _fetch_equity(self, symbol: str, config: AppConfig) -> PriceSnapshot:
        headers = {
            "APCA-API-KEY-ID": config.alpaca_api_key_id or "",
            "APCA-API-SECRET-KEY": config.alpaca_api_secret_key or "",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Latest quote
            quote_resp = await client.get(
                f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest",
                headers=headers,
            )
            quote_resp.raise_for_status()
            quote_data = quote_resp.json()

            # 1-minute bars (last 30)
            bars_resp = await client.get(
                f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
                headers=headers,
                params={"timeframe": "1Min", "limit": 30},
            )
            bars_resp.raise_for_status()
            bars_data = bars_resp.json()

        quote = quote_data.get("quote") or {}
        bid = float(quote.get("bp") or 0) or None
        ask = float(quote.get("ap") or 0) or None
        if bid and ask:
            price = (bid + ask) / 2.0
        elif ask:
            price = ask
        elif bid:
            price = bid
        else:
            price = 0.0

        bars = bars_data.get("bars") or []
        closes = [float(b["c"]) for b in bars if "c" in b]
        volumes = [float(b.get("v", 0)) for b in bars]

        return _compute_snapshot(symbol, price, bid, ask, closes, volumes)

    async def _fetch_crypto(self, symbol: str, config: AppConfig) -> PriceSnapshot:
        # Convert "BTC/USD" → "BTCUSDT"
        binance_sym = symbol.replace("/", "").replace("USD", "USDT")
        base = (
            "https://testnet.binance.vision"
            if config.binance_testnet
            else "https://api.binance.com"
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            price_resp = await client.get(
                f"{base}/api/v3/ticker/price",
                params={"symbol": binance_sym},
            )
            price_resp.raise_for_status()
            price = float(price_resp.json().get("price", 0))

            bars_resp = await client.get(
                f"{base}/api/v3/klines",
                params={"symbol": binance_sym, "interval": "1m", "limit": 30},
            )
            bars_resp.raise_for_status()
            klines = bars_resp.json()

        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]

        return _compute_snapshot(symbol, price, None, None, closes, volumes)


def _compute_snapshot(
    symbol: str,
    price: float,
    bid: Optional[float],
    ask: Optional[float],
    closes: list,
    volumes: list,
) -> PriceSnapshot:
    price_5min_ago = closes[-6] if len(closes) >= 6 else (closes[0] if closes else price)
    momentum_5m = (
        ((price - price_5min_ago) / price_5min_ago * 100.0)
        if price_5min_ago
        else 0.0
    )
    rsi = _rsi14(closes) if closes else 50.0
    vol_ratio = 1.0
    if volumes:
        last_vol = volumes[-1]
        avg_vol = mean(volumes[-20:]) if len(volumes) >= 20 else mean(volumes)
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

    return PriceSnapshot(
        symbol=symbol,
        price=price,
        bid=bid,
        ask=ask,
        price_5min_ago=price_5min_ago,
        momentum_5m_pct=momentum_5m,
        rsi_current=rsi,
        volume_ratio=vol_ratio,
        timestamp=datetime.now(timezone.utc),
    )
