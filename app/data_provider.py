from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import httpx

from app.models import OHLCVPoint


TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
TWELVE_DATA_QUOTE_URL = "https://api.twelvedata.com/quote"


def fetch_quote_close_sync(symbol: str, api_key: str, timeout: float = 30.0) -> float:
    """
    Last quote close price (USD) for sizing whole-share orders (e.g. Alpaca shorts).
    Uses the same Twelve Data key as hourly candles.
    """
    params = {"symbol": symbol, "apikey": api_key}
    response = httpx.get(TWELVE_DATA_QUOTE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") == "error":
        raise ValueError(f"TwelveData quote error for {symbol}: {payload.get('message')}")
    close = payload.get("close")
    if close is None or close == "":
        raise ValueError(f"No close price in quote for {symbol}")
    return float(close)


class TwelveDataClient:
    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def fetch_hourly_30d(self, symbol: str) -> List[OHLCVPoint]:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=30)
        params = {
            "symbol": symbol,
            "interval": "1h",
            # Need enough bars for 30 calendar days (24/7 ≈ 720; stocks are fewer)
            "outputsize": "1500",
            "apikey": self.api_key,
            "format": "JSON",
            "timezone": "UTC",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(TWELVE_DATA_URL, params=params)
            response.raise_for_status()
            payload = response.json()

        if payload.get("status") == "error":
            raise ValueError(f"TwelveData error for {symbol}: {payload.get('message')}")

        rows = payload.get("values", [])
        if not rows:
            raise ValueError(f"No data returned for {symbol}")

        points: List[OHLCVPoint] = []
        for item in rows:
            dt = datetime.fromisoformat(item["datetime"]).replace(tzinfo=timezone.utc)
            if dt < start_dt:
                continue
            points.append(
                OHLCVPoint(
                    datetime=dt,
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item.get("volume") or 0),
                )
            )

        points.sort(key=lambda x: x.datetime)
        if not points:
            raise ValueError(f"No 30-day hourly data available for {symbol}")
        return points
