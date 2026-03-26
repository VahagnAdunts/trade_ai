"""Fetch a historical hourly range from TwelveData (separate from app.data_provider)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import httpx

from app.models import OHLCVPoint

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"


async def fetch_hourly_range(
    api_key: str,
    symbol: str,
    start: datetime,
    end: datetime,
    timeout: float = 60.0,
) -> List[OHLCVPoint]:
    """
    Fetch 1h candles between start and end (UTC).
    TwelveData accepts start_date / end_date (YYYY-MM-DD or full datetime).
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    params = {
        "symbol": symbol,
        "interval": "1h",
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(TWELVE_DATA_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    if payload.get("status") == "error":
        raise ValueError(f"TwelveData error for {symbol}: {payload.get('message')}")

    rows = payload.get("values", [])
    if not rows:
        raise ValueError(f"No data returned for {symbol} in range")

    points: List[OHLCVPoint] = []
    for item in rows:
        dt = datetime.fromisoformat(item["datetime"]).replace(tzinfo=timezone.utc)
        if dt < start - timedelta(hours=1) or dt > end + timedelta(hours=1):
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
    return points


def slice_lookback_window(
    all_points: List[OHLCVPoint],
    as_of_index: int,
    lookback_days: int = 30,
) -> List[OHLCVPoint]:
    """Bars to show the model: completed hourly data up to and including all_points[as_of_index]."""
    if as_of_index < 0 or as_of_index >= len(all_points):
        raise IndexError("as_of_index out of range")
    t_end = all_points[as_of_index].datetime
    t_start = t_end - timedelta(days=lookback_days)
    return [p for p in all_points if t_start <= p.datetime <= t_end]
