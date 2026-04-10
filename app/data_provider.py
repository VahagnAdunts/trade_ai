from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Sequence

import httpx

from app.models import OHLCVPoint


TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
TWELVE_DATA_QUOTE_URL = "https://api.twelvedata.com/quote"


def _dedupe_keys(keys: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for k in keys:
        k = (k or "").strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def fetch_quote_close_sync_try_keys(symbol: str, *api_keys: str, timeout: float = 30.0) -> float:
    """
    Try TwelveData quote with first key, then fall back to additional keys on any error.
    """
    keys = _dedupe_keys(api_keys)
    if not keys:
        raise ValueError("No TwelveData API keys provided for quote")
    last_exc: Exception | None = None
    for key in keys:
        try:
            return fetch_quote_close_sync(symbol, key, timeout=timeout)
        except Exception as exc:
            last_exc = exc
    assert last_exc is not None
    raise last_exc


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

        # Note: Twelve Data often returns volume=0 on free tier for many equities; see
        # feature_snapshot.data_quality in build_feature_context (warnings for LLMs + console).
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


class TwelveDataMultiKeyClient:
    """
    Tries the first Twelve Data API key, then optional additional keys on any error
    (rate limits, daily credits, etc.).
    """

    def __init__(
        self,
        *api_keys: str,
        timeout: float = 30.0,
        log_label: str = "[TwelveData]",
    ) -> None:
        self._keys = _dedupe_keys(api_keys)
        if not self._keys:
            raise ValueError("At least one Twelve Data API key is required")
        self.timeout = timeout
        self.log_label = log_label

    async def fetch_hourly_30d(self, symbol: str) -> List[OHLCVPoint]:
        last_exc: Exception | None = None
        for i, key in enumerate(self._keys):
            try:
                client = TwelveDataClient(api_key=key, timeout=self.timeout)
                return await client.fetch_hourly_30d(symbol)
            except Exception as exc:
                last_exc = exc
                if i + 1 < len(self._keys):
                    label = "primary" if i == 0 else "secondary"
                    print(
                        f"{self.log_label} {label} key failed for {symbol}: {exc} "
                        f"— trying next key",
                        flush=True,
                    )
        assert last_exc is not None
        raise last_exc
