from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from app.data_provider import TwelveDataClient
from app.models import OHLCVPoint

# Twelve Data: spot VIX index is not a valid time_series symbol on this API; VIXY is a real
# CBOE-listed short-term VIX futures ETF (volatility regime proxy). Same hourly math as SPY/QQQ.
EQUITY_BENCHMARKS = ("SPY", "QQQ", "VIXY")
CRYPTO_BENCHMARKS = ("BTC/USD", "ETH/USD")


def _ret_pct_or_none(closes: Sequence[float], lookback: int) -> Optional[float]:
    """Percent change from close at -lookback to last close; None if not enough bars or bad prev."""
    if len(closes) <= lookback:
        return None
    prev = closes[-1 - lookback]
    last = closes[-1]
    if prev == 0:
        return None
    return (last / prev - 1.0) * 100.0


def _closes_at_or_before(points: Sequence[OHLCVPoint], t_end: datetime) -> List[float]:
    aligned = [p for p in points if p.datetime <= t_end]
    if not aligned:
        return []
    return [p.close for p in aligned]


def _horizon_metrics(closes: List[float]) -> Dict[str, Any]:
    """Realized returns from aligned closes; missing horizons are omitted (not zero-filled)."""
    out: Dict[str, Any] = {}
    if not closes:
        return out
    last = closes[-1]
    out["last_close"] = round(last, 6)
    for key, lb in (("ret_1h_pct", 1), ("ret_4h_pct", 4), ("ret_24h_pct", 24), ("ret_7d_pct", 24 * 7)):
        v = _ret_pct_or_none(closes, lb)
        if v is not None:
            out[key] = round(v, 4)
    return out


def _metrics_from_series(
    series: Optional[List[OHLCVPoint]],
    t_end: datetime,
) -> Optional[Dict[str, Any]]:
    if not series:
        return None
    aligned = [p for p in series if p.datetime <= t_end]
    if not aligned:
        return None
    closes = [p.close for p in aligned]
    m = _horizon_metrics(closes)
    if not m:
        return None
    t_last = aligned[-1].datetime
    base: Dict[str, Any] = {"as_of_utc": t_last.isoformat()}
    base["last_close"] = m.pop("last_close")
    base.update(m)
    return base


async def fetch_equity_benchmarks(client: TwelveDataClient) -> Dict[str, Any]:
    """Fetch SPY, QQQ, VIX hourly series once per run. Failed symbols are recorded, not invented."""

    async def one(sym: str) -> tuple:
        try:
            pts = await client.fetch_hourly_30d(sym)
            return sym, pts, None
        except Exception as exc:
            return sym, None, str(exc)

    results = await asyncio.gather(*(one(s) for s in EQUITY_BENCHMARKS))
    out: Dict[str, Any] = {"series": {}, "fetch_errors": {}}
    for sym, pts, err in results:
        if err is not None:
            out["fetch_errors"][sym] = err
        else:
            out["series"][sym] = pts
    return out


async def fetch_crypto_benchmarks(client: TwelveDataClient) -> Dict[str, Any]:
    async def one(sym: str) -> tuple:
        try:
            pts = await client.fetch_hourly_30d(sym)
            return sym, pts, None
        except Exception as exc:
            return sym, None, str(exc)

    results = await asyncio.gather(*(one(s) for s in CRYPTO_BENCHMARKS))
    out: Dict[str, Any] = {"series": {}, "fetch_errors": {}}
    for sym, pts, err in results:
        if err is not None:
            out["fetch_errors"][sym] = err
        else:
            out["series"][sym] = pts
    return out


def build_equity_regime_payload(
    symbol: str,
    points: Sequence[OHLCVPoint],
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    """Market/regime block for US equity names: SPY, QQQ, VIXY vs same as-of bar as the stock."""
    t_end = points[-1].datetime
    series_map: Dict[str, List[OHLCVPoint]] = cache.get("series") or {}
    errors: Dict[str, str] = dict(cache.get("fetch_errors") or {})

    payload: Dict[str, Any] = {
        "asset_class": "equity",
        "symbol": symbol,
        "benchmarks_note": "Hourly returns aligned to the last bar of the symbol (UTC).",
        "fetch_errors": errors,
    }

    spy = _metrics_from_series(series_map.get("SPY"), t_end)
    qqq = _metrics_from_series(series_map.get("QQQ"), t_end)
    vixy = _metrics_from_series(series_map.get("VIXY"), t_end)

    if spy:
        payload["SPY"] = spy
    if qqq:
        payload["QQQ"] = qqq
    if vixy:
        payload["VIXY"] = vixy

    payload["available"] = bool(spy or qqq or vixy)
    return payload


def build_crypto_regime_payload(
    symbol: str,
    points: Sequence[OHLCVPoint],
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    """
    BTC/ETH benchmarks vs the traded pair. Uses the same hourly bars as the symbol for the alt leg.
    No synthetic data: if BTC or ETH series failed to load, that leg is omitted and fetch_errors explains.
    """
    t_end = points[-1].datetime
    sym_u = symbol.strip().upper()
    series_map: Dict[str, List[OHLCVPoint]] = cache.get("series") or {}
    errors: Dict[str, str] = dict(cache.get("fetch_errors") or {})

    alt_closes = _closes_at_or_before(points, t_end)
    alt_metrics = _horizon_metrics(alt_closes) if alt_closes else {}
    if alt_metrics:
        t_last = points[-1].datetime
        alt_block = {"as_of_utc": t_last.isoformat(), **alt_metrics}

    payload: Dict[str, Any] = {
        "asset_class": "crypto",
        "pair": sym_u,
        "benchmarks_note": "BTC/ETH from Twelve Data hourly series; alt leg from this pair's bars.",
        "fetch_errors": errors,
    }

    if alt_metrics:
        payload["this_pair"] = alt_block

    btc_pts = series_map.get("BTC/USD")
    eth_pts = series_map.get("ETH/USD")

    btc_m = _metrics_from_series(btc_pts, t_end) if sym_u != "BTC/USD" else None
    eth_m = _metrics_from_series(eth_pts, t_end) if sym_u != "ETH/USD" else None

    if sym_u == "BTC/USD":
        payload["note"] = "Pair is BTC/USD; ETH/USD is the cross benchmark (no duplicate BTC leg)."
    elif sym_u == "ETH/USD":
        payload["note"] = "Pair is ETH/USD; BTC/USD is the cross benchmark (no duplicate ETH leg)."

    if btc_m:
        payload["BTC/USD"] = btc_m
    if eth_m:
        payload["ETH/USD"] = eth_m

    rel_out: Dict[str, float] = {}
    if alt_metrics:
        for h in ("ret_1h_pct", "ret_4h_pct", "ret_24h_pct", "ret_7d_pct"):
            av = alt_metrics.get(h)
            if sym_u != "BTC/USD" and isinstance(payload.get("BTC/USD"), dict):
                bv = payload["BTC/USD"].get(h)
                if av is not None and bv is not None:
                    rel_out[f"this_pair_minus_BTC_{h}"] = round(av - bv, 4)
            if sym_u != "ETH/USD" and isinstance(payload.get("ETH/USD"), dict):
                ev = payload["ETH/USD"].get(h)
                if av is not None and ev is not None:
                    rel_out[f"this_pair_minus_ETH_{h}"] = round(av - ev, 4)
    if rel_out:
        payload["relative_vs_benchmarks"] = rel_out

    payload["available"] = bool(
        payload.get("this_pair") or payload.get("BTC/USD") or payload.get("ETH/USD")
    )
    return payload


async def load_regime_cache(client: TwelveDataClient, *, crypto: bool) -> Dict[str, Any]:
    if crypto:
        return await fetch_crypto_benchmarks(client)
    return await fetch_equity_benchmarks(client)


def build_market_regime_payload(
    symbol: str,
    points: Sequence[OHLCVPoint],
    cache: Dict[str, Any],
    *,
    crypto: bool,
) -> Dict[str, Any]:
    if crypto:
        return build_crypto_regime_payload(symbol, points, cache)
    return build_equity_regime_payload(symbol, points, cache)
