from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.models import OHLCVPoint


def build_feature_context(symbol: str, points: Sequence[OHLCVPoint]) -> Dict[str, Any]:
    if len(points) < 35:
        raise ValueError(f"Not enough points for feature context: {len(points)}")

    closes = [p.close for p in points]
    highs = [p.high for p in points]
    lows = [p.low for p in points]
    volumes = [max(float(p.volume), 0.0) for p in points]

    rsi14 = _rsi(closes, 14)
    macd_line, macd_signal, macd_hist = _macd(closes)
    bb_mid, bb_upper, bb_lower = _bollinger(closes, 20, 2.0)
    atr14 = _atr(highs, lows, closes, 14)
    obv_series = _obv(closes, volumes)
    vwap, vwap_basis, vwap_volume_weighted, vwap_session_bars, vwap_session_date_utc = _vwap_session(
        points
    )
    dq_volume, dq_warnings = _volume_data_quality(volumes, vwap_volume_weighted=vwap_volume_weighted)

    last_close = closes[-1]
    last_volume = volumes[-1]
    returns = {
        "ret_1h_pct": _ret_pct(closes, 1),
        "ret_4h_pct": _ret_pct(closes, 4),
        "ret_24h_pct": _ret_pct(closes, 24),
        "ret_7d_pct": _ret_pct(closes, 24 * 7),
    }

    vol_z = _zscore(volumes, last_volume, 30)
    bb_width_pct = ((bb_upper - bb_lower) / bb_mid * 100.0) if bb_mid else 0.0
    sr = _support_resistance(closes, 48)

    ema_1h_20 = _ema(closes, 20)
    bias_1h = _price_vs_ema_bias(last_close, ema_1h_20)

    bars_4h = _aggregate_to_4h(points)
    bars_1d = _aggregate_to_daily(points)
    c4 = [p.close for p in bars_4h]
    c1d = [p.close for p in bars_1d]

    rsi_4h = _rsi(c4, 14) if c4 else 50.0
    ema_4h_20 = _ema(c4, 20) if c4 else 0.0
    last_4h = c4[-1] if c4 else last_close
    bias_4h = _price_vs_ema_bias(last_4h, ema_4h_20)
    trend_4h = _bias_to_trend(bias_4h)

    rsi_1d = _rsi(c1d, 14) if c1d else 50.0
    ema_1d_20 = _ema(c1d, 20) if c1d else 0.0
    last_1d = c1d[-1] if c1d else last_close
    bias_1d = _price_vs_ema_bias(last_1d, ema_1d_20)
    trend_1d = _bias_to_trend(bias_1d)
    trend_1h = _bias_to_trend(bias_1h)

    timeframe_alignment = _build_timeframe_alignment(
        trend_1h=trend_1h,
        trend_4h=trend_4h,
        trend_1d=trend_1d,
        bias_1h=bias_1h,
        bias_4h=bias_4h,
        bias_1d=bias_1d,
    )

    features: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": "1h",
        "bars": len(points),
        "as_of_utc": points[-1].datetime.isoformat(),
        "price": {
            "last_close": _r(last_close),
            "ema_20": _r(ema_1h_20),
            "ema_50": _r(_ema(closes, 50)),
            "dist_to_ema20_pct": _r(_dist_pct(last_close, ema_1h_20)),
            "dist_to_ema50_pct": _r(_dist_pct(last_close, _ema(closes, 50))),
            "vwap": _r(vwap),
            "vwap_deviation_pct": _r(_dist_pct(last_close, vwap)),
            "vwap_basis": vwap_basis,
            "vwap_scope": (
                "session_vwap_utc_calendar_day_of_last_bar"
                if vwap_basis == "session_day"
                else (
                    "last_24h_fallback"
                    if vwap_basis == "last_24h_fallback"
                    else "none"
                )
            ),
            "vwap_session_bars": vwap_session_bars,
            "vwap_session_date_utc": vwap_session_date_utc,
        },
        "momentum": {
            "rsi_14": _r(rsi14),
            "macd_line": _r(macd_line),
            "macd_signal": _r(macd_signal),
            "macd_hist": _r(macd_hist),
            **{k: _r(v) for k, v in returns.items()},
        },
        "volatility": {
            "atr_14": _r(atr14),
            "atr_14_pct_of_price": _r((atr14 / last_close * 100.0) if last_close else 0.0),
            "bb_mid_20": _r(bb_mid),
            "bb_upper_20_2": _r(bb_upper),
            "bb_lower_20_2": _r(bb_lower),
            "bb_zscore_20": _r(_bb_zscore(last_close, bb_mid, bb_upper, bb_lower)),
            "bb_width_pct": _r(bb_width_pct),
            "realized_vol_24h_pct": _r(_realized_vol_pct(closes, 24)),
        },
        "volume": {
            "last_volume": _r(last_volume),
            "volume_zscore_30": _r(vol_z),
            "obv": _r(obv_series[-1]),
            "obv_slope_24": _r(_slope(obv_series, 24)),
        },
        "structure": {
            "support_48h": _r(sr[0]),
            "resistance_48h": _r(sr[1]),
            "distance_to_support_pct": _r(_dist_pct(last_close, sr[0])),
            "distance_to_resistance_pct": _r(_dist_pct(last_close, sr[1])),
            "near_breakout_up": _near_breakout(last_close, sr[1]),
            "near_breakout_down": _near_breakout(sr[0], last_close),
        },
        "data_quality": {
            "volume": dq_volume,
            "warnings": dq_warnings,
        },
        "timeframe_4h": {
            "bars": len(bars_4h),
            "rsi_14": _r(rsi_4h),
            "ema_20": _r(ema_4h_20),
            "last_close": _r(last_4h),
            "dist_to_ema20_pct": _r(_dist_pct(last_4h, ema_4h_20)),
            "trend": trend_4h,
        },
        "timeframe_1d": {
            "bars": len(bars_1d),
            "rsi_14": _r(rsi_1d),
            "ema_20": _r(ema_1d_20),
            "last_close": _r(last_1d),
            "dist_to_ema20_pct": _r(_dist_pct(last_1d, ema_1d_20)),
            "trend": trend_1d,
        },
        "timeframe_alignment": timeframe_alignment,
    }

    return features


def recent_bars_snapshot(points: Sequence[OHLCVPoint], count: int = 16) -> List[Dict[str, Any]]:
    snap = []
    for p in points[-count:]:
        snap.append(
            {
                "dt": p.datetime.isoformat(),
                "o": _r(p.open),
                "h": _r(p.high),
                "l": _r(p.low),
                "c": _r(p.close),
                "v": _r(p.volume),
            }
        )
    return snap


def _aggregate_to_4h(points: Sequence[OHLCVPoint]) -> List[OHLCVPoint]:
    """Collapse consecutive 4×1h bars into one 4h OHLCV bar (aligned from series start)."""
    out: List[OHLCVPoint] = []
    for i in range(0, len(points), 4):
        chunk = points[i : i + 4]
        if not chunk:
            continue
        last = chunk[-1]
        o = chunk[0].open
        h = max(p.high for p in chunk)
        l = min(p.low for p in chunk)
        c = last.close
        v = sum(max(float(p.volume), 0.0) for p in chunk)
        out.append(
            OHLCVPoint(
                datetime=last.datetime,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
            )
        )
    return out


def _aggregate_to_daily(points: Sequence[OHLCVPoint]) -> List[OHLCVPoint]:
    """One OHLCV bar per calendar day (UTC date of bar timestamp)."""
    by_day: Dict[str, List[OHLCVPoint]] = defaultdict(list)
    for p in points:
        by_day[p.datetime.date().isoformat()].append(p)
    out: List[OHLCVPoint] = []
    for day in sorted(by_day.keys()):
        chunk = sorted(by_day[day], key=lambda x: x.datetime)
        last = chunk[-1]
        o = chunk[0].open
        h = max(p.high for p in chunk)
        l = min(p.low for p in chunk)
        c = last.close
        v = sum(max(float(p.volume), 0.0) for p in chunk)
        out.append(
            OHLCVPoint(
                datetime=last.datetime,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
            )
        )
    return out


def _price_vs_ema_bias(close: float, ema: float) -> int:
    """+1 price above EMA, -1 below, 0 neutral or invalid."""
    if ema <= 0 or close <= 0:
        return 0
    if close > ema:
        return 1
    if close < ema:
        return -1
    return 0


def _bias_to_trend(bias: int) -> str:
    if bias > 0:
        return "bullish"
    if bias < 0:
        return "bearish"
    return "neutral"


def _tf_biases_agree(b1: int, b2: int) -> bool:
    """True when directional bias vs EMA20 matches (including both neutral)."""
    return b1 == b2


def _build_timeframe_alignment(
    *,
    trend_1h: str,
    trend_4h: str,
    trend_1d: str,
    bias_1h: int,
    bias_4h: int,
    bias_1d: int,
) -> Dict[str, Any]:
    """
    Independent trends: 1h from hourly closes vs EMA20; 4h from aggregated 4h bars vs EMA20;
    daily from UTC-calendar daily bars vs EMA20. Conflicting directions imply a weak setup.
    """
    alignment_score, all_aligned = _timeframe_alignment_score_and_flag(
        trend_1h, trend_4h, trend_1d
    )
    return {
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "trend_1d": trend_1d,
        "alignment_score": _r(alignment_score),
        "all_aligned": all_aligned,
        "1h_4h_agree": _tf_biases_agree(bias_1h, bias_4h),
        "1h_1d_agree": _tf_biases_agree(bias_1h, bias_1d),
    }


def _timeframe_alignment_score_and_flag(
    trend_1h: str,
    trend_4h: str,
    trend_1d: str,
) -> Tuple[float, bool]:
    """
    alignment_score = max(count bullish, count bearish, count neutral) / 3 (e.g. one bull, one bear, one neutral → 1/3).
    all_aligned = all three bullish or all three bearish (strong multi-TF agreement; excludes all-neutral).
    """
    trends = [trend_1h, trend_4h, trend_1d]
    c = Counter(trends)
    max_count = max(c.values()) if trends else 0
    alignment_score = max_count / 3.0
    all_aligned = (c.get("bullish", 0) == 3) or (c.get("bearish", 0) == 3)
    return alignment_score, all_aligned


def _ret_pct(closes: Sequence[float], lookback: int) -> float:
    if len(closes) <= lookback:
        return 0.0
    prev = closes[-1 - lookback]
    if prev == 0:
        return 0.0
    return (closes[-1] / prev - 1.0) * 100.0


def _ema(values: Sequence[float], period: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (period + 1.0)
    e = values[0]
    for v in values[1:]:
        e = alpha * v + (1 - alpha) * e
    return e


def _rsi(closes: Sequence[float], period: int) -> float:
    if len(closes) <= period:
        return 50.0
    gains = []
    losses = []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = mean(gains)
    avg_loss = mean(losses)
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(closes: Sequence[float]) -> Tuple[float, float, float]:
    if len(closes) < 35:
        return 0.0, 0.0, 0.0
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    macd = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema_series(macd, 9)
    return macd[-1], signal[-1], macd[-1] - signal[-1]


def _ema_series(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return [0.0]
    alpha = 2.0 / (period + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


def _bollinger(closes: Sequence[float], period: int, std_mult: float) -> Tuple[float, float, float]:
    if len(closes) < period:
        c = closes[-1]
        return c, c, c
    window = closes[-period:]
    mid = mean(window)
    sd = pstdev(window) if len(window) > 1 else 0.0
    return mid, mid + std_mult * sd, mid - std_mult * sd


def _atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int) -> float:
    if len(closes) < 2:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    if not trs:
        return 0.0
    if len(trs) < period:
        return mean(trs)
    atr = mean(trs[:period])
    for tr in trs[period:]:
        atr = ((atr * (period - 1)) + tr) / period
    return atr


def _obv(closes: Sequence[float], volumes: Sequence[float]) -> List[float]:
    out = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            out.append(out[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            out.append(out[-1] - volumes[i])
        else:
            out.append(out[-1])
    return out


def _nonzero_fraction(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v > 0) / len(values)


def _volume_data_quality(
    volumes: Sequence[float],
    *,
    vwap_volume_weighted: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Twelve Data (especially free tier) often returns volume=0 for many symbols.
    Surfaces explicit warnings so volume-derived features are not interpreted blindly.
    """
    n = len(volumes)
    warnings: List[str] = []

    last24 = volumes[-24:] if n >= 24 else list(volumes)
    last72 = volumes[-72:] if n >= 72 else list(volumes)

    nz_full = _nonzero_fraction(volumes)
    nz_72 = _nonzero_fraction(last72)
    nz_24 = _nonzero_fraction(last24)

    if n >= 1 and nz_24 == 0.0:
        warnings.append(
            "Volume is zero on all of the last 24 hourly bars — OBV is flat, volume z-score is 0, "
            "and VWAP is not true volume-weighted price (common on Twelve Data free tier for some tickers)."
        )
    elif not vwap_volume_weighted:
        warnings.append(
            "VWAP is not volume-weighted in the session window (total volume is zero); "
            "vwap/vwap_deviation_pct use typical price of the last bar in that window, not true VWAP."
        )

    if n >= 24 and nz_24 > 0.0 and nz_72 < 0.15:
        warnings.append(
            f"Only ~{nz_72 * 100:.0f}% of the last 72 bars have nonzero volume — OBV, volume z-score, "
            "and VWAP are weakly informative; prefer price/volatility indicators."
        )
    elif nz_full < 0.25 and nz_24 > 0.0:
        warnings.append(
            f"Only ~{nz_full * 100:.0f}% of bars in this series have nonzero volume — treat volume metrics with caution."
        )

    trusted = (
        vwap_volume_weighted
        and nz_72 >= 0.20
        and nz_24 > 0.0
        and nz_full >= 0.15
    )

    meta: Dict[str, Any] = {
        "nonzero_fraction_full_series": round(nz_full, 4),
        "nonzero_fraction_last_72_bars": round(nz_72, 4),
        "nonzero_fraction_last_24_bars": round(nz_24, 4),
        "vwap_volume_weighted": vwap_volume_weighted,
        "volume_features_trusted": trusted,
    }
    return meta, warnings


def _vwap_session(points: Sequence[OHLCVPoint]) -> Tuple[float, str, bool, int, str]:
    """
    Session VWAP — not a 30-day or full-history blend.

    Uses only bars on the same UTC calendar date as the latest bar (institutional-style
    session VWAP on hourly data). If that set is empty, falls back to the last 24 bars.

    Returns (vwap, basis_label, volume_weighted, session_bar_count, session_date_utc YYYY-MM-DD).
    volume_weighted is False if sum(volume)==0 (caller must not treat as true VWAP).
    """
    if not points:
        return 0.0, "none", False, 0, ""
    last = points[-1]
    day = last.datetime.date()
    session = [p for p in points if p.datetime.date() == day]
    basis = "session_day"
    if not session:
        session = list(points[-24:]) if len(points) >= 24 else list(points)
        basis = "last_24h_fallback"

    session_date = session[-1].datetime.date().isoformat() if session else day.isoformat()

    numer = 0.0
    denom = 0.0
    for p in session:
        typical = (p.high + p.low + p.close) / 3.0
        vol = max(float(p.volume), 0.0)
        numer += typical * vol
        denom += vol
    if denom > 0:
        return numer / denom, basis, True, len(session), session_date
    # No volume in window: use typical of the last bar in the window
    p = session[-1]
    return (p.high + p.low + p.close) / 3.0, basis, False, len(session), session_date


def _zscore(values: Sequence[float], current: float, lookback: int) -> float:
    window = list(values[-lookback:]) if len(values) >= lookback else list(values)
    if len(window) < 2:
        return 0.0
    mu = mean(window)
    sd = pstdev(window)
    if sd == 0:
        return 0.0
    return (current - mu) / sd


def _support_resistance(closes: Sequence[float], lookback: int) -> Tuple[float, float]:
    window = list(closes[-lookback:]) if len(closes) >= lookback else list(closes)
    return min(window), max(window)


def _slope(series: Sequence[float], lookback: int) -> float:
    if len(series) < 2:
        return 0.0
    window = list(series[-lookback:]) if len(series) >= lookback else list(series)
    n = len(window)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = mean(window)
    num = 0.0
    den = 0.0
    for i, y in enumerate(window):
        dx = i - x_mean
        num += dx * (y - y_mean)
        den += dx * dx
    return num / den if den else 0.0


def _bb_zscore(last_close: float, mid: float, upper: float, lower: float) -> float:
    width = upper - lower
    if width <= 0:
        return 0.0
    return (last_close - mid) / (width / 2.0)


def _realized_vol_pct(closes: Sequence[float], lookback: int) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    rets = []
    for i in range(len(closes) - lookback, len(closes)):
        prev = closes[i - 1]
        if prev <= 0:
            continue
        rets.append(math.log(closes[i] / prev))
    if len(rets) < 2:
        return 0.0
    return pstdev(rets) * 100.0


def _near_breakout(a: float, b: float) -> bool:
    if b == 0:
        return False
    return abs(a - b) / abs(b) <= 0.003


def _dist_pct(price: float, ref: float) -> float:
    if ref == 0:
        return 0.0
    return (price / ref - 1.0) * 100.0


def _r(v: Optional[float], ndigits: int = 4) -> float:
    if v is None:
        return 0.0
    return round(float(v), ndigits)
