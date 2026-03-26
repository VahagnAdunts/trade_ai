from __future__ import annotations

import math
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
    vwap = _vwap(points)

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

    features: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": "1h",
        "bars": len(points),
        "as_of_utc": points[-1].datetime.isoformat(),
        "price": {
            "last_close": _r(last_close),
            "ema_20": _r(_ema(closes, 20)),
            "ema_50": _r(_ema(closes, 50)),
            "dist_to_ema20_pct": _r(_dist_pct(last_close, _ema(closes, 20))),
            "dist_to_ema50_pct": _r(_dist_pct(last_close, _ema(closes, 50))),
            "vwap": _r(vwap),
            "vwap_deviation_pct": _r(_dist_pct(last_close, vwap)),
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


def _vwap(points: Sequence[OHLCVPoint]) -> float:
    numer = 0.0
    denom = 0.0
    for p in points:
        typical = (p.high + p.low + p.close) / 3.0
        vol = max(float(p.volume), 0.0)
        numer += typical * vol
        denom += vol
    return numer / denom if denom else points[-1].close


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
