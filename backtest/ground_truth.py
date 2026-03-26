"""Realized 1-hour forward return vs directional label."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from app.models import OHLCVPoint

Side = Literal["long", "short"]


def forward_close_to_close(
    points: list, index_t: int
) -> Optional[Tuple[float, Side]]:
    """
    From bar at index_t to bar at index_t+1, return (simple_return, ideal_side).
    Return is (close[t+1]/close[t]) - 1.
    ideal_side = long if return > 0, short if < 0; None if exactly 0 (skip).
    """
    if index_t + 1 >= len(points):
        return None
    c0 = points[index_t].close
    c1 = points[index_t + 1].close
    if c0 == 0:
        return None
    r = (c1 / c0) - 1.0
    if r > 0:
        return r, "long"
    if r < 0:
        return r, "short"
    return None
