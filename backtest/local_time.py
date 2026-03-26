"""Parse user local datetime (e.g. European DD.MM.YYYY) and anchor to hourly bar open in UTC."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List

from zoneinfo import ZoneInfo

from app.models import OHLCVPoint


def parse_at_datetime(s: str, tz_name: str) -> datetime:
    """
    Returns timezone-aware datetime in the given IANA zone.

    Supported forms:
    - 25.02.2025 14:00  (European day.month.year)
    - 2025-02-25 14:00
    - ISO 8601 with optional offset
    """
    s = s.strip()
    tz = ZoneInfo(tz_name)

    m = re.match(
        r"^(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$",
        s,
    )
    if m:
        d, mo, y, h, mi = (int(x) for x in m.groups()[:5])
        return datetime(y, mo, d, h, mi, tzinfo=tz)

    m2 = re.match(
        r"^(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$",
        s,
    )
    if m2:
        y, mo, d, h, mi = (int(x) for x in m2.groups()[:5])
        return datetime(y, mo, d, h, mi, tzinfo=tz)

    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return dt


def floor_to_hour_local(local_dt: datetime) -> datetime:
    """Start of the hour in the same timezone (for '2pm' bar anchoring)."""
    return local_dt.replace(minute=0, second=0, microsecond=0)


def hour_start_utc(local_dt: datetime) -> datetime:
    """UTC instant for the start of the hour containing local_dt (after flooring)."""
    floored = floor_to_hour_local(local_dt)
    return floored.astimezone(timezone.utc)


def find_decision_bar_index(points: List[OHLCVPoint], hour_open_utc: datetime) -> int:
    """
    Last bar whose open time is <= hour_open_utc (TwelveData bar timestamps are typically bar open, UTC).
    """
    hour_open_utc = hour_open_utc.astimezone(timezone.utc)
    decision_i = None
    for j in range(len(points)):
        t = points[j].datetime
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        else:
            t = t.astimezone(timezone.utc)
        if t <= hour_open_utc:
            decision_i = j
        else:
            break
    if decision_i is None:
        raise ValueError("No hourly bar at or before the target hour (check date range / symbol).")
    return decision_i
