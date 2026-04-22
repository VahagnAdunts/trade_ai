from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class IngestSource(str, Enum):
    """Where an event was observed (extensible)."""

    BLUESKY = "bluesky"
    X = "x"
    WEBHOOK = "webhook"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PostEvent:
    """Normalised post or headline, suitable for sinks (log, DB, bus)."""

    id: str
    source: IngestSource
    text: str
    author_handle: str
    url: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    observed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
