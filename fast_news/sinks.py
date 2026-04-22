from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any

from fast_news.models import PostEvent


def _json_serial(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError


async def print_json_line(event: PostEvent) -> None:
    """Default sink: one JSON object per line (ndjson) on stdout."""
    payload = {
        "id": event.id,
        "source": event.source.value,
        "text": event.text,
        "author": event.author_handle,
        "url": event.url,
        "observed_at": event.observed_at.isoformat(),
    }
    line = json.dumps(payload, default=_json_serial, ensure_ascii=False)
    print(line, flush=True, file=sys.stdout)
