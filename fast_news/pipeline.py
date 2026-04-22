from __future__ import annotations

import asyncio
from typing import List

from fast_news.config import load_config
from fast_news.models import IngestSource, PostEvent
from fast_news.sinks import print_json_line
from fast_news.sources.base import OnPost
from fast_news.sources.bluesky_jetstream import BlueskyJetstreamSource
from fast_news.sources.x_filtered_stream import XFilteredStreamSource


def _build_sink() -> OnPost:
    cfg = load_config()
    if cfg.print_json:
        return print_json_line
    return _null_sink


async def _null_sink(_: PostEvent) -> None:
    pass


def _build_sources() -> List[object]:
    cfg = load_config()
    out: List[object] = []
    if cfg.bluesky_enabled:
        out.append(BlueskyJetstreamSource())
    if cfg.x_stream_enabled and cfg.x_bearer_token:
        out.append(XFilteredStreamSource())
    return out


async def _run_one(src: object, on_event: OnPost) -> None:
    name = getattr(src, "name", type(src).__name__)
    try:
        run = getattr(src, "run")
        await run(on_event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[fast_news] source {name!r} error: {exc!r}", flush=True)


async def run_ingest() -> None:
    """
    Run all enabled sources in parallel; each post is sent to the sink.
    Stubs raise until you implement the WebSocket/stream clients.
    """
    on_event = _build_sink()
    sources = _build_sources()
    if not sources:
        print(
            "[fast_news] No sources enabled. Set FAST_NEWS_BLUESKY_ENABLED=true "
            "and/or FAST_NEWS_X_STREAM_ENABLED=true (with X_BEARER_TOKEN) in .env",
            flush=True,
        )
        return

    await asyncio.gather(
        *(_run_one(s, on_event) for s in sources),
        return_exceptions=True,
    )


# Dev helper: print one synthetic line without any network
async def run_demo() -> None:
    await print_json_line(
        PostEvent(
            id="demo:1",
            source=IngestSource.UNKNOWN,
            text="fast_news: add real sources under fast_news/sources/",
            author_handle="demo",
        )
    )
