from __future__ import annotations

"""
Bluesky Jetstream: public WebSocket firehose, filtered by collection + DIDs.
Low latency (typically a few seconds or less) compared to HTTP polling.

Implementation: wire ``websockets`` to your subscribe URL, parse JSON commit
events for ``app.bsky.feed.post`` creates, map DID → handle, build PostEvent.
See Bluesky's jetstream repo for the exact event shape.

This module is a **stub** — add handle list + DID resolution, then connect.
"""

# Placeholder: real implementation goes here (moved from removed app news package).


class BlueskyJetstreamSource:
    @property
    def name(self) -> str:
        return "bluesky_jetstream"

    async def run(self, on_event) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError(
            "Bluesky Jetstream: implement connection + post parsing in "
            "fast_news.sources.bluesky_jetstream (see Jetstream docs)."
        )
