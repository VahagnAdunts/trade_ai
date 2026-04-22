from __future__ import annotations

"""
X (Twitter) API v2 **filtered stream** (Bearer, app context).
Near real-time; requires a developer project with access to this endpoint
(not available on all tiers; may return 403).
"""


class XFilteredStreamSource:
    @property
    def name(self) -> str:
        return "x_filtered_stream"

    async def run(self, on_event) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError(
            "X filtered stream: implement rules sync + stream reader in "
            "fast_news.sources.x_filtered_stream (X API v2 documentation)."
        )
