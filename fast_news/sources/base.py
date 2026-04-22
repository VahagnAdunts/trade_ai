from __future__ import annotations

from typing import Awaitable, Callable, Protocol

from fast_news.models import PostEvent

OnPost = Callable[[PostEvent], Awaitable[None]]


class PostSource(Protocol):
    """A long-running producer that calls ``on_event`` for each post."""

    @property
    def name(self) -> str: ...

    async def run(self, on_event: OnPost) -> None:
        """
        Block until the source is done or cancelled.
        Implementations should reconnect internally on failure when appropriate.
        """
        ...
