from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load from repo .env or cwd when running as `python -m fast_news` from Tred_ai root
load_dotenv()


def _b(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class FastNewsConfig:
    """All flags are optional; disabled sources are skipped."""

    # Bluesky Jetstream (no app password; DIDs or handles resolved at runtime)
    bluesky_enabled: bool
    bluesky_jetstream_url: str
    # X API v2 filtered stream (Bearer only in app; tier must allow the endpoint)
    x_stream_enabled: bool
    x_bearer_token: str
    # Generic: receive pushes from your own poller (optional)
    print_json: bool  # line-delimited JSON to stdout (good for debugging)


def load_config() -> FastNewsConfig:
    return FastNewsConfig(
        bluesky_enabled=_b("FAST_NEWS_BLUESKY_ENABLED", False),
        bluesky_jetstream_url=(
            os.getenv("FAST_NEWS_JETSTREAM_URL", "").strip()
            or "wss://jetstream2.us-east.bsky.network/subscribe"
        ),
        x_stream_enabled=_b("FAST_NEWS_X_STREAM_ENABLED", False),
        x_bearer_token=(os.getenv("X_BEARER_TOKEN", "") or "").strip(),
        print_json=_b("FAST_NEWS_PRINT_JSON", True),
    )
