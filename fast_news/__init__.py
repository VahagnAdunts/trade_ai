"""
fast_news — subproject: ingest posts / headlines with minimum latency.

Design: prefer long-lived WebSocket or HTTP streaming (e.g. Bluesky Jetstream,
X API v2 filtered stream) over poll loops. Sinks (stdout, webhook, your DB) are
separate from sources.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
