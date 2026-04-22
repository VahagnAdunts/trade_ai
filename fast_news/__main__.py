from __future__ import annotations

import argparse
import asyncio
import sys

from fast_news.pipeline import run_demo, run_ingest


def main() -> None:
    p = argparse.ArgumentParser(prog="fast_news", description="Low-latency post ingest (subproject)")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Print one sample ndjson line and exit (no network).",
    )
    args = p.parse_args()
    if args.demo:
        asyncio.run(run_demo())
    else:
        try:
            asyncio.run(run_ingest())
        except KeyboardInterrupt:
            print("\n[fast_news] stopped", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
