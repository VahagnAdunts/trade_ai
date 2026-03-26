"""
Run from project root:

  # New York 2pm on 25 Feb 2025 — all top-100 symbols, 30d hourly context, LLMs, terminal table
  python -m backtest --at "25.02.2025 14:00" --tz America/New_York

  # Use engineered-features context only
  python -m backtest --at "25.02.2025 14:00" --context-mode features

  # Same with European-style date and explicit symbol list
  python -m backtest --at "2025-02-25 14:00" --symbols AAPL,MSFT

  # Calendar day (UTC midnight) mode
  python -m backtest --as-of 2024-06-15 --symbol AAPL

  # Range mode
  python -m backtest --start 2024-01-02 --end 2024-01-09 --symbol AAPL --max-evals 3

Requires `.env` with same API keys as the main app.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.runner import async_main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Backtest: production LLMs + app.engine._candles_to_context. "
            "Use --at for a local datetime (e.g. 2pm New York); default symbols = top 100."
        )
    )
    p.add_argument(
        "--context-mode",
        default=None,
        choices=["raw", "hybrid", "features"],
        help=(
            "Prompt context format. "
            "Default: CONTEXT_MODE from .env (fallback hybrid)."
        ),
    )
    p.add_argument(
        "--at",
        default=None,
        metavar='"DATE TIME"',
        help=(
            "Local wall time for the decision hour, e.g. '25.02.2025 14:00' or '2025-02-25 14:00'. "
            "Uses 30d hourly bars ending at that hour, then compares to the next hour. "
            "Default symbols: top 100 (override with --symbol or --symbols)."
        ),
    )
    p.add_argument(
        "--tz",
        default="America/New_York",
        help="IANA timezone for --at (default: America/New_York)",
    )
    p.add_argument(
        "--symbol",
        default=None,
        help="Single ticker (overrides default symbol list for --at when --symbols not set)",
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated tickers for --at (e.g. AAPL,MSFT,NVDA)",
    )
    p.add_argument(
        "--as-of",
        default=None,
        metavar="YYYY-MM-DD",
        help="UTC calendar day: last bar before that day 00:00 UTC (single symbol from --symbol)",
    )
    p.add_argument(
        "--start",
        default=None,
        help="Range mode: window start (UTC)",
    )
    p.add_argument(
        "--end",
        default=None,
        help="Range mode: window end (UTC)",
    )
    p.add_argument(
        "--step-hours",
        type=int,
        default=24,
        help="Range mode: minimum hours between evaluation points (default: 24)",
    )
    p.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Range mode: cap number of evaluations",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Days of history for the model context (default: 30)",
    )
    p.add_argument(
        "--min-bars",
        type=int,
        default=48,
        help="Minimum hourly bars required in the lookback window",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data and labels only; do not call LLMs",
    )
    p.add_argument(
        "--output",
        type=str,
        default="backtest/results",
        help="Directory for JSON reports",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    modes = [
        bool(args.at),
        bool(args.as_of),
        bool(args.start and args.end),
    ]
    if sum(modes) != 1:
        raise SystemExit(
            "Exactly one mode required:\n"
            "  --at \"DATE TIME\" [--tz ...]   (multi-symbol default: top 100)\n"
            "  --as-of YYYY-MM-DD             (single --symbol)\n"
            "  --start ... --end ...          (range backtest)\n"
        )

    if args.at and (args.start or args.end or args.as_of):
        raise SystemExit("Do not combine --at with --as-of or --start/--end.")

    if bool(args.start) ^ bool(args.end):
        raise SystemExit("Range mode requires both --start and --end.")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
