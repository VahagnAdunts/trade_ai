# Backtest (imports production code; does not edit `app/`)

## `--at` — local time (e.g. **2pm New York**), all top stocks, terminal output

You give a **wall-clock time** in a **timezone** (default **America/New_York**). The run:

1. Converts that to the **UTC open** of that hour (e.g. 25 Feb 2025 2:00 PM ET).
2. For **each symbol** (default: **same top 100 as `DEFAULT_TOP100_SP500` in `app.config`**):
   - Fetches hourly history from TwelveData.
   - Picks the **decision bar**: last 1h bar whose open time is **≤** that UTC instant.
   - Builds **30 days** of hourly bars up to that bar (`slice_lookback_window`, same as production).
   - Builds the prompt with **`app.engine._candles_to_context`** (`raw`, `hybrid`, or `features` mode).
   - Calls **all four LLMs** (unless `--dry-run`).
   - Compares each model’s chosen side to the **realized next-hour** return:  
     `close(next)/close(decision) - 1` → ideal **long** / **short**.

**Results are printed to the terminal** (table + summary). A JSON file is also written under `backtest/results/`.

Confidence rule for summary metrics:
- If a model's winning confidence is below **60%**, that prediction is treated as **SKIP**
  (not counted as correct/wrong in accuracy).

### Examples

```bash
# European-style date, 2pm New York, all default symbols (top 100)
python -m backtest --at "25.02.2025 14:00" --tz America/New_York

# ISO-style, subset of symbols
python -m backtest --at "2025-02-25 14:00" --symbols AAPL,MSFT,NVDA

# Only one symbol
python -m backtest --at "2025-02-25 14:00" --symbol AAPL

# Feature-only context (indicator snapshot, no raw candle table)
python -m backtest --at "2025-02-25 14:00" --symbol AAPL --context-mode features
```

### Notes

- US stocks: hourly bars follow **exchange/RTH** as returned by TwelveData; **2pm ET** must fall in a valid trading hour or the nearest bar is used.
- **BRK.B** and other symbols must be valid for your data provider.

## `--as-of` — UTC calendar day (single symbol)

```bash
python -m backtest --as-of 2024-06-15 --symbol AAPL
```

## Range mode

```bash
python -m backtest --start 2024-01-02 --end 2024-01-20 --symbol MSFT --step-hours 72 --max-evals 5
```

## Requirements

Same `.env` as the main app. You can set `CONTEXT_MODE=raw|hybrid|features` globally,
or override per run via `--context-mode`.
