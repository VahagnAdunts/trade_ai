# Multi-LLM S&P 500 Trading Consensus Tool

This project analyzes 100 major S&P 500 stocks using:
- 30 days of hourly OHLCV price data
- 4 separate LLMs (ChatGPT, Gemini, Claude, Grok)
- each model outputs **`long_confidence`** and **`short_confidence`** (0–100 each); the app **chooses long if long ≥ short, else short** (ties → long), using the **winning** score as that model’s confidence
- consensus filter: signal passes only when **at least 3 of 4 models agree on side** and each agreeing model has **winning confidence 60%+**

## What it does

1. Pulls hourly candles for each stock from TwelveData API.
2. Sends each stock's data to each model independently.
3. Validates model JSON output format.
4. Finds consensus trade opportunities.
5. Writes reports to:
   - `outputs/report.json`
   - `outputs/report.md`
   - `outputs/report.html`

### Live progress in the browser (recommended)

Starts a small local server, opens the dashboard, and **streams each step** (data loaded → each model as it finishes → consensus) over WebSocket:

```bash
python main.py --serve
```

Then click **Start analysis**. The latest static report is also available at `http://127.0.0.1:8765/outputs/report.html` after the run.

### Historical backtest (optional, separate code)

The **`backtest/`** package uses **`app.engine._candles_to_context`** and the same LLM classes as production. Pass **`--as-of YYYY-MM-DD`** (UTC) to use **30 days of hourly data before that day**, then call all models and compare to the realized next-hour move. See **`backtest/README.md`**.

```bash
# Local time (e.g. 25 Feb 2025 2pm New York) — top 100 symbols, table printed to terminal
python -m backtest --at "25.02.2025 14:00" --tz America/New_York

python -m backtest --as-of 2024-06-15 --symbol AAPL
```

## Default stock universe (top 100 large-cap)

Defined in `app.config.DEFAULT_TOP100_SP500`.

You can override with `SYMBOLS` env var.

## Setup

1. Create and activate virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

```bash
cp .env.example .env
```

Fill all required API keys in `.env`.

## Run

```bash
python main.py
```

Optional: `python main.py --serve --port 8765` for the live dashboard.

Telegram-triggered runs (production style):

```bash
python main.py --telegram-runner
```

Then send `/run` to your configured bot chat. Supported commands: `/run`, `/status`, `/help`.

### Render (Background Worker)

Use **Background Worker** with start command `python main.py --telegram-runner`.  
The repo includes **`.python-version`** (`3.12.8`) so installs use wheels for `pydantic-core` (avoid Python 3.14 + Rust build failures).  
Alternatively set env **`PYTHON_VERSION=3.12.8`** in the Render dashboard.

## Environment variables

Required:
- `STOCK_DATA_API_KEY` (TwelveData)
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `ANTHROPIC_API_KEY`
- `XAI_API_KEY`

Optional model overrides:
- `OPENAI_MODEL`
- `GEMINI_MODEL`
- `CLAUDE_MODEL`
- `GROK_MODEL`
- `SYMBOLS` (comma-separated list, e.g. `AAPL,MSFT,NVDA`)
- `CONTEXT_MODE` (`raw|hybrid|features`, default `hybrid`)

Optional notifications:
- `TELEGRAM_ENABLED` (`true|false`)
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Notes

- This is an analysis assistant, not financial advice.
- LLM output is non-deterministic and can be wrong.
- Add risk management before real trading execution.
