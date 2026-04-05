from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import AppConfig
from app.engine import _candles_to_context
from app.llm_clients import ClaudeAnalyzer, GeminiAnalyzer, GrokAnalyzer, OpenAIAnalyzer
from app.models import LLMDecision

from backtest.ground_truth import forward_close_to_close
from backtest.historical_data import fetch_hourly_range, slice_lookback_window
from backtest.local_time import find_decision_bar_index, hour_start_utc, parse_at_datetime
from backtest.metrics import summarize_runs

MIN_DECISION_CONFIDENCE = 60


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_as_of_day_utc(s: str) -> datetime:
    """Start of calendar day UTC (YYYY-MM-DD). All bars strictly before this instant are 'history'."""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            base = datetime.strptime(s, fmt)
            return base.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Invalid --as-of date (use YYYY-MM-DD): {s!r}")


def _print_one_symbol_predictions(row: Dict[str, Any]) -> None:
    """Print each model's scores for a single completed symbol (used after each stock in --at loop)."""
    model_order = ("chatgpt", "gemini", "claude", "grok")
    sym = row.get("symbol", "?")
    if row.get("error") or row.get("dry_run") or not row.get("per_model"):
        return

    print(f"      — per-model:")
    pm = row["per_model"]
    for key in model_order:
        c = pm.get(key, {})
        if c.get("low_confidence_skip"):
            win = c.get("winning_confidence", "-")
            print(f"         {key:8}  SKIP  low confidence ({win}% < {MIN_DECISION_CONFIDENCE}%)")
            continue
        if c.get("skipped") or "error" in c:
            err = str(c.get("error", "error"))[:70]
            print(f"         {key:8}  ERR  {err}")
            continue
        lc = c.get("long_confidence", "-")
        sc = c.get("short_confidence", "-")
        side = (c.get("predicted_side") or "?").upper()
        win = c.get("winning_confidence", "-")
        ok = c.get("correct")
        tag = "OK vs market" if ok is True else ("X vs market" if ok is False else "?")
        print(
            f"         {key:8}  L={lc}%  S={sc}%  → {side} (win {win}%)  [{tag}]"
        )


def _print_at_snapshot_terminal(
    meta: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    """Human-readable results to stdout."""
    print()
    print("=" * 92)
    print("BACKTEST — local hour → 30d hourly context → LLMs → vs next-hour return")
    print("=" * 92)
    print(f"  When (local):     {meta['when_local']}")
    print(f"  Timezone:         {meta['timezone']}")
    print(f"  Hour open (UTC):  {meta['hour_open_utc']}")
    print(f"  Lookback:         {meta['lookback_days']} days (same window logic as production)")
    print(f"  Symbols:          {', '.join(meta['symbols'])}")
    print("=" * 92)

    headers = ["Symbol", "Ideal", "Ret%", "chatgpt", "gemini", "claude", "grok"]
    col_w = [8, 6, 8, 9, 9, 9, 9]
    line = "".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(line)
    print("(chatgpt…grok: OK = matched market, X = wrong, SKIP = <60% confidence, ERR = failed)")
    print("-" * 92)

    for row in rows:
        sym = row.get("symbol", "?")[:7]
        if row.get("error"):
            print(f"{sym.ljust(8)} ERROR: {row['error'][:70]}")
            continue
        ideal = (row.get("ideal_side") or "-")[:5]
        ret = row.get("forward_return_pct")
        ret_s = f"{ret:+.3f}%" if isinstance(ret, (int, float)) else "-"
        cells = [sym, ideal, ret_s]
        pm = row.get("per_model", {})
        for key in ("chatgpt", "gemini", "claude", "grok"):
            c = pm.get(key, {})
            if c.get("low_confidence_skip"):
                cells.append("SKIP")
            elif c.get("skipped") or "error" in c:
                cells.append("ERR")
            elif c.get("correct") is True:
                cells.append("OK")
            elif c.get("correct") is False:
                cells.append("X")
            else:
                cells.append("-")
        print(
            "".join(str(x).ljust(w) for x, w in zip(cells, col_w))
        )

    print("-" * 92)
    summary = summarize_runs([r for r in rows if r.get("per_model")])
    print("Summary (correct / decided):")
    for name, s in summary.get("models", {}).items():
        acc = s.get("accuracy")
        acc_s = f"{acc:.0%}" if acc is not None else "n/a"
        print(
            f"  {name:8}  {s['correct']} correct  {s['wrong']} wrong  "
            f"{s['skipped']} skipped  accuracy {acc_s}"
        )
    print("=" * 92)
    print()


async def run_at_time_snapshot(
    config: AppConfig,
    symbols: List[str],
    at_string: str,
    tz_name: str,
    lookback_days: int,
    context_mode: str,
    dry_run: bool,
    output_dir: Path,
    min_window_bars: int = 48,
) -> Dict[str, Any]:
    """
    For each symbol: fetch history, anchor decision bar to the user's local hour (e.g. 2pm NY),
    build 30d lookback window, run all LLMs, compare to next-hour return. Prints table to terminal.
    """
    local_dt = parse_at_datetime(at_string, tz_name)
    hour_utc = hour_start_utc(local_dt)
    when_local = local_dt.strftime("%Y-%m-%d %H:%M") + f" ({tz_name})"

    rows: List[Dict[str, Any]] = []
    analyzers = _build_analyzers(config)

    print(
        f"\n[backtest] --at mode | hour_open_utc={hour_utc.isoformat()} "
        f"| symbols={len(symbols)} | context={context_mode}"
    )

    for sym in symbols:
        fetch_start = hour_utc - timedelta(days=lookback_days + 15)
        fetch_end = hour_utc + timedelta(days=3)
        try:
            all_points = await fetch_hourly_range(
                config.stock_data_api_key, sym, fetch_start, fetch_end
            )
            decision_i = find_decision_bar_index(all_points, hour_utc)
        except Exception as exc:
            rows.append({"symbol": sym, "error": str(exc)})
            print(f"  [{sym}] fetch/index error: {exc}")
            continue

        if decision_i + 1 >= len(all_points):
            rows.append(
                {
                    "symbol": sym,
                    "error": "No next hour bar (choose an older --at or check market hours).",
                }
            )
            print(f"  [{sym}] no forward bar")
            continue

        window = slice_lookback_window(all_points, decision_i, lookback_days)
        gt = forward_close_to_close(all_points, decision_i)

        row: Dict[str, Any] = {
            "symbol": sym,
            "mode": "at_local_time",
            "timezone": tz_name,
            "when_local": when_local,
            "hour_open_utc": hour_utc.isoformat(),
            "decision_bar_time_utc": all_points[decision_i].datetime.isoformat(),
            "window_bars": len(window),
        }

        if len(window) < min_window_bars:
            row["error"] = f"window too small ({len(window)} < {min_window_bars})"
            rows.append(row)
            print(f"  [{sym}] skip: {row['error']}")
            continue

        if gt is None:
            row["error"] = "zero forward return or missing bar"
            rows.append(row)
            print(f"  [{sym}] skip: no ground truth")
            continue

        ret, ideal_side = gt
        row["forward_return_pct"] = round(ret * 100, 6)
        row["ideal_side"] = ideal_side

        if dry_run:
            row["dry_run"] = True
            rows.append(row)
            print(
                f"  [{sym}] dry-run ideal={ideal_side} ret={row['forward_return_pct']}% "
                f"(no LLM)"
            )
            continue

        context = _candles_to_context(sym, window, context_mode)
        per_model: Dict[str, Any] = {}

        async def run_one(label: str, analyzer) -> tuple:
            try:
                d = await asyncio.to_thread(analyzer.analyze, sym, context)
                return label, d, None
            except Exception as exc:
                return label, None, exc

        tasks = [asyncio.create_task(run_one(l, a)) for l, a in analyzers]
        for done in asyncio.as_completed(tasks):
            label, decision, err = await done
            if err is not None:
                per_model[label] = {"error": str(err), "skipped": True}
                continue
            assert decision is not None
            pred: LLMDecision = decision
            below_threshold = pred.confidence < MIN_DECISION_CONFIDENCE
            per_model[label] = {
                "long_confidence": pred.long_confidence,
                "short_confidence": pred.short_confidence,
                "predicted_side": pred.action,
                "winning_confidence": pred.confidence,
                "correct": (pred.action == ideal_side) if not below_threshold else None,
                "skipped": below_threshold,
                "low_confidence_skip": below_threshold,
            }

        row["per_model"] = per_model
        rows.append(row)
        print(
            f"  [{sym}] ideal={ideal_side} ret={row['forward_return_pct']}% "
            f"| models OK"
        )
        _print_one_symbol_predictions(row)

    meta_out = {
        "when_local": when_local,
        "timezone": tz_name,
        "hour_open_utc": hour_utc.isoformat(),
        "lookback_days": lookback_days,
        "symbols": symbols,
        "context_builder": "app.engine._candles_to_context",
        "context_mode": context_mode,
        "min_decision_confidence": MIN_DECISION_CONFIDENCE,
        "dry_run": dry_run,
    }
    summary = summarize_runs([r for r in rows if r.get("per_model")])
    report = {
        "meta": meta_out,
        "summary": summary,
        "rows": rows,
    }

    _print_at_snapshot_terminal({**meta_out, "symbols": symbols}, rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe = hour_utc.strftime("%Y%m%d_%H%M") + "_utc"
    out_json = output_dir / f"backtest_at_{safe}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[backtest] Wrote {out_json}")

    return report


def _build_analyzers(config: AppConfig):
    """Same analyzer construction as app.engine (production)."""
    return [
        ("chatgpt", OpenAIAnalyzer(config.openai_api_key, config.openai_model)),
        ("gemini", GeminiAnalyzer(config.google_api_key, config.gemini_model)),
        ("claude", ClaudeAnalyzer(config.anthropic_api_key, config.claude_model)),
        ("grok", GrokAnalyzer(config.xai_api_key, config.grok_model)),
    ]


async def run_as_of_backtest(
    config: AppConfig,
    symbol: str,
    as_of_day: datetime,
    lookback_days: int,
    context_mode: str,
    dry_run: bool,
    output_dir: Path,
    min_window_bars: int = 48,
) -> Dict[str, Any]:
    """
    Single evaluation: use **the same 30-day lookback logic as production** (slice_lookback_window),
    anchored at the **last hourly bar strictly before** ``as_of_day`` (UTC midnight).

    Context string is built with **app.engine._candles_to_context** (same as production).
    """
    cutoff = as_of_day.astimezone(timezone.utc)
    fetch_start = cutoff - timedelta(days=lookback_days + 15)
    fetch_end = cutoff + timedelta(days=5)

    print(
        f"[backtest] Mode: --as-of {cutoff.date()} UTC | "
        f"decision = last bar before {cutoff.isoformat()}"
    )
    print(f"[backtest] Fetching {symbol} hourly {fetch_start.date()} .. {fetch_end.date()} ...")

    all_points = await fetch_hourly_range(
        config.stock_data_api_key, symbol, fetch_start, fetch_end
    )

    # Last index with bar strictly before cutoff (start of chosen calendar day)
    decision_i: Optional[int] = None
    for j in range(len(all_points)):
        if all_points[j].datetime < cutoff:
            decision_i = j
        else:
            break

    if decision_i is None:
        raise ValueError("No hourly bars before as-of cutoff; widen fetch range or check symbol.")
    if decision_i + 1 >= len(all_points):
        raise ValueError(
            "No bar after decision bar (need next hour for realized return). "
            "Try a slightly older --as-of or widen end fetch."
        )

    window = slice_lookback_window(all_points, decision_i, lookback_days)
    gt = forward_close_to_close(all_points, decision_i)

    row: Dict[str, Any] = {
        "symbol": symbol,
        "mode": "as_of",
        "as_of_cutoff_utc": cutoff.isoformat(),
        "decision_bar_index": decision_i,
        "decision_time_utc": all_points[decision_i].datetime.isoformat(),
        "window_bars": len(window),
    }

    if len(window) < min_window_bars:
        raise ValueError(
            f"Lookback window too small ({len(window)} bars < {min_window_bars}). "
            "Older history may be missing from the API for this symbol."
        )

    if gt is None:
        raise ValueError("Could not compute forward return (zero move or missing bar).")

    ret, ideal_side = gt
    row["forward_return_pct"] = round(ret * 100, 6)
    row["ideal_side"] = ideal_side

    if dry_run:
        row["dry_run"] = True
        rows = [row]
    else:
        # Production context builder
        context = _candles_to_context(symbol, window, context_mode)
        analyzers = _build_analyzers(config)
        per_model: Dict[str, Any] = {}

        async def run_one(label: str, analyzer) -> tuple:
            try:
                d = await asyncio.to_thread(analyzer.analyze, symbol, context)
                return label, d, None
            except Exception as exc:
                return label, None, exc

        tasks = [asyncio.create_task(run_one(l, a)) for l, a in analyzers]
        for done in asyncio.as_completed(tasks):
            label, decision, err = await done
            if err is not None:
                per_model[label] = {"error": str(err), "skipped": True}
                continue
            assert decision is not None
            pred: LLMDecision = decision
            below_threshold = pred.confidence < MIN_DECISION_CONFIDENCE
            per_model[label] = {
                "long_confidence": pred.long_confidence,
                "short_confidence": pred.short_confidence,
                "predicted_side": pred.action,
                "winning_confidence": pred.confidence,
                "correct": (pred.action == ideal_side) if not below_threshold else None,
                "skipped": below_threshold,
                "low_confidence_skip": below_threshold,
            }

        row["per_model"] = per_model
        rows = [row]

        print(
            f"[backtest] Decision bar: {row['decision_time_utc']} | "
            f"ideal={ideal_side} ret={row['forward_return_pct']}%"
        )

    summary = summarize_runs([r for r in rows if "per_model" in r])
    report = {
        "meta": {
            "symbol": symbol,
            "mode": "as_of",
            "as_of_cutoff_utc": cutoff.isoformat(),
            "lookback_days": lookback_days,
            "dry_run": dry_run,
            "total_bars_fetched": len(all_points),
            "context_builder": "app.engine._candles_to_context",
            "context_mode": context_mode,
            "min_decision_confidence": MIN_DECISION_CONFIDENCE,
        },
        "summary": summary,
        "rows": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{symbol.replace('.', '_')}_asof_{cutoff.strftime('%Y%m%d')}"
    out_json = output_dir / f"backtest_{tag}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[backtest] Wrote {out_json}")

    out_md = output_dir / f"backtest_{tag}.md"
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"[backtest] Wrote {out_md}")

    return report


async def run_backtest(
    config: AppConfig,
    symbol: str,
    range_start: datetime,
    range_end: datetime,
    step_hours: int,
    max_evals: Optional[int],
    lookback_days: int,
    context_mode: str,
    dry_run: bool,
    output_dir: Path,
    min_window_bars: int = 48,
) -> Dict[str, Any]:
    """
    For each evaluation time in [range_start, range_end], use only data up to that bar,
    run LLMs, compare chosen side to realized close[t]->close[t+1] direction.
    Uses **app.engine._candles_to_context** for the prompt text (same as production).
    """
    range_start = range_start.astimezone(timezone.utc)
    range_end = range_end.astimezone(timezone.utc)

    fetch_start = range_start - timedelta(days=lookback_days + 5)
    fetch_end = range_end + timedelta(days=3)

    print(f"[backtest] Mode: range | Fetching {symbol} hourly {fetch_start.date()} .. {fetch_end.date()} ...")
    points = await fetch_hourly_range(
        config.stock_data_api_key, symbol, fetch_start, fetch_end
    )
    if len(points) < min_window_bars + 2:
        raise ValueError(f"Not enough bars ({len(points)}). widen date range or check symbol.")

    last_eval: Optional[datetime] = None
    eval_indices: List[int] = []
    step = timedelta(hours=step_hours)

    for i in range(len(points) - 1):
        t = points[i].datetime
        if t < range_start or t > range_end:
            continue
        if last_eval is None or (t - last_eval) >= step:
            eval_indices.append(i)
            last_eval = t

    if max_evals is not None:
        eval_indices = eval_indices[: max_evals]

    print(
        f"[backtest] {len(eval_indices)} evaluation times "
        f"(step={step_hours}h, max={max_evals or 'all'})"
    )

    rows: List[Dict[str, Any]] = []
    analyzers = _build_analyzers(config)

    for idx, i in enumerate(eval_indices):
        t = points[i].datetime
        gt = forward_close_to_close(points, i)
        window = slice_lookback_window(points, i, lookback_days)

        row: Dict[str, Any] = {
            "symbol": symbol,
            "mode": "range",
            "decision_bar_index": i,
            "decision_time_utc": t.isoformat(),
            "window_bars": len(window),
        }

        if len(window) < min_window_bars:
            row["skip_reason"] = f"window too small ({len(window)} < {min_window_bars})"
            rows.append(row)
            continue

        if gt is None:
            row["skip_reason"] = "no forward bar or zero return"
            rows.append(row)
            continue

        ret, ideal_side = gt
        row["forward_return_pct"] = round(ret * 100, 6)
        row["ideal_side"] = ideal_side

        if dry_run:
            row["dry_run"] = True
            rows.append(row)
            continue

        context = _candles_to_context(symbol, window, context_mode)
        per_model: Dict[str, Any] = {}

        async def run_one(label: str, analyzer) -> tuple:
            try:
                d = await asyncio.to_thread(analyzer.analyze, symbol, context)
                return label, d, None
            except Exception as exc:
                return label, None, exc

        tasks = [asyncio.create_task(run_one(l, a)) for l, a in analyzers]
        for done in asyncio.as_completed(tasks):
            label, decision, err = await done
            if err is not None:
                per_model[label] = {
                    "error": str(err),
                    "skipped": True,
                }
                continue
            assert decision is not None
            pred: LLMDecision = decision
            below_threshold = pred.confidence < MIN_DECISION_CONFIDENCE
            per_model[label] = {
                "long_confidence": pred.long_confidence,
                "short_confidence": pred.short_confidence,
                "predicted_side": pred.action,
                "winning_confidence": pred.confidence,
                "correct": (pred.action == ideal_side) if not below_threshold else None,
                "skipped": below_threshold,
                "low_confidence_skip": below_threshold,
            }

        row["per_model"] = per_model
        rows.append(row)

        print(
            f"  [{idx + 1}/{len(eval_indices)}] {t.isoformat()} "
            f"ideal={ideal_side} ret={row['forward_return_pct']}%"
        )

    summary = summarize_runs([r for r in rows if "per_model" in r])
    report = {
        "meta": {
            "symbol": symbol,
            "mode": "range",
            "range_start_utc": range_start.isoformat(),
            "range_end_utc": range_end.isoformat(),
            "step_hours": step_hours,
            "lookback_days": lookback_days,
            "dry_run": dry_run,
            "total_bars_fetched": len(points),
            "context_builder": "app.engine._candles_to_context",
            "context_mode": context_mode,
            "min_decision_confidence": MIN_DECISION_CONFIDENCE,
        },
        "summary": summary,
        "rows": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"backtest_{symbol.replace('.', '_')}_range.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[backtest] Wrote {out_json}")

    out_md = output_dir / f"backtest_{symbol.replace('.', '_')}_range.md"
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"[backtest] Wrote {out_md}")

    return report


def _to_markdown(report: Dict[str, Any]) -> str:
    meta = report["meta"]
    lines = [
        "# Backtest report",
        "",
        f"Symbol: **{meta['symbol']}**",
        f"Context builder: `{meta.get('context_builder', '')}`",
    ]
    if meta.get("mode") == "as_of":
        lines.append(f"**As-of cutoff (UTC)**: {meta.get('as_of_cutoff_utc')} — uses **{meta.get('lookback_days')}** days of hourly bars ending at the last bar **before** this instant.")
    else:
        lines.append(
            f"Range (UTC): {meta.get('range_start_utc')} → {meta.get('range_end_utc')} | "
            f"step {meta.get('step_hours')}h | lookback {meta.get('lookback_days')}d"
        )
    lines.append(f"Context mode: `{meta.get('context_mode', 'hybrid')}`")
    lines.append(f"Min decision confidence: `{meta.get('min_decision_confidence', MIN_DECISION_CONFIDENCE)}`")
    lines.extend(
        [
            "",
            "## Accuracy (ideal_side vs predicted_side)",
        ]
    )
    for name, s in report.get("summary", {}).get("models", {}).items():
        acc = s.get("accuracy")
        acc_s = f"{acc:.1%}" if acc is not None else "n/a"
        lines.append(
            f"- **{name}**: correct={s['correct']} wrong={s['wrong']} skipped={s['skipped']} accuracy={acc_s}"
        )
    lines.append("")
    lines.append("## Rows")
    lines.append("(See JSON for full per-model details.)")
    return "\n".join(lines)


async def async_main(args: Any) -> None:
    config = AppConfig.from_env()
    out = Path(args.output)

    if getattr(args, "at", None):
        if getattr(args, "symbols", None):
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        elif getattr(args, "symbol", None):
            symbols = [args.symbol.upper()]
        else:
            symbols = list(config.symbols)
        await run_at_time_snapshot(
            config=config,
            symbols=symbols,
            at_string=args.at,
            tz_name=getattr(args, "tz", "America/New_York"),
            lookback_days=args.lookback_days,
            context_mode=getattr(args, "context_mode", config.context_mode),
            dry_run=args.dry_run,
            output_dir=out,
            min_window_bars=args.min_bars,
        )
        return

    symbol = args.symbol.upper() if getattr(args, "symbol", None) else config.symbols[0]
    if getattr(args, "as_of", None):
        as_of_day = _parse_as_of_day_utc(args.as_of)
        await run_as_of_backtest(
            config=config,
            symbol=symbol,
            as_of_day=as_of_day,
            lookback_days=args.lookback_days,
            context_mode=getattr(args, "context_mode", config.context_mode),
            dry_run=args.dry_run,
            output_dir=out,
            min_window_bars=args.min_bars,
        )
        return

    await run_backtest(
        config=config,
        symbol=symbol,
        range_start=_parse_utc(args.start),
        range_end=_parse_utc(args.end),
        step_hours=args.step_hours,
        max_evals=args.max_evals,
        lookback_days=args.lookback_days,
        context_mode=getattr(args, "context_mode", config.context_mode),
        dry_run=args.dry_run,
        output_dir=out,
        min_window_bars=args.min_bars,
    )
