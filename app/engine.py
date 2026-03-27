from __future__ import annotations

import asyncio
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from app.alpaca_trading import alpaca_consensus_round_trip, log_alpaca_account_health
from app.config import AppConfig
from app.data_provider import TwelveDataClient
from app.features import build_feature_context, recent_bars_snapshot
from app.llm_clients import ClaudeAnalyzer, GeminiAnalyzer, GrokAnalyzer, OpenAIAnalyzer
from app.telegram_notifier import TelegramConfig, send_telegram_message
from app.models import ConsensusResult, LLMDecision

CONSENSUS_MIN_MODELS = 3
CONSENSUS_MIN_CONFIDENCE = 60


def _candles_to_context(symbol: str, points: list, mode: str = "hybrid") -> str:
    mode = (mode or "hybrid").strip().lower()
    if mode not in {"raw", "hybrid", "features"}:
        raise ValueError(f"Unsupported context mode: {mode}")
    if mode == "raw":
        return _candles_to_raw_context(symbol, points)

    features = build_feature_context(symbol, points)
    payload = {
        "symbol": symbol,
        "context_mode": mode,
        "feature_snapshot": features,
    }
    if mode == "hybrid":
        payload["recent_hourly_bars"] = recent_bars_snapshot(points, count=16)
    return json.dumps(payload, indent=2)


def _candles_to_raw_context(symbol: str, points: list) -> str:
    lines = ["datetime,open,high,low,close,volume"]
    for p in points:
        lines.append(
            f"{p.datetime.isoformat()},{p.open:.4f},{p.high:.4f},{p.low:.4f},{p.close:.4f},{p.volume:.0f}"
        )
    return f"Stock: {symbol}\n" + "\n".join(lines)


def _write_candles_csv(symbol: str, points: list, output_dir: Path) -> Path:
    data_dir = output_dir / "raw_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{symbol.replace('.', '_')}_1h_30d.csv"
    lines = ["datetime,open,high,low,close,volume"]
    for p in points:
        lines.append(
            f"{p.datetime.isoformat()},{p.open:.4f},{p.high:.4f},{p.low:.4f},{p.close:.4f},{p.volume:.0f}"
        )
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path


def _print_data_summary(symbol: str, points: list, csv_path: Path) -> None:
    first = points[0]
    last = points[-1]
    print(f"\n=== [{symbol}] STEP 1: DATA RECEIVED ===")
    print(f"Candles: {len(points)} hourly points (30 days)")
    print(
        f"Range: {first.datetime.isoformat()} -> {last.datetime.isoformat()} | "
        f"Close: {first.close:.2f} -> {last.close:.2f}"
    )
    print(f"Saved raw candles: {csv_path}")


def _print_model_result(decision: LLMDecision) -> None:
    print(
        f"[{decision.symbol}] {decision.model.upper()} -> "
        f"L{decision.long_confidence}% S{decision.short_confidence}% "
        f"=> {decision.action.upper()} (win {decision.confidence}%) | {decision.rationale}"
    )


def _print_model_error(symbol: str, exc: Exception) -> None:
    print(f"[{symbol}] MODEL ERROR -> {exc}")


def _print_consensus(symbol: str, consensus: dict) -> None:
    action = consensus["aligned_action"] or "none"
    print(
        f"[{symbol}] STEP 3: CONSENSUS -> action={action}, "
        f"min_conf={consensus['minimum_confidence']}%, passes={consensus['passes_threshold']}"
    )


def _consensus(symbol: str, decisions: Iterable[LLMDecision]) -> ConsensusResult:
    decisions = list(decisions)
    qualified = {"long": [], "short": []}
    for d in decisions:
        if d.confidence >= CONSENSUS_MIN_CONFIDENCE and d.action in qualified:
            qualified[d.action].append(d)

    chosen_side = max(qualified.keys(), key=lambda side: len(qualified[side]))
    supporters = qualified[chosen_side]
    passes = len(supporters) >= CONSENSUS_MIN_MODELS
    min_conf = min((d.confidence for d in supporters), default=0)

    return ConsensusResult(
        symbol=symbol,
        aligned_action=chosen_side if passes else None,
        minimum_confidence=min_conf,
        passes_threshold=passes,
        model_count=len(decisions),
    )


def _emit(on_event: Optional[Callable[[dict], None]], payload: dict) -> None:
    if on_event:
        on_event(payload)


async def run_analysis(
    config: AppConfig,
    out_dir: str = "outputs",
    on_event: Optional[Callable[[dict], None]] = None,
) -> Path:
    provider = TwelveDataClient(api_key=config.stock_data_api_key)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzers = [
        ("chatgpt", OpenAIAnalyzer(config.openai_api_key, config.openai_model)),
        ("gemini", GeminiAnalyzer(config.google_api_key, config.gemini_model)),
        ("claude", ClaudeAnalyzer(config.anthropic_api_key, config.claude_model)),
        ("grok", GrokAnalyzer(config.xai_api_key, config.grok_model)),
    ]

    telegram_cfg = TelegramConfig(
        enabled=config.telegram_enabled,
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
    )

    full_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "context_mode": config.context_mode,
        "consensus_rule": (
            f">= {CONSENSUS_MIN_MODELS} of 4 models agree "
            f"with confidence >= {CONSENSUS_MIN_CONFIDENCE}"
        ),
        "symbols": {},
        "consensus_signals": [],
    }

    _emit(
        on_event,
        {
            "type": "run_started",
            "symbols": list(config.symbols),
            "generated_at_utc": full_report["generated_at_utc"],
        },
    )

    alpaca_tasks: List[Tuple[str, asyncio.Task]] = []

    for symbol in config.symbols:
        print(f"\n==================== SYMBOL: {symbol} ====================")
        _emit(on_event, {"type": "symbol_phase", "phase": "start", "symbol": symbol})

        points = await provider.fetch_hourly_30d(symbol)
        context = _candles_to_context(symbol, points, config.context_mode)
        csv_path = _write_candles_csv(symbol, points, output_dir)
        _print_data_summary(symbol, points, csv_path)

        first, last = points[0], points[-1]
        _emit(
            on_event,
            {
                "type": "data_loaded",
                "symbol": symbol,
                "candles_count": len(points),
                "range_start": first.datetime.isoformat(),
                "range_end": last.datetime.isoformat(),
                "first_close": first.close,
                "last_close": last.close,
                "csv_path": str(csv_path),
                "context_mode": config.context_mode,
            },
        )

        print(f"[{symbol}] STEP 2: RUNNING MODEL ANALYSIS")
        _emit(on_event, {"type": "models_started", "symbol": symbol})

        decisions: List[dict] = []
        parsed: List[LLMDecision] = []

        async def _run_one(label: str, analyzer) -> tuple:
            try:
                result = await asyncio.to_thread(analyzer.analyze, symbol, context)
                return label, result, None
            except Exception as exc:
                return label, None, exc

        model_tasks = [
            asyncio.create_task(_run_one(label, a)) for label, a in analyzers
        ]

        for done in asyncio.as_completed(model_tasks):
            label, result, err = await done
            if err is not None:
                _print_model_error(symbol, err)
                row = {
                    "model": "unknown",
                    "error": str(err),
                    "long_confidence": 0,
                    "short_confidence": 0,
                }
                decisions.append(row)
                _emit(
                    on_event,
                    {
                        "type": "model_result",
                        "symbol": symbol,
                        "model_label": label,
                        "ok": False,
                        "error": str(err),
                    },
                )
                continue
            assert result is not None
            parsed.append(result)
            _print_model_result(result)
            decisions.append(result.model_dump())
            _emit(
                on_event,
                {
                    "type": "model_result",
                    "symbol": symbol,
                    "model_label": label,
                    "ok": True,
                    "decision": result.model_dump(),
                },
            )

        if parsed:
            consensus = _consensus(symbol, parsed).model_dump()
        else:
            consensus = ConsensusResult(
                symbol=symbol,
                aligned_action=None,
                minimum_confidence=0,
                passes_threshold=False,
                model_count=0,
            ).model_dump()

        if consensus["passes_threshold"]:
            full_report["consensus_signals"].append(consensus)

        telegram_sent = False
        telegram_error: Optional[str] = None
        if consensus.get("passes_threshold"):
            aligned = consensus.get("aligned_action")
            min_conf = consensus.get("minimum_confidence")
            telegram_text = (
                f"CONSENSUS: {symbol} -> {str(aligned).upper()} "
                f"(min confidence {min_conf}%) | context={config.context_mode}"
            )
            telegram_sent, telegram_error = await send_telegram_message(
                telegram_cfg,
                telegram_text,
            )

        _print_consensus(symbol, consensus)
        _emit(on_event, {"type": "consensus", "symbol": symbol, "consensus": consensus})

        sym_entry = {
            "candles_count": len(points),
            "data_csv_path": str(csv_path),
            "context_mode": config.context_mode,
            "model_decisions": decisions,
            "consensus": consensus,
            "telegram_notified": telegram_sent,
            "telegram_error": telegram_error,
        }

        if consensus.get("passes_threshold") and config.alpaca_enabled:
            aligned = consensus.get("aligned_action")
            assert aligned in ("long", "short")
            print(
                f"[{symbol}] ALPACA: scheduling {aligned.upper()} market order, "
                f"close after {config.alpaca_hold_seconds}s (paper={config.alpaca_paper})"
            )
            task = asyncio.create_task(
                alpaca_consensus_round_trip(config, symbol, aligned)
            )
            alpaca_tasks.append((symbol, task))
            sym_entry["alpaca"] = {"scheduled": True, "side": aligned}

        full_report["symbols"][symbol] = sym_entry

    if alpaca_tasks:
        log_alpaca_account_health(config)
        print(
            f"\n[Alpaca] Waiting for {len(alpaca_tasks)} round-trip(s) "
            f"({config.alpaca_hold_seconds}s hold each, parallel)..."
        )
        symbols_ordered = [s for s, _ in alpaca_tasks]
        gather_tasks = [t for _, t in alpaca_tasks]
        results = await asyncio.gather(*gather_tasks, return_exceptions=True)
        for sym, res in zip(symbols_ordered, results):
            if isinstance(res, Exception):
                full_report["symbols"][sym]["alpaca"] = {
                    "ok": False,
                    "error": str(res),
                }
            else:
                full_report["symbols"][sym]["alpaca"] = res

    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    html_path = output_dir / "report.html"

    json_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(full_report), encoding="utf-8")
    html_path.write_text(_to_html(full_report), encoding="utf-8")
    print(f"\nSaved final JSON report: {json_path}")
    print(f"Saved final Markdown report: {md_path}")
    print(f"Saved final HTML report: {html_path}")

    _emit(
        on_event,
        {
            "type": "finished",
            "json_path": str(json_path.resolve()),
            "md_path": str(md_path.resolve()),
            "html_path": str(html_path.resolve()),
            "report": full_report,
        },
    )
    return json_path


def _to_markdown(report: dict) -> str:
    lines = [
        "# Trading Consensus Report",
        "",
        f"- Generated at (UTC): {report['generated_at_utc']}",
        f"- Context mode: {report.get('context_mode', 'hybrid')}",
        "",
        f"## Consensus Signals ({CONSENSUS_MIN_MODELS} of 4 agree, confidence >= {CONSENSUS_MIN_CONFIDENCE})",
    ]
    if report["consensus_signals"]:
        for s in report["consensus_signals"]:
            lines.append(
                f"- {s['symbol']}: **{s['aligned_action'].upper()}** (min confidence: {s['minimum_confidence']}%)"
            )
    else:
        lines.append("- No consensus signals found.")

    lines.append("")
    lines.append("## Per Symbol Decisions")
    for symbol, data in report["symbols"].items():
        lines.append(f"### {symbol}")
        lines.append(f"- Hourly candles analyzed: {data['candles_count']}")
        c = data["consensus"]
        lines.append(
            f"- Consensus: {c['aligned_action'] or 'none'} | min confidence: {c['minimum_confidence']} | passes: {c['passes_threshold']}"
        )
        if data.get("alpaca") is not None:
            lines.append(f"- Alpaca: {json.dumps(data['alpaca'])}")
        for d in data["model_decisions"]:
            if "error" in d:
                lines.append(f"  - model error: {d['error']}")
            else:
                lines.append(
                    f"  - {d['model']}: long {d['long_confidence']}% | short {d['short_confidence']}% "
                    f"=> {d['action']} (win {d['confidence']}%) — {d['rationale']}"
                )
        lines.append("")
    return "\n".join(lines)


def _to_html(report: dict) -> str:
    consensus_rows = ""
    if report["consensus_signals"]:
        for s in report["consensus_signals"]:
            consensus_rows += (
                "<tr>"
                f"<td>{html.escape(s['symbol'])}</td>"
                f"<td>{html.escape((s['aligned_action'] or '').upper())}</td>"
                f"<td>{s['minimum_confidence']}%</td>"
                "</tr>"
            )
    else:
        consensus_rows = '<tr><td colspan="3">No aligned high-confidence signals found.</td></tr>'

    symbol_sections = []
    for symbol, data in report["symbols"].items():
        decision_rows = ""
        for d in data["model_decisions"]:
            if "error" in d:
                decision_rows += (
                    "<tr>"
                    f"<td>{html.escape(str(d.get('model', 'unknown')))}</td>"
                    "<td colspan=\"5\">"
                    f"<span style=\"color:#b91c1c\">{html.escape(d['error'])}</span>"
                    "</td>"
                    "</tr>"
                )
            else:
                decision_rows += (
                    "<tr>"
                    f"<td>{html.escape(d['model'])}</td>"
                    f"<td>{d['long_confidence']}%</td>"
                    f"<td>{d['short_confidence']}%</td>"
                    f"<td>{html.escape(d['action'])}</td>"
                    f"<td>{d['confidence']}%</td>"
                    f"<td>{html.escape(d['rationale'])}</td>"
                    "</tr>"
                )

        c = data["consensus"]
        has_consensus = bool(c.get("passes_threshold"))
        badge = (
            '<span class="badge consensus">CONSENSUS SIGNAL</span>'
            if has_consensus
            else '<span class="badge muted-badge">no consensus</span>'
        )
        card_class = "card consensus-card" if has_consensus else "card"
        section = f"""
        <section class="{card_class}">
          <h3>{html.escape(symbol)} {badge}</h3>
          <p><strong>Hourly candles analyzed:</strong> {data['candles_count']}</p>
          <p><strong>Raw data:</strong> {html.escape(data.get('data_csv_path', ''))}</p>
          <p><strong>Consensus:</strong> {html.escape(str(c['aligned_action'] or 'none'))}
             | min confidence: {c['minimum_confidence']}%
             | passes: {c['passes_threshold']}</p>
          <p><strong>Alpaca:</strong> {html.escape(json.dumps(data.get("alpaca"))) if data.get("alpaca") is not None else "—"}</p>
            <table>
            <thead>
              <tr><th>Model</th><th>Long %</th><th>Short %</th><th>Chosen</th><th>Win %</th><th>Rationale</th></tr>
            </thead>
            <tbody>{decision_rows}</tbody>
          </table>
        </section>
        """
        symbol_sections.append(section)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trading Consensus Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    .muted {{ color: #475569; margin-bottom: 16px; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    .consensus-card {{ border-color: #16a34a; box-shadow: 0 0 0 2px rgba(22,163,74,0.12); }}
    .badge {{ display: inline-block; font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 999px; vertical-align: middle; margin-left: 8px; }}
    .badge.consensus {{ background: #dcfce7; color: #166534; border: 1px solid #86efac; }}
    .muted-badge {{ background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; }}
  </style>
</head>
<body>
  <h1>Trading Consensus Report</h1>
  <div class="muted">Generated at (UTC): {html.escape(report["generated_at_utc"])}</div>
  <div class="muted">Context mode: {html.escape(str(report.get("context_mode", "hybrid")))}</div>

  <section class="card">
    <h2>Consensus Signals ({CONSENSUS_MIN_MODELS} of 4 agree, confidence &gt;= {CONSENSUS_MIN_CONFIDENCE})</h2>
    <table>
      <thead><tr><th>Symbol</th><th>Action</th><th>Minimum Confidence</th></tr></thead>
      <tbody>{consensus_rows}</tbody>
    </table>
  </section>

  <h2>Per Symbol Decisions</h2>
  {''.join(symbol_sections)}
</body>
</html>
"""
