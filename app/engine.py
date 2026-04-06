from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from app.alpaca_pending import reconcile_pending_closes_on_startup
from app.alpaca_trading import alpaca_consensus_round_trip, log_alpaca_account_health
from app.config import AppConfig
from app.data_provider import TwelveDataClient
from app.features import build_feature_context, recent_bars_snapshot
from app.llm_clients import ClaudeAnalyzer, GeminiAnalyzer, GrokAnalyzer, OpenAIAnalyzer
from app.regime import build_market_regime_payload, load_regime_cache
from app.consensus_format import format_consensus_telegram_message
from app.telegram_notifier import TelegramConfig, send_telegram_message
from app.models import ConsensusResult, LLMDecision


def _candles_to_context(
    symbol: str,
    points: list,
    mode: str = "hybrid",
    *,
    crypto: bool = False,
    market_regime: Optional[dict] = None,
) -> str:
    mode = (mode or "hybrid").strip().lower()
    if mode not in {"raw", "hybrid", "features"}:
        raise ValueError(f"Unsupported context mode: {mode}")
    if mode == "raw":
        body = _candles_to_raw_context(symbol, points, crypto=crypto)
        if market_regime:
            body += "\n\nmarket_regime (JSON, Twelve Data benchmarks):\n"
            body += json.dumps(market_regime, indent=2)
        return body

    features = build_feature_context(symbol, points)
    for w in features.get("data_quality", {}).get("warnings", []):
        print(f"[{symbol}] Data quality: {w}", flush=True)
    payload = {
        "symbol": symbol,
        "asset_class": "crypto" if crypto else "equity",
        "context_mode": mode,
        "feature_snapshot": features,
    }
    if market_regime is not None:
        payload["market_regime"] = market_regime
    if mode == "hybrid":
        payload["recent_hourly_bars"] = recent_bars_snapshot(points, count=16)
    return json.dumps(payload, indent=2)


def _candles_to_raw_context(symbol: str, points: list, *, crypto: bool = False) -> str:
    lines = ["datetime,open,high,low,close,volume"]
    for p in points:
        lines.append(
            f"{p.datetime.isoformat()},{p.open:.4f},{p.high:.4f},{p.low:.4f},{p.close:.4f},{p.volume:.0f}"
        )
    label = "Crypto pair" if crypto else "Stock"
    return f"{label}: {symbol}\n" + "\n".join(lines)


def _print_data_summary(symbol: str, points: list) -> None:
    first = points[0]
    last = points[-1]
    print(f"\n=== [{symbol}] STEP 1: DATA RECEIVED ===")
    print(f"Candles: {len(points)} hourly points (30 days)")
    print(
        f"Range: {first.datetime.isoformat()} -> {last.datetime.isoformat()} | "
        f"Close: {first.close:.2f} -> {last.close:.2f}"
    )


def _print_model_result(decision: LLMDecision) -> None:
    print(
        f"[{decision.symbol}] {decision.model.upper()} -> "
        f"L{decision.long_confidence}% S{decision.short_confidence}% "
        f"=> {decision.action.upper()} (win {decision.confidence}%)"
    )


def _print_model_error(symbol: str, exc: Exception) -> None:
    print(f"[{symbol}] MODEL ERROR -> {exc}")


def _print_consensus(symbol: str, consensus: dict) -> None:
    action = consensus["aligned_action"] or "none"
    print(
        f"[{symbol}] STEP 3: CONSENSUS -> action={action}, "
        f"min_conf={consensus['minimum_confidence']}%, passes={consensus['passes_threshold']}"
    )


# Claude tends to report lower scores than other models; use (base − this) as its bar.
CLAUDE_CONSENSUS_CONFIDENCE_OFFSET_PCT = 18


def _min_confidence_threshold_for_decision(decision: LLMDecision, base_min_pct: int) -> int:
    if (decision.model or "").strip().lower() == "claude":
        return max(0, base_min_pct - CLAUDE_CONSENSUS_CONFIDENCE_OFFSET_PCT)
    return base_min_pct


def _consensus(
    symbol: str,
    decisions: Iterable[LLMDecision],
    config: AppConfig,
    *,
    min_confidence_pct: Optional[int] = None,
) -> ConsensusResult:
    decisions = list(decisions)
    min_conf = (
        min_confidence_pct
        if min_confidence_pct is not None
        else config.consensus_min_confidence_pct
    )
    min_models = config.consensus_min_models
    qualified = {"long": [], "short": []}
    for d in decisions:
        need = _min_confidence_threshold_for_decision(d, min_conf)
        if d.confidence >= need and d.action in qualified:
            qualified[d.action].append(d)

    chosen_side = max(
        ("long", "short"),
        key=lambda s: (len(qualified[s]), 1 if s == "long" else 0),
    )
    supporters = qualified[chosen_side]
    passes = len(supporters) >= min_models
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
    *,
    crypto: bool = False,
) -> Path:
    if config.alpaca_api_key_id and config.alpaca_api_secret_key:
        await reconcile_pending_closes_on_startup(config)
    provider = TwelveDataClient(api_key=config.stock_data_api_key)
    if out_dir == "outputs" and crypto:
        out_dir = "outputs_crypto"
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols_list = list(config.crypto_symbols if crypto else config.symbols)

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
        "asset_class": "crypto" if crypto else "equity",
        "context_mode": config.context_mode,
        "consensus_rule": (
            (
                f">= {config.consensus_min_models} of 4 models agree on long vs short "
                f"with winning confidence >= {config.consensus_min_confidence_crypto_pct} "
                f"(Claude: >= {max(0, config.consensus_min_confidence_crypto_pct - CLAUDE_CONSENSUS_CONFIDENCE_OFFSET_PCT)}; crypto)"
            )
            if crypto
            else (
                f">= {config.consensus_min_models} of 4 models agree on long vs short "
                f"with winning confidence >= {config.consensus_min_confidence_pct} "
                f"(Claude: >= {max(0, config.consensus_min_confidence_pct - CLAUDE_CONSENSUS_CONFIDENCE_OFFSET_PCT)})"
            )
        ),
        "symbols": {},
        "consensus_signals": [],
    }

    _emit(
        on_event,
        {
            "type": "run_started",
            "symbols": symbols_list,
            "crypto": crypto,
            "generated_at_utc": full_report["generated_at_utc"],
        },
    )

    alpaca_tasks: List[Tuple[str, asyncio.Task]] = []

    regime_cache = await load_regime_cache(provider, crypto=crypto)
    re_err = regime_cache.get("fetch_errors") or {}
    if re_err:
        print(
            f"[Regime] Some benchmarks failed to load (see report market_regime.fetch_errors): "
            f"{', '.join(re_err.keys())}",
            flush=True,
        )
    else:
        print("[Regime] Benchmark series loaded for this run.", flush=True)

    for symbol in symbols_list:
        print(f"\n==================== SYMBOL: {symbol} ====================")
        _emit(on_event, {"type": "symbol_phase", "phase": "start", "symbol": symbol})

        points = await provider.fetch_hourly_30d(symbol)
        market_regime = build_market_regime_payload(
            symbol, points, regime_cache, crypto=crypto
        )
        context = _candles_to_context(
            symbol,
            points,
            config.context_mode,
            crypto=crypto,
            market_regime=market_regime,
        )
        _print_data_summary(symbol, points)

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
                "context_mode": config.context_mode,
            },
        )

        print(f"[{symbol}] STEP 2: RUNNING MODEL ANALYSIS")
        _emit(on_event, {"type": "models_started", "symbol": symbol})

        decisions: List[dict] = []
        parsed: List[LLMDecision] = []
        per_model: dict[str, dict] = {}

        async def _run_one(label: str, analyzer) -> tuple:
            try:
                result = await asyncio.to_thread(
                    analyzer.analyze, symbol, context, crypto=crypto
                )
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
                # Do not fabricate confidence scores — only record which model failed and why.
                row = {
                    "model_label": label,
                    "error": str(err),
                }
                per_model[label] = {"error": str(err)}
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
            d = result.model_dump()
            d["model_label"] = label
            per_model[label] = d
            decisions.append(d)
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
            min_pct = (
                config.consensus_min_confidence_crypto_pct
                if crypto
                else config.consensus_min_confidence_pct
            )
            consensus = _consensus(
                symbol, parsed, config, min_confidence_pct=min_pct
            ).model_dump()
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
            telegram_text = (
                format_consensus_telegram_message(
                    symbol,
                    consensus,
                    per_model,
                    crypto=crypto,
                    manual_execution_note=crypto,
                )
                + f"\ncontext={config.context_mode}"
            )
            telegram_sent, telegram_error = await send_telegram_message(
                telegram_cfg,
                telegram_text,
            )

        _print_consensus(symbol, consensus)
        _emit(on_event, {"type": "consensus", "symbol": symbol, "consensus": consensus})

        sym_entry = {
            "candles_count": len(points),
            "context_mode": config.context_mode,
            "market_regime": market_regime,
            "model_decisions": decisions,
            "consensus": consensus,
            "telegram_notified": telegram_sent,
            "telegram_error": telegram_error,
        }

        if consensus.get("passes_threshold") and config.alpaca_enabled and not crypto:
            aligned = consensus.get("aligned_action")
            assert aligned in ("long", "short")
            print(
                f"[{symbol}] ALPACA: scheduling {aligned.upper()} market order, "
                f"close after {config.alpaca_hold_seconds}s (paper={config.alpaca_paper})"
            )
            task = asyncio.create_task(
                alpaca_consensus_round_trip(
                    config,
                    symbol,
                    aligned,
                    crypto=False,
                    telegram_cfg=telegram_cfg,
                )
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
    json_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nSaved research report: {json_path}")

    _emit(
        on_event,
        {
            "type": "finished",
            "json_path": str(json_path.resolve()),
            "report": full_report,
        },
    )
    return json_path
