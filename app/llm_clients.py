from __future__ import annotations

import json
import re
from typing import Callable, List, Protocol

import anthropic
import google.generativeai as genai
from openai import OpenAI

from app.models import LLMDecision


SYSTEM_PROMPT_CRYPTO = """
You are a trading signal model for a very short intraday strategy on CRYPTOCURRENCY (USD spot pairs).

Focus strictly on HOURLY positions only: interpret the chart and momentum at the 1-hour timeframe.
Do not optimize for daily, weekly, or longer horizons—every judgment must be for an hourly position.

You receive one crypto pair context in one of these forms:
- RAW: full hourly OHLCV candles.
- FEATURES: pre-computed technical indicators/statistics.
- HYBRID: features + small recent-candles snapshot.
When "market_regime" is present in the JSON, it is real benchmark data (BTC/ETH vs this pair from the data provider), not fabricated.
Use features as primary signal. If recent bars are present, use them as secondary confirmation.

Trading intent (critical):
- We only evaluate whether opening a LONG for this hour is worthwhile (spot execution does not use shorting).
- We open a long for exactly one hour, then close it after that hour (no multi-hour holds).

You must output ONE score (0–100):
- long_confidence = how worthwhile / favorable it is to open a 1-hour LONG now (profit if price rises this hour).
  Low values mean “not worth entering”; high values mean a more attractive long setup for this hour.

Return ONLY valid JSON with:
{
  "long_confidence": <integer 0-100>,
  "horizon": "hourly"
}

Output rules (critical):
- Your entire message must be nothing but that JSON object (optionally wrapped in a ```json code block).
- Do not write analysis, markdown headings, bullet lists, or any text before or after the JSON.
- If you add any prose, the pipeline will fail — reply with JSON only (first non-whitespace character: open-brace or backtick for a fence).

Rules:
- Do not output short_confidence, rationale, or any short-trade logic.
- Stay focused on hourly logic only.
- Use only the provided data.
"""


SYSTEM_PROMPT = """
You are a trading signal model for a very short intraday strategy.

Focus strictly on HOURLY positions only: interpret the chart and momentum at the 1-hour timeframe.
Do not optimize for daily, weekly, or longer horizons—every judgment must be for an hourly position.

You receive one stock context in one of these forms:
- RAW: full hourly OHLCV candles.
- FEATURES: pre-computed technical indicators/statistics.
- HYBRID: features + small recent-candles snapshot.
When "market_regime" is present in the JSON, it is real benchmark data (SPY, QQQ, VIXY volatility ETF from the data provider), not fabricated.
Use features as primary signal. If recent bars are present, use them as secondary confirmation.

Trading intent (critical):
- We open a position for exactly one hour, then close it after that hour (no multi-hour holds).
- There are only two directional choices for this hour: open a LONG or open a SHORT.

You must score BOTH options independently (0–100 each):
- long_confidence = how favorable / likely-successful a 1-hour LONG would be (profit if price rises this hour).
- short_confidence = how favorable / likely-successful a 1-hour SHORT would be (profit if price falls this hour).

The downstream system will choose long if long_confidence >= short_confidence, else short. Ties favor long.
Do NOT output a single "action" only — always output BOTH numbers.

Return ONLY valid JSON with:
{
  "long_confidence": <integer 0-100>,
  "short_confidence": <integer 0-100>,
  "horizon": "hourly"
}

Output rules (critical):
- Your entire message must be nothing but that JSON object (optionally wrapped in a ```json code block).
- Do not write analysis, markdown headings, bullet lists, or any text before or after the JSON.
- Reply with JSON only — any prose will break parsing.

Rules:
- Both integers are required. They need not sum to 100; they are separate strength estimates.
- Do not output rationale or prose beyond the JSON.
- Stay focused on hourly position logic only; ignore longer-term bias unless it clearly affects the next hour.
- Think only about the next one-hour bar: open now, close after one hour.
- Use only the provided data.
"""


class ModelAnalyzer(Protocol):
    model_name: str

    def analyze(
        self, symbol: str, market_context: str, *, crypto: bool = False
    ) -> LLMDecision:
        ...


def _coerce_decision(model: str, symbol: str, raw_text: str, *, crypto: bool = False) -> LLMDecision:
    data = _parse_json_object(raw_text)
    return _llm_decision_from_json(model, symbol, data, crypto=crypto)


def _llm_decision_from_json(
    model: str, symbol: str, data: dict, *, crypto: bool = False
) -> LLMDecision:
    lc = data.get("long_confidence", data.get("longConfidence"))
    sc = data.get("short_confidence", data.get("shortConfidence"))
    horizon = str(data.get("horizon", "hourly")).strip()

    if crypto:
        if lc is None:
            raise ValueError("JSON must include long_confidence (integer 0-100).")
        return LLMDecision(
            model=model,
            symbol=symbol,
            long_confidence=int(lc),
            short_confidence=0,
            horizon=horizon,
            crypto_mode=True,
        )

    if lc is None or sc is None:
        raise ValueError(
            "JSON must include long_confidence and short_confidence (integers 0-100 each)."
        )
    return LLMDecision(
        model=model,
        symbol=symbol,
        long_confidence=int(lc),
        short_confidence=int(sc),
        horizon=horizon,
        crypto_mode=False,
    )


def _parse_json_object(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Model returned empty text; expected JSON decision object.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fenced ```json ... ``` (non-greedy inner object is often too small; try full fence body)
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # First balanced JSON object anywhere in the string (handles prose before the object)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    preview = text[:220].replace("\n", " ")
    raise ValueError(f"Unable to parse JSON from model output. Preview: {preview}")


def _is_model_not_found(err: Exception) -> bool:
    text = str(err).lower()
    return "model not found" in text or "not found" in text or "404" in text


class OpenAIAnalyzer:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.fallback_models = ["gpt-4o-mini"]

    def analyze(self, symbol: str, market_context: str, *, crypto: bool = False) -> LLMDecision:
        sys_prompt = SYSTEM_PROMPT_CRYPTO if crypto else SYSTEM_PROMPT
        model_candidates = [self.model_name] + self.fallback_models
        return _run_with_model_fallback(
            model_candidates=model_candidates,
            run_call=lambda model: self.client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": f"Symbol: {symbol}\n\nData:\n{market_context}",
                    },
                ],
            ).output_text,
            provider="OpenAI",
            output_model_name="chatgpt",
            symbol=symbol,
            crypto=crypto,
        )


class GeminiAnalyzer:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.fallback_models = ["gemini-1.5-flash", "gemini-2.0-flash"]

    def analyze(self, symbol: str, market_context: str, *, crypto: bool = False) -> LLMDecision:
        sys_prompt = SYSTEM_PROMPT_CRYPTO if crypto else SYSTEM_PROMPT
        model_candidates = [self.model_name] + self.fallback_models
        return _run_with_model_fallback(
            model_candidates=model_candidates,
            run_call=lambda model: (
                genai.GenerativeModel(model_name=model)
                .generate_content(
                    f"{sys_prompt}\n\nSymbol: {symbol}\n\nData:\n{market_context}"
                )
                .text
                or ""
            ),
            provider="Gemini",
            output_model_name="gemini",
            symbol=symbol,
            crypto=crypto,
        )


class ClaudeAnalyzer:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)
        self.fallback_models = ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]

    def analyze(self, symbol: str, market_context: str, *, crypto: bool = False) -> LLMDecision:
        sys_prompt = SYSTEM_PROMPT_CRYPTO if crypto else SYSTEM_PROMPT
        model_candidates = [self.model_name] + self.fallback_models
        return _run_with_model_fallback(
            model_candidates=model_candidates,
            run_call=lambda model: "".join(
                block.text
                for block in self.client.messages.create(
                    model=model,
                    max_tokens=800,
                    system=sys_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Symbol: {symbol}\n\nData:\n{market_context}\n\n"
                                "Respond with ONLY the JSON object (no analysis, no markdown body)."
                            ),
                        }
                    ],
                ).content
                if getattr(block, "type", "") == "text"
            ).strip(),
            provider="Claude",
            output_model_name="claude",
            symbol=symbol,
            crypto=crypto,
        )


class GrokAnalyzer:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self.fallback_models = ["grok-beta"]

    def analyze(self, symbol: str, market_context: str, *, crypto: bool = False) -> LLMDecision:
        sys_prompt = SYSTEM_PROMPT_CRYPTO if crypto else SYSTEM_PROMPT
        model_candidates = [self.model_name] + self.fallback_models
        return _run_with_model_fallback(
            model_candidates=model_candidates,
            run_call=lambda model: self.client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": f"Symbol: {symbol}\n\nData:\n{market_context}",
                    },
                ],
            ).output_text,
            provider="Grok",
            output_model_name="grok",
            symbol=symbol,
            crypto=crypto,
        )


def _run_with_model_fallback(
    model_candidates: List[str],
    run_call: Callable[[str], str],
    provider: str,
    output_model_name: str,
    symbol: str,
    *,
    crypto: bool = False,
) -> LLMDecision:
    errors: List[str] = []
    tried = set()
    for model in model_candidates:
        if model in tried:
            continue
        tried.add(model)
        try:
            text = run_call(model).strip()
            return _coerce_decision(output_model_name, symbol, text, crypto=crypto)
        except Exception as exc:
            if _is_model_not_found(exc):
                errors.append(f"{model}: {exc}")
                continue
            raise ValueError(f"{provider} ({model}) failed: {exc}")
    raise ValueError(
        f"{provider} failed: no valid model found. Tried {', '.join(tried)}. "
        f"Errors: {' | '.join(errors)}"
    )
