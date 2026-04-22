from __future__ import annotations

import json
import re
from typing import Callable, List, Protocol

import anthropic
import google.generativeai as genai
from openai import OpenAI

from app.models import LLMDecision

# Cap stored rationale (from "thinking") to keep reports and payloads bounded.
_MAX_THINKING_CHARS = 8000


SYSTEM_PROMPT_CRYPTO = """
You are a trading signal model for a very short intraday strategy on CRYPTOCURRENCY (USD-quoted pairs).

Focus strictly on HOURLY positions only: interpret the chart and momentum at the 1-hour timeframe.
Do not optimize for daily, weekly, or longer horizons—every judgment must be for an hourly position.

You receive one crypto pair context in one of these forms:
- RAW: full hourly OHLCV candles.
- FEATURES: pre-computed technical indicators/statistics.
- HYBRID: features + small recent-candles snapshot.
When "market_regime" is present in the JSON, it is real benchmark data (BTC/ETH vs this pair from the data provider), not fabricated.
Use features as primary signal. If recent bars are present, use them as secondary confirmation.

Trading intent (critical):
- We open a directional position for exactly one hour, then close it after that hour (no multi-hour holds).
- There are only two directional choices for this hour: favor a LONG (price rise over the hour) or a SHORT (price fall over the hour).
  Execution venue may be spot, margin, or derivatives — the scores only express which direction is more attractive this hour.

You must score BOTH options independently (0–100 each):
- long_confidence = how favorable / likely-successful a 1-hour long bias would be (profit if price rises this hour).
- short_confidence = how favorable / likely-successful a 1-hour short bias would be (profit if price falls this hour).

The downstream system will choose long if long_confidence >= short_confidence, else short. Ties favor long.
Do NOT output a single "action" only — always output BOTH numbers.

Reasoning (critical for quality):
- First reason step-by-step in the "thinking" string (indicators, regime, conflicts, uncertainty).
- Then set long_confidence and short_confidence to match that reasoning. Scores must be consistent with thinking.

Return ONLY valid JSON with exactly these keys:
{
  "thinking": "<brief chain-of-thought: what you observe and why before scoring>",
  "long_confidence": <integer 0-100>,
  "short_confidence": <integer 0-100>,
  "horizon": "hourly"
}

Output rules (critical):
- Your entire message must be nothing but that JSON object (optionally wrapped in a ```json code block).
- Do not put any text outside the JSON object (no markdown headings or preamble).
- All prose belongs inside the "thinking" string.

Rules:
- "thinking" is required (2–8 sentences). Both integers are required. They need not sum to 100.
- Stay focused on hourly position logic only; ignore longer-term bias unless it clearly affects the next hour.
- Think only about the next one-hour bar: open now, close after one hour.
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

Reasoning (critical for quality):
- First reason step-by-step in the "thinking" string (indicators, regime, conflicts, uncertainty).
- Then set long_confidence and short_confidence to match that reasoning. Scores must be consistent with thinking.

Return ONLY valid JSON with exactly these keys:
{
  "thinking": "<brief chain-of-thought: what you observe and why before scoring>",
  "long_confidence": <integer 0-100>,
  "short_confidence": <integer 0-100>,
  "horizon": "hourly"
}

Output rules (critical):
- Your entire message must be nothing but that JSON object (optionally wrapped in a ```json code block).
- Do not put any text outside the JSON object (no markdown headings or preamble).
- All prose belongs inside the "thinking" string.

Rules:
- "thinking" is required (2–8 sentences). Both integers are required. They need not sum to 100.
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


def _coerce_decision(model: str, symbol: str, raw_text: str) -> LLMDecision:
    data = _parse_json_object(raw_text)
    return _llm_decision_from_json(model, symbol, data)


def _thinking_to_rationale(data: dict) -> str:
    """Map scratchpad-style keys to a single stored rationale (not fed back to the model)."""
    raw = (
        data.get("thinking")
        or data.get("reasoning")
        or data.get("scratchpad")
        or data.get("chain_of_thought")
    )
    if raw is None:
        return ""
    text = str(raw).strip()
    if len(text) > _MAX_THINKING_CHARS:
        return text[: _MAX_THINKING_CHARS] + "…"
    return text


def _llm_decision_from_json(model: str, symbol: str, data: dict) -> LLMDecision:
    lc = data.get("long_confidence", data.get("longConfidence"))
    sc = data.get("short_confidence", data.get("shortConfidence"))
    horizon = str(data.get("horizon", "hourly")).strip()
    rationale = _thinking_to_rationale(data)

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
        rationale=rationale,
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


_EXIT_MODELS = {
    "OpenAI": "gpt-4o-mini",
    "Gemini": "gemini-2.0-flash",
    # Small/cheap Claude for quick exit prompts; avoid deprecated *-latest ids (404 on API).
    "Claude": "claude-haiku-4-5",
    "Grok": "grok-beta",
}


class OpenAIAnalyzer:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def analyze(self, symbol: str, market_context: str, *, crypto: bool = False) -> LLMDecision:
        sys_prompt = SYSTEM_PROMPT_CRYPTO if crypto else SYSTEM_PROMPT
        return _run_configured_model(
            model_candidates=[self.model_name],
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
        )

    def quick_exit_decision(self, sys_prompt: str, user_msg: str) -> dict:
        try:
            text = self.client.responses.create(
                model=_EXIT_MODELS["OpenAI"],
                input=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
            ).output_text.strip()
            data = _parse_json_object(text)
            return {
                "decision": data.get("decision", "hold"),
                "urgency": data.get("urgency", "normal"),
            }
        except Exception:
            return {"decision": "hold", "urgency": "normal"}


class GeminiAnalyzer:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        genai.configure(api_key=api_key)

    def analyze(self, symbol: str, market_context: str, *, crypto: bool = False) -> LLMDecision:
        sys_prompt = SYSTEM_PROMPT_CRYPTO if crypto else SYSTEM_PROMPT
        return _run_configured_model(
            model_candidates=[self.model_name],
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
        )

    def quick_exit_decision(self, sys_prompt: str, user_msg: str) -> dict:
        try:
            text = (
                genai.GenerativeModel(model_name=_EXIT_MODELS["Gemini"])
                .generate_content(f"{sys_prompt}\n\n{user_msg}")
                .text
                or ""
            ).strip()
            data = _parse_json_object(text)
            return {
                "decision": data.get("decision", "hold"),
                "urgency": data.get("urgency", "normal"),
            }
        except Exception:
            return {"decision": "hold", "urgency": "normal"}


class ClaudeAnalyzer:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def analyze(self, symbol: str, market_context: str, *, crypto: bool = False) -> LLMDecision:
        sys_prompt = SYSTEM_PROMPT_CRYPTO if crypto else SYSTEM_PROMPT
        return _run_configured_model(
            model_candidates=[self.model_name],
            run_call=lambda model: "".join(
                block.text
                for block in self.client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=sys_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Symbol: {symbol}\n\nData:\n{market_context}\n\n"
                                "Output a single JSON object only (thinking + scores per system prompt; "
                                "no text outside the JSON)."
                            ),
                        }
                    ],
                ).content
                if getattr(block, "type", "") == "text"
            ).strip(),
            provider="Claude",
            output_model_name="claude",
            symbol=symbol,
        )

    def quick_exit_decision(self, sys_prompt: str, user_msg: str) -> dict:
        try:
            text = "".join(
                block.text
                for block in self.client.messages.create(
                    model=_EXIT_MODELS["Claude"],
                    max_tokens=512,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": user_msg}],
                ).content
                if getattr(block, "type", "") == "text"
            ).strip()
            data = _parse_json_object(text)
            return {
                "decision": data.get("decision", "hold"),
                "urgency": data.get("urgency", "normal"),
            }
        except Exception:
            return {"decision": "hold", "urgency": "normal"}


class GrokAnalyzer:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    def analyze(self, symbol: str, market_context: str, *, crypto: bool = False) -> LLMDecision:
        sys_prompt = SYSTEM_PROMPT_CRYPTO if crypto else SYSTEM_PROMPT
        return _run_configured_model(
            model_candidates=[self.model_name],
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
        )

    def quick_exit_decision(self, sys_prompt: str, user_msg: str) -> dict:
        try:
            text = self.client.responses.create(
                model=_EXIT_MODELS["Grok"],
                input=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
            ).output_text.strip()
            data = _parse_json_object(text)
            return {
                "decision": data.get("decision", "hold"),
                "urgency": data.get("urgency", "normal"),
            }
        except Exception:
            return {"decision": "hold", "urgency": "normal"}


def _run_configured_model(
    model_candidates: List[str],
    run_call: Callable[[str], str],
    provider: str,
    output_model_name: str,
    symbol: str,
) -> LLMDecision:
    """Calls the API with the configured model id only (from env / AppConfig)."""
    if not model_candidates or not (model_candidates[0] or "").strip():
        raise ValueError(f"{provider}: model name is empty; set the matching *_MODEL in .env")
    model = model_candidates[0].strip()
    try:
        text = run_call(model).strip()
        return _coerce_decision(output_model_name, symbol, text)
    except Exception as exc:
        raise ValueError(f"{provider} ({model}) failed: {exc}") from exc
