"""
news_prompt.py — builds LLM prompts for news-driven entry and exit decisions.

Uses the same "thinking" JSON format established in llm_clients.py, extended
with news-specific fields (news_impact, max_hold_minutes, invalidation).
"""
from __future__ import annotations

import json
from typing import Tuple


FAST_CLASSIFY_SYSTEM_PROMPT = """\
You are a financial news classifier. Decide quickly if a news article is relevant to a specific publicly traded stock.

Return ONLY valid JSON with exactly these keys:
{
  "is_relevant": true|false,
  "symbol": "<TICKER or empty string if not relevant>",
  "reason": "<one sentence>"
}

Rules:
- is_relevant = true only if this news could meaningfully move the price of a specific stock TODAY.
- symbol = the primary affected US equity ticker (e.g. "AAPL", "NVDA"). If multiple stocks, pick the most directly affected one.
- If the article is general market commentary, macro, or unrelated to any specific stock, set is_relevant=false and symbol="".
- No text outside the JSON object.
"""


def build_fast_classify_prompt(headline: str, summary: str) -> tuple:
    """Returns (system_prompt, user_message) for fast symbol/relevance classification."""
    user_msg = f"Headline: {headline}\nSummary: {summary or '(none)'}"
    return FAST_CLASSIFY_SYSTEM_PROMPT, user_msg


NEWS_ENTRY_SYSTEM_PROMPT = """\
You are an event-driven trading signal model. A significant news event just broke.
Your job: decide whether to open a position RIGHT NOW based on the news catalyst
and the pre-news technical setup.

You will receive:
- The news headline and summary
- Pre-computed technical indicators (features snapshot)
- Market regime context

Your decision covers the NEXT 15-120 MINUTES following this news event.

Think step by step in the "thinking" field:
1. Is this news genuinely significant for this symbol's price?
2. Is the impact direction clear (bullish/bearish) or ambiguous?
3. How much of the move might already be priced in?
4. Does the technical setup support or conflict with the news direction?
5. What is your confidence in a directional move over the next 15-120 minutes?

Return ONLY valid JSON with exactly these keys:
{
  "thinking": "<step-by-step reasoning>",
  "long_confidence": <integer 0-100>,
  "short_confidence": <integer 0-100>,
  "max_hold_minutes": <integer 15-120>,
  "invalidation": "<1 sentence: what would immediately invalidate this trade>",
  "horizon": "event_driven"
}

Rules:
- long_confidence and short_confidence must be consistent with your thinking.
- max_hold_minutes: use 15-30 for fast/volatile events, 60-120 for slower catalysts.
- All fields required. No text outside the JSON object.
"""

NEWS_EXIT_SYSTEM_PROMPT = """\
You are monitoring an open trading position opened on a news event.
Decide: should we HOLD or CLOSE this position RIGHT NOW?

Consider:
1. Has the news-driven move already played out?
2. Is the invalidation condition triggered?
3. Is momentum slowing or reversing against the position?
4. Is the P&L deteriorating or well in profit?

Return ONLY valid JSON:
{
  "thinking": "<2-3 sentence reasoning>",
  "decision": "hold|close",
  "urgency": "normal|urgent"
}

Rules:
- "urgent" means close even if fewer models agree (e.g. stop-loss scenario approaching).
- All fields required. No text outside the JSON.
"""


def build_news_entry_prompt(
    symbol: str,
    headline: str,
    summary: str,
    features: dict,
    market_regime: dict,
    direction_hint: str,
    asset_class: str,
    content: str = "",
) -> Tuple[str, str]:
    """
    Returns (system_prompt, user_message) for news entry analysis.
    content: full article body (plain text, stripped of HTML). If available, appended after summary.
    """
    # Use full content when available, truncated to 2000 chars to stay within token budgets
    article_body = ""
    if content and content.strip():
        truncated = content.strip()[:2000]
        if len(content.strip()) > 2000:
            truncated += "... [truncated]"
        article_body = f"\nFull article:\n{truncated}"

    user_msg = (
        f"NEWS EVENT:\n"
        f"Headline: {headline}\n"
        f"Summary: {summary}"
        f"{article_body}\n"
        f"\n"
        f"TECHNICAL CONTEXT (pre-news, {asset_class}):\n"
        f"{json.dumps(features, indent=2)}\n"
        f"\n"
        f"MARKET REGIME:\n"
        f"{json.dumps(market_regime, indent=2)}\n"
        f"\n"
        f"Symbol: {symbol}"
    )
    return NEWS_ENTRY_SYSTEM_PROMPT, user_msg


def build_news_exit_prompt(
    symbol: str,
    side: str,
    entry_price: float,
    current_price: float,
    pnl_pct: float,
    minutes_held: float,
    original_headline: str,
    original_thinking: str,
    invalidation_condition: str,
    current_rsi: float,
    price_momentum_5m: float,
    volume_ratio: float,
) -> Tuple[str, str]:
    """
    Returns (system_prompt, user_message) for position exit decision.
    """
    user_msg = (
        f"ORIGINAL TRADE:\n"
        f"Symbol: {symbol}\n"
        f"Direction: {side}\n"
        f'News: "{original_headline}"\n'
        f"Original reasoning: {original_thinking}\n"
        f"Invalidation condition: {invalidation_condition}\n"
        f"\n"
        f"CURRENT STATUS:\n"
        f"Entry price: {entry_price}\n"
        f"Current price: {current_price}\n"
        f"P&L: {pnl_pct:+.2f}%\n"
        f"Time held: {minutes_held:.0f} minutes\n"
        f"Current RSI: {current_rsi:.1f}\n"
        f"Price momentum (last 5 min): {price_momentum_5m:+.3f}%\n"
        f"Volume vs 20-min average: {volume_ratio:.1f}x\n"
        f"\n"
        f"Should we hold or close?"
    )
    return NEWS_EXIT_SYSTEM_PROMPT, user_msg
