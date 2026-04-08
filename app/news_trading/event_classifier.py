"""
event_classifier.py — pure Python, synchronous, no API calls.

Fast headline classification before spending LLM credits. Returns an
EventClassification describing impact level and direction hint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EventClassification:
    headline: str
    symbols: List[str]
    asset_class: str             # "equity" or "crypto"
    impact: str                  # "high", "medium", "low", "skip"
    direction_hint: str          # "bullish", "bearish", "uncertain"
    category: str                # "earnings", "fda", "macro", "merger", etc.
    requires_llm_analysis: bool
    skip_reason: Optional[str]


# ── Keyword sets ──────────────────────────────────────────────────────────────

_HIGH_EARNINGS = {
    "beats", "misses", "earnings beat", "earnings miss", "eps beat", "eps miss",
    "revenue beat", "revenue miss", "raises guidance", "lowers guidance",
    "withdraws guidance",
}

_HIGH_FDA = {
    "fda approval", "fda approves", "fda rejects", "fda rejection", "fda holds",
    "complete response letter", "breakthrough therapy",
}

_HIGH_CORPORATE = {
    "merger", "acquisition", "acquires", "buyout", "takeover",
    "going private", "hostile bid",
}

_HIGH_LEADERSHIP = {
    "ceo resign", "ceo fired", "ceo steps down", "coo resign", "cfo resign",
}

_HIGH_CRISIS = {
    "bankruptcy", "chapter 11", "chapter 7", "fraud", "sec investigation",
    "doj investigation", "criminal charges", "restatement",
}

_HIGH_CAPITAL = {
    "special dividend", "dividend cut", "dividend eliminated",
    "dividend increase", "stock split", "share buyback", "tender offer",
}

_HIGH_CRYPTO = {
    "etf approval", "etf rejected", "exchange hack", "depegged",
    "regulatory ban", "legal tender", "sec crypto", "cftc crypto",
}

_MEDIUM_ANALYST = {
    "analyst upgrade", "analyst downgrade", "price target raised", "price target cut",
    "initiates coverage", "outperform", "underperform",
}

_MEDIUM_PRODUCT = {
    "product launch", "product recall", "partnership", "joint venture",
    "licensing deal", "government contract",
}

_MEDIUM_RESTRUCTURE = {
    "layoffs", "restructuring", "plant closure",
}

_LOW_SKIP = {
    "reports earnings", "scheduled to report", "expected to report",
    "will report", "set to report",
}

# Direction hints
_BULLISH_KEYWORDS = {
    "beats", "beat", "approval", "approves", "raises guidance", "special dividend",
    "buyback", "share buyback", "etf approval", "upgrade", "outperform",
    "acquires",   # target company
    "merger",     # target usually pops
    "dividend increase",
}

_BEARISH_KEYWORDS = {
    "miss", "misses", "rejection", "rejects", "lowers guidance", "bankruptcy",
    "chapter 11", "chapter 7", "fraud", "investigation", "ceo resign", "ceo fired",
    "ceo steps down", "recall", "downgrade", "underperform", "exchange hack",
    "depegged", "regulatory ban", "criminal charges", "restatement",
    "dividend cut", "dividend eliminated", "withdraws guidance",
}

# Category labels (first match wins)
_CATEGORY_MAP = [
    (frozenset(_HIGH_EARNINGS), "earnings"),
    (frozenset(_HIGH_FDA), "fda"),
    (frozenset(_HIGH_CORPORATE), "merger"),
    (frozenset(_HIGH_LEADERSHIP), "leadership"),
    (frozenset(_HIGH_CRISIS), "crisis"),
    (frozenset(_HIGH_CAPITAL), "capital_action"),
    (frozenset(_HIGH_CRYPTO), "crypto_regulatory"),
    (frozenset(_MEDIUM_ANALYST), "analyst"),
    (frozenset(_MEDIUM_PRODUCT), "product"),
    (frozenset(_MEDIUM_RESTRUCTURE), "restructure"),
]


def _headline_lower(headline: str) -> str:
    return headline.lower()


def _matches_any(text: str, keywords: set) -> bool:
    return any(kw in text for kw in keywords)


def _detect_category(text_lower: str) -> str:
    for kws, label in _CATEGORY_MAP:
        if _matches_any(text_lower, kws):
            return label
    return "general"


def _detect_direction(text_lower: str) -> str:
    bullish = _matches_any(text_lower, _BULLISH_KEYWORDS)
    bearish = _matches_any(text_lower, _BEARISH_KEYWORDS)
    if bullish and not bearish:
        return "bullish"
    if bearish and not bullish:
        return "bearish"
    return "uncertain"


def classify_event(
    headline: str,
    symbols: List[str],
    asset_class: str = "equity",
) -> EventClassification:
    """
    Classify a headline into impact level and direction hint.
    Pure Python — no IO, no LLM calls.
    """
    text = _headline_lower(headline)
    category = _detect_category(text)
    direction = _detect_direction(text)

    # Skip: no symbols attached and no crypto context
    if not symbols and asset_class == "equity":
        return EventClassification(
            headline=headline,
            symbols=symbols,
            asset_class=asset_class,
            impact="skip",
            direction_hint=direction,
            category=category,
            requires_llm_analysis=False,
            skip_reason="no_symbols",
        )

    # Skip: low-value announcements
    if _matches_any(text, _LOW_SKIP):
        return EventClassification(
            headline=headline,
            symbols=symbols,
            asset_class=asset_class,
            impact="low",
            direction_hint=direction,
            category=category,
            requires_llm_analysis=False,
            skip_reason="pre_event_announcement",
        )

    # HIGH impact
    high_sets = [
        _HIGH_EARNINGS, _HIGH_FDA, _HIGH_CORPORATE, _HIGH_LEADERSHIP,
        _HIGH_CRISIS, _HIGH_CAPITAL, _HIGH_CRYPTO,
    ]
    if any(_matches_any(text, s) for s in high_sets):
        return EventClassification(
            headline=headline,
            symbols=symbols,
            asset_class=asset_class,
            impact="high",
            direction_hint=direction,
            category=category,
            requires_llm_analysis=True,
            skip_reason=None,
        )

    # MEDIUM impact
    medium_sets = [_MEDIUM_ANALYST, _MEDIUM_PRODUCT, _MEDIUM_RESTRUCTURE]
    if any(_matches_any(text, s) for s in medium_sets):
        return EventClassification(
            headline=headline,
            symbols=symbols,
            asset_class=asset_class,
            impact="medium",
            direction_hint=direction,
            category=category,
            requires_llm_analysis=True,
            skip_reason=None,
        )

    # Default skip: generic headline with no signal keywords
    return EventClassification(
        headline=headline,
        symbols=symbols,
        asset_class=asset_class,
        impact="skip",
        direction_hint=direction,
        category="general",
        requires_llm_analysis=False,
        skip_reason="no_signal_keywords",
    )


def is_earnings_season_risk(symbol: str) -> bool:
    """Stub — returns False until an earnings calendar feed is wired in."""
    return False


def should_skip_due_to_existing_position(symbol: str, open_positions: set) -> bool:
    """True if the symbol already has an open news-trade position."""
    return symbol.upper() in {s.upper() for s in open_positions}
