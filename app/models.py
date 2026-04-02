from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field, computed_field, field_validator

if TYPE_CHECKING:
    from app.config import AppConfig


# Chosen side for the hourly trade (equity: derived from long vs short scores; crypto: long-only pipeline)
Side = Literal["long", "short"]


@dataclass(frozen=True)
class _ConsensusRuntime:
    min_models: int
    min_confidence_equity: int
    min_confidence_crypto: int


# Defaults match .env.example; call configure_consensus_from_config when AppConfig is loaded.
_consensus = _ConsensusRuntime(3, 60, 70)


def configure_consensus_from_config(config: "AppConfig") -> None:
    """Sync Pydantic computed fields (e.g. LLMDecision.action in crypto mode) with env-driven thresholds."""
    global _consensus
    _consensus = _ConsensusRuntime(
        config.consensus_min_models,
        config.consensus_min_confidence_pct,
        config.consensus_min_confidence_crypto_pct,
    )


class OHLCVPoint(BaseModel):
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0


class LLMDecision(BaseModel):
    """
    Equity: scores for BOTH long and short (0–100 each); higher side wins (ties → long).
    Crypto (spot): only long_confidence — whether a long this hour is worthwhile; short_confidence is 0.
    """

    model: str
    symbol: str
    long_confidence: int = Field(
        ge=0,
        le=100,
        description="Equity: 1h long favorability. Crypto: worth entering a long this hour.",
    )
    short_confidence: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Equity: 1h short favorability. Crypto: unused (0).",
    )
    rationale: str = ""
    horizon: str = "hourly"
    crypto_mode: bool = Field(
        default=False,
        description="If True, spot-crypto long-only prompt; consensus uses long_confidence only.",
    )

    @field_validator("horizon")
    @classmethod
    def horizon_must_be_hourly(cls, value: str) -> str:
        if "hour" not in value.lower():
            raise ValueError("Horizon must indicate hourly.")
        return value

    @computed_field
    @property
    def action(self) -> Side:
        if self.crypto_mode:
            return (
                "long"
                if self.long_confidence >= _consensus.min_confidence_crypto
                else "short"
            )
        if self.long_confidence >= self.short_confidence:
            return "long"
        return "short"

    @computed_field
    @property
    def confidence(self) -> int:
        if self.crypto_mode:
            return self.long_confidence
        return max(self.long_confidence, self.short_confidence)


class ConsensusResult(BaseModel):
    symbol: str
    aligned_action: Optional[Side]
    minimum_confidence: int
    passes_threshold: bool
    model_count: int
