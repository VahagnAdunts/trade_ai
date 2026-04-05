from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field, field_validator


# Chosen side for the hourly trade (higher long vs short score wins; ties → long)
Side = Literal["long", "short"]


class OHLCVPoint(BaseModel):
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0


class LLMDecision(BaseModel):
    """
    Long and short favorability (0–100 each); higher side wins (ties → long).
    Same schema for equities and crypto.
    """

    model: str
    symbol: str
    long_confidence: int = Field(
        ge=0,
        le=100,
        description="1h long favorability.",
    )
    short_confidence: int = Field(
        default=0,
        ge=0,
        le=100,
        description="1h short favorability.",
    )
    rationale: str = Field(
        default="",
        description="Chain-of-thought from the model (thinking field); not used for scoring.",
    )
    horizon: str = "hourly"

    @field_validator("horizon")
    @classmethod
    def horizon_must_be_hourly(cls, value: str) -> str:
        if "hour" not in value.lower():
            raise ValueError("Horizon must indicate hourly.")
        return value

    @computed_field
    @property
    def action(self) -> Side:
        if self.long_confidence >= self.short_confidence:
            return "long"
        return "short"

    @computed_field
    @property
    def confidence(self) -> int:
        return max(self.long_confidence, self.short_confidence)


class ConsensusResult(BaseModel):
    symbol: str
    aligned_action: Optional[Side]
    minimum_confidence: int
    passes_threshold: bool
    model_count: int
