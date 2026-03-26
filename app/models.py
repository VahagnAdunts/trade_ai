from __future__ import annotations

from datetime import datetime
from typing import Literal, List, Optional

from pydantic import BaseModel, Field, computed_field, field_validator


# Chosen side for the hourly trade (derived from long_confidence vs short_confidence in LLMDecision)
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
    Model outputs a score for BOTH long and short (0–100 each).
    We pick the side with the higher score; ties go to long.
    """

    model: str
    symbol: str
    long_confidence: int = Field(ge=0, le=100, description="How favorable a 1h long is")
    short_confidence: int = Field(ge=0, le=100, description="How favorable a 1h short is")
    rationale: str
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
        """Confidence of the chosen side (the larger of the two scores)."""
        return max(self.long_confidence, self.short_confidence)


class StockAnalysis(BaseModel):
    symbol: str
    decisions: List[LLMDecision]


class ConsensusResult(BaseModel):
    symbol: str
    aligned_action: Optional[Side]
    minimum_confidence: int
    passes_threshold: bool
    model_count: int
