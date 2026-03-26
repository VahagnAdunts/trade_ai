from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelStats:
    name: str
    correct: int = 0
    wrong: int = 0
    skipped: int = 0  # no ground truth (flat return) or API error

    @property
    def total_decided(self) -> int:
        return self.correct + self.wrong

    @property
    def accuracy(self) -> Optional[float]:
        if self.total_decided == 0:
            return None
        return self.correct / self.total_decided


def summarize_runs(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, ModelStats] = {}
    for row in rows:
        for key, comp in row.get("per_model", {}).items():
            if key not in by_model:
                by_model[key] = ModelStats(name=key)
            st = by_model[key]
            if comp.get("skipped"):
                st.skipped += 1
            elif comp.get("correct") is True:
                st.correct += 1
            elif comp.get("correct") is False:
                st.wrong += 1
            else:
                st.skipped += 1

    return {
        "models": {
            k: {
                "correct": v.correct,
                "wrong": v.wrong,
                "skipped": v.skipped,
                "accuracy": v.accuracy,
            }
            for k, v in by_model.items()
        }
    }
