"""Decision records used by logs, UI explanations, and future training."""

from dataclasses import dataclass
from typing import Any

from mahjong_ai.baseline.yaku import YakuEvaluation


@dataclass(frozen=True)
class CandidateEvaluation:
    action: dict[str, Any]
    score: float
    features: dict[str, float]
    normalized_features: dict[str, float]
    contributions: dict[str, float]
    group_contributions: dict[str, float]
    shanten: int | None
    ukeire_kinds: tuple[int, ...]
    ukeire_count: int
    danger: float
    reasons: tuple[str, ...]
    yaku_before: tuple[YakuEvaluation, ...] = ()
    yaku_after: tuple[YakuEvaluation, ...] = ()
    pruned: bool = False
    prune_reason: str | None = None


@dataclass(frozen=True)
class DecisionRecord:
    policy: str
    selected: dict[str, Any]
    candidates: tuple[CandidateEvaluation, ...]
