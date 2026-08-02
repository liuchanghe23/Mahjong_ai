"""Grouped, normalized action scoring kept separate for weight fitting."""

from mahjong_ai.baseline.config import BaselineConfig
from mahjong_ai.baseline.feature_registry import FEATURE_BY_NAME, FEATURE_GROUPS
from mahjong_ai.baseline.features import CandidateFeatures


def linear_score(features: CandidateFeatures, config: BaselineConfig) -> tuple[float, dict[str, float]]:
    contributions = {
        name: value * config.weights[name] * config.group_weights[FEATURE_BY_NAME[name].group]
        for name, value in features.normalized_values.items()
    }
    return sum(contributions.values()), contributions


def select_best_variant(
    candidates: tuple[CandidateFeatures, ...], config: BaselineConfig
) -> tuple[float, dict[str, float], CandidateFeatures]:
    """Let configured weights, rather than the recognizer, choose a decomposition."""
    scored = [(*linear_score(features, config), features) for features in candidates]
    return max(
        scored,
        key=lambda item: (item[0], tuple(sorted(item[2].normalized_values.items()))),
    )


def aggregate_group_contributions(contributions: dict[str, float]) -> dict[str, float]:
    groups = {name: 0.0 for name in FEATURE_GROUPS}
    for name, contribution in contributions.items():
        groups[FEATURE_BY_NAME[name].group] += contribution
    return groups
