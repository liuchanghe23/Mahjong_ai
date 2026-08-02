"""Grouped, normalized action scoring kept separate for weight fitting."""

from mahjong_ai.baseline.config import BaselineConfig
from mahjong_ai.baseline.features import CandidateFeatures


FEATURE_GROUPS = {
    "ukeire_count": "efficiency", "ukeire_types": "efficiency",
    "retained_dora": "value", "retained_red": "value",
    "yaku_yakuhai_delta": "value", "yaku_tanyao_delta": "value",
    "yaku_chiitoitsu_delta": "value", "yaku_flush_delta": "value",
    "complete_meld": "shape", "required_ryanmen": "shape",
    "required_kanchan": "shape", "required_penchan": "shape",
    "head_pair": "shape", "extra_pair": "shape", "unused_middle": "shape",
    "unused_near_terminal": "shape", "unused_terminal": "shape",
    "unused_honor": "shape",
    "shape_flexibility": "shape", "legacy_value_honor_pair": "shape",
    "discard_risk": "risk",
}


def linear_score(features: CandidateFeatures, config: BaselineConfig) -> tuple[float, dict[str, float]]:
    contributions = {
        name: value * config.weights[name] * config.group_weights[FEATURE_GROUPS[name]]
        for name, value in features.normalized_values.items()
    }
    return sum(contributions.values()), contributions


def aggregate_group_contributions(contributions: dict[str, float]) -> dict[str, float]:
    groups = {name: 0.0 for name in {group for group in FEATURE_GROUPS.values()}}
    for name, contribution in contributions.items():
        groups[FEATURE_GROUPS[name]] += contribution
    return groups
