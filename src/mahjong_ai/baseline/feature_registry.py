"""Single source of truth for baseline feature metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    normalization: str | None = None


FEATURE_SPECS = (
    FeatureSpec("ukeire_count", "efficiency", "ukeire_count"),
    FeatureSpec("ukeire_types", "efficiency", "ukeire_types"),
    FeatureSpec("lookahead_win_probability", "efficiency"),
    FeatureSpec("lookahead_tenpai_probability", "efficiency"),
    FeatureSpec("lookahead_wide_wait_probability", "efficiency"),
    FeatureSpec("lookahead_furiten_probability", "efficiency"),
    FeatureSpec("lookahead_expected_ukeire", "efficiency", "ukeire_count"),
    FeatureSpec("retained_dora", "value", "hand_tiles"),
    FeatureSpec("retained_red", "value", "hand_tiles"),
    FeatureSpec("yaku_yakuhai_delta", "value", "yaku_potential"),
    FeatureSpec("yaku_tanyao_delta", "value", "yaku_potential"),
    FeatureSpec("yaku_chiitoitsu_delta", "value", "yaku_potential"),
    FeatureSpec("yaku_flush_delta", "value", "yaku_potential"),
    FeatureSpec("complete_meld", "shape", "hand_tiles"),
    FeatureSpec("ryanmen_taatsu", "shape", "hand_tiles"),
    FeatureSpec("kanchan_taatsu", "shape", "hand_tiles"),
    FeatureSpec("penchan_taatsu", "shape", "hand_tiles"),
    FeatureSpec("head_pair", "shape", "hand_tiles"),
    FeatureSpec("extra_pair", "shape", "hand_tiles"),
    FeatureSpec("unused_middle", "shape", "hand_tiles"),
    FeatureSpec("unused_near_terminal", "shape", "hand_tiles"),
    FeatureSpec("unused_terminal", "shape", "hand_tiles"),
    FeatureSpec("unused_honor", "shape", "hand_tiles"),
    FeatureSpec("legacy_value_honor_pair", "shape", "hand_tiles"),
    FeatureSpec("discard_risk", "risk"),
)

FEATURE_BY_NAME = {spec.name: spec for spec in FEATURE_SPECS}
FEATURE_NAMES = frozenset(FEATURE_BY_NAME)
FEATURE_GROUPS = frozenset(spec.group for spec in FEATURE_SPECS)
NORMALIZATION_NAMES = frozenset(
    spec.normalization for spec in FEATURE_SPECS if spec.normalization is not None
)


def feature_names(group: str) -> frozenset[str]:
    return frozenset(spec.name for spec in FEATURE_SPECS if spec.group == group)


def normalize_features(values: dict[str, float], scales: dict[str, float]) -> dict[str, float]:
    unknown = set(values) - FEATURE_NAMES
    if unknown:
        raise ValueError(f"Unregistered extracted features: {sorted(unknown)}")
    return {
        name: value / scales[spec.normalization] if spec.normalization else value
        for name, value in values.items()
        for spec in (FEATURE_BY_NAME[name],)
    }
