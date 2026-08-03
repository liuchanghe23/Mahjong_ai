"""Orchestration for modular discard feature extractors."""

from dataclasses import dataclass

from mahjong_ai.baseline.config import BaselineConfig, YakuConfig
from mahjong_ai.baseline.efficiency_features import extract_efficiency
from mahjong_ai.baseline.feature_registry import feature_names, normalize_features
from mahjong_ai.baseline.lookahead_features import extract_lookahead
from mahjong_ai.baseline.risk_features import discard_danger, scale_risk
from mahjong_ai.baseline.shape_features import extract_shape_candidates
from mahjong_ai.baseline.state import PublicState
from mahjong_ai.baseline.tiles import tile_kind
from mahjong_ai.baseline.value_features import extract_value
from mahjong_ai.baseline.yaku import YakuDelta


DEFAULT_NORMALIZATION = {
    "ukeire_count": 40.0,
    "ukeire_types": 34.0,
    "hand_tiles": 13.0,
    "yaku_potential": 2.0,
}


@dataclass(frozen=True)
class CandidateFeatures:
    values: dict[str, float]
    normalized_values: dict[str, float]
    shanten: int
    ukeire_kinds: tuple[int, ...]
    ukeire_count: int
    danger: float
    yaku: YakuDelta | None = None


def remove_one_tile(hand: tuple[int, ...], tile_id: int) -> list[int]:
    remaining = list(hand)
    remaining.remove(tile_id)
    return remaining


@dataclass(frozen=True)
class FeaturePipeline:
    """Configured composition of independent baseline feature extractors."""

    danger: dict[str, float]
    yaku_config: YakuConfig | None
    normalization: dict[str, float]
    risk_context: dict[str, float] | None
    shape_mode: str
    compute_shape: bool
    compute_lookahead: bool

    @classmethod
    def from_config(cls, config: BaselineConfig) -> "FeaturePipeline":
        yaku_names = frozenset(
            name for name in feature_names("value") if name.startswith("yaku_")
        )
        shape_names = feature_names("shape")
        lookahead_names = frozenset(
            name for name in feature_names("efficiency") if name.startswith("lookahead_")
        )
        return cls(
            danger=config.danger,
            yaku_config=(
                config.yaku if any(config.weights[name] for name in yaku_names) else None
            ),
            normalization=config.normalization,
            risk_context=config.risk_context,
            shape_mode=config.shape_mode,
            compute_shape=bool(config.group_weights["shape"])
            and any(config.weights[name] for name in shape_names),
            compute_lookahead=bool(config.group_weights["efficiency"])
            and any(config.weights[name] for name in lookahead_names),
        )

    def extract(self, tile_id: int, state: PublicState) -> CandidateFeatures:
        return self.extract_candidates(tile_id, state)[0]

    def preview_efficiency(self, tile_id: int, state: PublicState) -> CandidateFeatures:
        """Cheap first-stage features used before hard shanten pruning."""
        efficiency = extract_efficiency(
            remove_one_tile(state.hand, tile_id), state.visible_counts
        )
        values = efficiency.values()
        return CandidateFeatures(
            values=values,
            normalized_values=normalize_features(values, self.normalization),
            shanten=efficiency.shanten,
            ukeire_kinds=efficiency.ukeire_kinds,
            ukeire_count=efficiency.ukeire_count,
            danger=0.0,
        )

    def extract_candidates(
        self, tile_id: int, state: PublicState
    ) -> tuple[CandidateFeatures, ...]:
        return _extract_candidates(
            tile_id=tile_id,
            state=state,
            danger=self.danger,
            yaku_config=self.yaku_config,
            normalization=self.normalization,
            risk_context=self.risk_context,
            shape_mode=self.shape_mode,
            compute_shape=self.compute_shape,
            compute_lookahead=self.compute_lookahead,
        )


def extract_discard_features(
    tile_id: int,
    state: PublicState,
    danger: dict[str, float],
    yaku_config: YakuConfig | None = None,
    normalization: dict[str, float] | None = None,
    risk_context: dict[str, float] | None = None,
    shape_mode: str = "decomposition",
    compute_shape: bool = True,
    compute_lookahead: bool = False,
) -> CandidateFeatures:
    """Compatibility wrapper for tests and lightweight custom callers."""
    return _extract_candidates(
        tile_id, state, danger, yaku_config, normalization or DEFAULT_NORMALIZATION,
        risk_context, shape_mode, compute_shape, compute_lookahead,
    )[0]


def _extract_candidates(
    tile_id: int,
    state: PublicState,
    danger: dict[str, float],
    yaku_config: YakuConfig | None,
    normalization: dict[str, float],
    risk_context: dict[str, float] | None,
    shape_mode: str,
    compute_shape: bool,
    compute_lookahead: bool,
) -> tuple[CandidateFeatures, ...]:
    remaining_hand = remove_one_tile(state.hand, tile_id)
    efficiency = extract_efficiency(remaining_hand, state.visible_counts)
    common_values = efficiency.values()
    if compute_lookahead:
        common_values.update(
            extract_lookahead(remaining_hand, state, tile_kind(tile_id)).values()
        )

    value_features, yaku_delta = extract_value(
        state.hand, remaining_hand, state, yaku_config
    )
    common_values.update(value_features)
    shape_candidates = (
        extract_shape_candidates(remaining_hand, state.value_honor_kinds, shape_mode)
        if compute_shape else ({},)
    )

    risk = discard_danger(tile_kind(tile_id), state, danger)
    retained_value_tiles = common_values["retained_dora"] + common_values["retained_red"]
    scaled_risk = scale_risk(
        risk, state, efficiency.shanten, retained_value_tiles, risk_context
    )
    candidates = []
    for shape_values in shape_candidates:
        values = {**common_values, **shape_values, "discard_risk": risk}
        normalized = normalize_features(values, normalization)
        normalized["discard_risk"] = scaled_risk
        candidates.append(CandidateFeatures(
            values=values,
            normalized_values=normalized,
            shanten=efficiency.shanten,
            ukeire_kinds=efficiency.ukeire_kinds,
            ukeire_count=efficiency.ukeire_count,
            danger=risk,
            yaku=yaku_delta,
        ))
    return tuple(candidates)
