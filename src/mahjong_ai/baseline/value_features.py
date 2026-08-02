"""Dora and yaku-value feature extraction."""

from mahjong_ai.baseline.config import YakuConfig
from mahjong_ai.baseline.state import PublicState
from mahjong_ai.baseline.tiles import counts34, dora_kind, is_red, tile_kind
from mahjong_ai.baseline.yaku import YakuDelta, YakuPotentialEvaluator


def extract_value(
    original_hand: tuple[int, ...],
    remaining_hand: list[int],
    state: PublicState,
    yaku_config: YakuConfig | None,
) -> tuple[dict[str, float], YakuDelta | None]:
    counts = counts34(remaining_hand)
    dora_kinds = [dora_kind(tile_kind(tile)) for tile in state.dora_indicators]
    values = {
        "retained_dora": float(sum(counts[kind] for kind in dora_kinds)),
        "retained_red": float(sum(is_red(tile) for tile in remaining_hand)),
    }
    yaku_delta = None
    if yaku_config is not None:
        yaku_delta = YakuPotentialEvaluator(yaku_config).compare(
            list(original_hand), remaining_hand, state
        )
        values.update(yaku_delta.values)
    return values, yaku_delta
