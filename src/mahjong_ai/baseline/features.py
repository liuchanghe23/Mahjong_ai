"""Pure feature extraction for discard candidates."""

from dataclasses import dataclass

from riichienv import calculate_shanten

from mahjong_ai.baseline.config import YakuConfig
from mahjong_ai.baseline.state import PublicState
from mahjong_ai.baseline.tiles import (
    canonical_tile_id,
    counts34,
    dora_kind,
    is_honor,
    is_red,
    tile_kind,
    tile_rank,
)
from mahjong_ai.baseline.yaku import YakuDelta, YakuPotentialEvaluator
from mahjong_ai.baseline.shape import decompose_shape


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
    result = list(hand)
    result.remove(tile_id)
    return result


def _ukeire(hand: list[int], visible_counts: tuple[int, ...]) -> tuple[tuple[int, ...], int, int]:
    shanten = calculate_shanten(hand)
    kinds: list[int] = []
    remaining = 0
    for kind in range(34):
        if visible_counts[kind] >= 4:
            continue
        if calculate_shanten([*hand, canonical_tile_id(kind)]) < shanten:
            kinds.append(kind)
            remaining += 4 - visible_counts[kind]
    return tuple(kinds), remaining, shanten


def _shape_features(hand: list[int], value_honor_kinds: frozenset[int]) -> dict[str, float]:
    counts = counts34(hand)
    ryanmen = kanchan = penchan = 0
    for suit_start in (0, 9, 18):
        for rank in range(1, 9):
            left = suit_start + rank - 1
            right = left + 1
            if counts[left] and counts[right]:
                if rank in {1, 8}:
                    penchan += 1
                else:
                    ryanmen += 1
        for rank in range(1, 8):
            left = suit_start + rank - 1
            right = left + 2
            if counts[left] and counts[right]:
                kanchan += 1

    pairs = sum(count >= 2 for count in counts)
    value_honor_pairs = sum(counts[kind] >= 2 for kind in value_honor_kinds)
    return {
        "complete_meld": 0.0,
        "required_ryanmen": float(ryanmen),
        "required_kanchan": float(kanchan),
        "required_penchan": float(penchan),
        "head_pair": float(pairs),
        "extra_pair": 0.0,
        "unused_middle": 0.0,
        "unused_near_terminal": 0.0,
        "unused_terminal": 0.0,
        "unused_honor": 0.0,
        "shape_flexibility": 0.0,
        "legacy_value_honor_pair": float(value_honor_pairs),
    }


def discard_danger(kind: int, state: PublicState, danger: dict[str, float]) -> float:
    threats = [
        opponent
        for opponent, declared in enumerate(state.riichi_declared)
        if opponent != state.player_id and declared
    ]
    if not threats:
        return 0.0
    if state.is_genbutsu(kind):
        return danger["genbutsu"]
    if is_honor(kind):
        visible = state.visible_counts[kind]
        key = {
            3: "honor_three_visible",
            2: "honor_two_visible",
            1: "honor_one_visible",
        }.get(visible, "honor_live")
        return danger[key]
    rank = tile_rank(kind)
    if rank in {1, 9}:
        return danger["terminal"]
    if rank in {2, 8}:
        return danger["near_terminal"]
    return danger["middle"]


def extract_discard_features(
    tile_id: int,
    state: PublicState,
    danger: dict[str, float],
    yaku_config: YakuConfig | None = None,
    normalization: dict[str, float] | None = None,
    risk_context: dict[str, float] | None = None,
    shape_mode: str = "decomposition",
    compute_shape: bool = True,
) -> CandidateFeatures:
    remaining_hand = remove_one_tile(state.hand, tile_id)
    ukeire_kinds, ukeire_count, shanten = _ukeire(remaining_hand, state.visible_counts)
    counts = counts34(remaining_hand)
    indicator_kinds = [tile_kind(tile) for tile in state.dora_indicators]
    dora_kinds = [dora_kind(kind) for kind in indicator_kinds]

    values: dict[str, float] = {
        "ukeire_count": float(ukeire_count),
        "ukeire_types": float(len(ukeire_kinds)),
        "retained_dora": float(sum(counts[kind] for kind in dora_kinds)),
        "retained_red": float(sum(is_red(tile) for tile in remaining_hand)),
    }
    if compute_shape:
        if shape_mode == "legacy_overlap":
            values.update(_shape_features(remaining_hand, state.value_honor_kinds))
        else:
            values.update(decompose_shape(tuple(counts)).values())
            values["legacy_value_honor_pair"] = 0.0
    risk = discard_danger(tile_kind(tile_id), state, danger)
    values["discard_risk"] = risk
    yaku_delta = None
    if yaku_config is not None:
        yaku_delta = YakuPotentialEvaluator(yaku_config).compare(list(state.hand), remaining_hand, state)
        values.update(yaku_delta.values)

    scales = normalization or {
        "ukeire_count": 40.0, "ukeire_types": 34.0,
        "hand_tiles": 13.0, "yaku_potential": 2.0,
    }
    normalized = dict(values)
    normalized["ukeire_count"] /= scales["ukeire_count"]
    normalized["ukeire_types"] /= scales["ukeire_types"]
    count_features = {
        "complete_meld", "required_ryanmen", "required_kanchan",
        "required_penchan", "head_pair", "extra_pair", "unused_middle",
        "unused_near_terminal", "unused_terminal", "unused_honor",
        "legacy_value_honor_pair",
        "retained_dora", "retained_red",
    }
    for name in count_features & normalized.keys():
        normalized[name] /= scales["hand_tiles"]
    for name in {
        "yaku_yakuhai_delta", "yaku_tanyao_delta",
        "yaku_chiitoitsu_delta", "yaku_flush_delta",
    } & normalized.keys():
        normalized[name] /= scales["yaku_potential"]

    context_multiplier = 1.0
    if risk_context is not None and risk > 0:
        turn = len(state.discards[state.player_id]) + 1
        stage = "early" if turn <= 6 else "middle" if turn <= 12 else "late"
        context_multiplier *= risk_context[stage]
        threats = [
            pid for pid, declared in enumerate(state.riichi_declared)
            if pid != state.player_id and declared
        ]
        if len(threats) == 2:
            context_multiplier *= risk_context["two_threats"]
        elif len(threats) >= 3:
            context_multiplier *= risk_context["three_threats"]
        if state.dealer in threats:
            context_multiplier *= risk_context["dealer_threat"]
        if shanten == 0:
            context_multiplier *= risk_context["self_tenpai"]
        elif shanten == 1:
            context_multiplier *= risk_context["self_one_shanten"]
        if values["retained_dora"] + values["retained_red"] >= 2:
            context_multiplier *= risk_context["high_value_hand"]
    normalized["discard_risk"] *= context_multiplier
    return CandidateFeatures(values, normalized, shanten, ukeire_kinds, ukeire_count, risk, yaku_delta)
