"""Discard danger and state-dependent risk scaling."""

from mahjong_ai.baseline.state import PublicState
from mahjong_ai.baseline.tiles import is_honor, tile_rank


def discard_danger(kind: int, state: PublicState, danger: dict[str, float]) -> float:
    threats = [
        opponent for opponent, declared in enumerate(state.riichi_declared)
        if opponent != state.player_id and declared
    ]
    if not threats:
        return 0.0
    if state.is_genbutsu(kind):
        return danger["genbutsu"]
    if is_honor(kind):
        visible = state.visible_counts[kind]
        key = {3: "honor_three_visible", 2: "honor_two_visible", 1: "honor_one_visible"}.get(
            visible, "honor_live"
        )
        return danger[key]
    rank = tile_rank(kind)
    if rank in {1, 9}:
        return danger["terminal"]
    if rank in {2, 8}:
        return danger["near_terminal"]
    return danger["middle"]


def scale_risk(
    risk: float,
    state: PublicState,
    shanten: int,
    retained_value_tiles: float,
    context: dict[str, float] | None,
) -> float:
    if context is None or risk <= 0:
        return risk
    turn = len(state.discards[state.player_id]) + 1
    stage = "early" if turn <= 6 else "middle" if turn <= 12 else "late"
    multiplier = context[stage]
    threats = [
        pid for pid, declared in enumerate(state.riichi_declared)
        if pid != state.player_id and declared
    ]
    if len(threats) == 2:
        multiplier *= context["two_threats"]
    elif len(threats) >= 3:
        multiplier *= context["three_threats"]
    if state.dealer in threats:
        multiplier *= context["dealer_threat"]
    if shanten == 0:
        multiplier *= context["self_tenpai"]
    elif shanten == 1:
        multiplier *= context["self_one_shanten"]
    if retained_value_tiles >= 2:
        multiplier *= context["high_value_hand"]
    return risk * multiplier
