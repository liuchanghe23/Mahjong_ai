"""Public-information one-draw Expectimax for effective-tile quality."""

from dataclasses import dataclass
from functools import lru_cache

from mahjong_ai.baseline.efficiency_features import (
    extract_efficiency_counts,
    shanten_from_counts,
)
from mahjong_ai.baseline.state import PublicState
from mahjong_ai.baseline.tiles import counts34, tile_kind


@dataclass(frozen=True)
class LookaheadFeatures:
    win_probability: float
    tenpai_probability: float
    wide_wait_probability: float
    furiten_probability: float
    expected_ukeire: float

    def values(self) -> dict[str, float]:
        return {
            "lookahead_win_probability": self.win_probability,
            "lookahead_tenpai_probability": self.tenpai_probability,
            "lookahead_wide_wait_probability": self.wide_wait_probability,
            "lookahead_furiten_probability": self.furiten_probability,
            "lookahead_expected_ukeire": self.expected_ukeire,
        }


@lru_cache(maxsize=500_000)
def _best_after_draw(
    counts14: tuple[int, ...], visible_after_draw: tuple[int, ...]
) -> tuple[int, int, int, tuple[int, ...]]:
    hands = []
    for discard_kind, count in enumerate(counts14):
        if count <= 0:
            continue
        remaining = list(counts14)
        remaining[discard_kind] -= 1
        remaining_tuple = tuple(remaining)
        hands.append((shanten_from_counts(remaining_tuple), discard_kind, remaining_tuple))
    best_shanten = min(item[0] for item in hands)
    candidates = []
    for _, discard_kind, remaining in hands:
        if shanten_from_counts(remaining) != best_shanten:
            continue
        efficiency = extract_efficiency_counts(remaining, visible_after_draw)
        candidates.append((
            -efficiency.ukeire_count,
            -len(efficiency.ukeire_kinds),
            discard_kind,
            efficiency,
        ))
    _, _, _, best = min(candidates)
    return best.shanten, best.ukeire_count, len(best.ukeire_kinds), best.ukeire_kinds


@lru_cache(maxsize=200_000)
def _extract_lookahead_cached(
    counts13: tuple[int, ...],
    visible_counts: tuple[int, ...],
    own_discard_kinds: frozenset[int],
) -> LookaheadFeatures:
    draw_weights = [max(0, 4 - visible) for visible in visible_counts]
    total_unknown = sum(draw_weights)
    if total_unknown <= 0:
        return LookaheadFeatures(0.0, 0.0, 0.0, 0.0, 0.0)

    base = extract_efficiency_counts(counts13, visible_counts)
    effective_kinds = frozenset(base.ukeire_kinds)
    effective_weight = sum(draw_weights[kind] for kind in effective_kinds)
    non_effective_probability = 1.0 - effective_weight / total_unknown

    # Non-effective draws are represented by the guaranteed tsumogiri branch,
    # which restores the current 13-tile state. This keeps the expectation over
    # all unknown tiles without expanding equivalent low-value branches.
    win = 0.0
    tenpai = non_effective_probability if base.shanten == 0 else 0.0
    wide_now = len(base.ukeire_kinds) >= 2 and base.ukeire_count >= 6
    wide = non_effective_probability if base.shanten == 0 and wide_now else 0.0
    furiten_now = bool(own_discard_kinds.intersection(base.ukeire_kinds))
    furiten = non_effective_probability if base.shanten == 0 and furiten_now else 0.0
    expected_ukeire = non_effective_probability * base.ukeire_count

    for draw_kind in effective_kinds:
        remaining_copies = draw_weights[draw_kind]
        if remaining_copies <= 0:
            continue
        probability = remaining_copies / total_unknown
        counts14 = list(counts13)
        counts14[draw_kind] += 1
        counts14_tuple = tuple(counts14)
        if shanten_from_counts(counts14_tuple) == -1:
            win += probability
            continue

        visible = list(visible_counts)
        visible[draw_kind] += 1
        leaf_shanten, leaf_ukeire, wait_types, waits = _best_after_draw(
            counts14_tuple, tuple(visible)
        )
        expected_ukeire += probability * leaf_ukeire
        if leaf_shanten == 0:
            tenpai += probability
            if wait_types >= 2 and leaf_ukeire >= 6:
                wide += probability
            if own_discard_kinds.intersection(waits):
                furiten += probability

    return LookaheadFeatures(win, tenpai, wide, furiten, expected_ukeire)


def extract_lookahead(
    remaining_hand: list[int], state: PublicState, new_discard_kind: int | None = None
) -> LookaheadFeatures:
    own_discards = {tile_kind(tile) for tile in state.discards[state.player_id]}
    if new_discard_kind is not None:
        own_discards.add(new_discard_kind)
    return _extract_lookahead_cached(
        tuple(counts34(remaining_hand)),
        state.visible_counts,
        frozenset(own_discards),
    )
