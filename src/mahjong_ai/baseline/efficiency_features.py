"""Tile-efficiency feature extraction."""

from dataclasses import dataclass

from riichienv import calculate_shanten

from mahjong_ai.baseline.tiles import canonical_tile_id


@dataclass(frozen=True)
class EfficiencyFeatures:
    shanten: int
    ukeire_kinds: tuple[int, ...]
    ukeire_count: int

    def values(self) -> dict[str, float]:
        return {
            "ukeire_count": float(self.ukeire_count),
            "ukeire_types": float(len(self.ukeire_kinds)),
        }


def extract_efficiency(hand: list[int], visible_counts: tuple[int, ...]) -> EfficiencyFeatures:
    shanten = calculate_shanten(hand)
    kinds: list[int] = []
    remaining = 0
    for kind in range(34):
        if visible_counts[kind] >= 4:
            continue
        if calculate_shanten([*hand, canonical_tile_id(kind)]) < shanten:
            kinds.append(kind)
            remaining += 4 - visible_counts[kind]
    return EfficiencyFeatures(shanten, tuple(kinds), remaining)
