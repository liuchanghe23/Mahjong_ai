"""Cached tile-efficiency feature extraction."""

from dataclasses import dataclass
from functools import lru_cache

from riichienv import calculate_shanten

from mahjong_ai.baseline.tiles import canonical_tile_id, counts34


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


def _hand_from_counts(counts: tuple[int, ...]) -> list[int]:
    return [canonical_tile_id(kind) for kind, count in enumerate(counts) for _ in range(count)]


@lru_cache(maxsize=500_000)
def shanten_from_counts(counts: tuple[int, ...]) -> int:
    return calculate_shanten(_hand_from_counts(counts))


@lru_cache(maxsize=500_000)
def extract_efficiency_counts(
    counts: tuple[int, ...], visible_counts: tuple[int, ...]
) -> EfficiencyFeatures:
    shanten = shanten_from_counts(counts)
    kinds: list[int] = []
    remaining = 0
    for kind in range(34):
        if visible_counts[kind] >= 4:
            continue
        added = list(counts)
        added[kind] += 1
        if shanten_from_counts(tuple(added)) < shanten:
            kinds.append(kind)
            remaining += 4 - visible_counts[kind]
    return EfficiencyFeatures(shanten, tuple(kinds), remaining)


def extract_efficiency(hand: list[int], visible_counts: tuple[int, ...]) -> EfficiencyFeatures:
    return extract_efficiency_counts(tuple(counts34(hand)), visible_counts)
