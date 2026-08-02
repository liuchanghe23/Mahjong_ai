"""Tile-index helpers independent of decision policy."""

from collections.abc import Iterable


def tile_kind(tile_id: int) -> int:
    return tile_id // 4


def tile_rank(kind: int) -> int | None:
    return kind % 9 + 1 if kind < 27 else None


def is_honor(kind: int) -> bool:
    return kind >= 27


def is_red(tile_id: int) -> bool:
    return tile_id in {16, 52, 88}


def counts34(tile_ids: Iterable[int]) -> list[int]:
    counts = [0] * 34
    for tile_id in tile_ids:
        counts[tile_kind(tile_id)] += 1
    return counts


def canonical_tile_id(kind: int) -> int:
    """Return a non-red physical tile id for shanten trial calculations."""
    base = kind * 4
    return base + 1 if kind in {4, 13, 22} else base


def dora_kind(indicator_kind: int) -> int:
    if indicator_kind < 27:
        suit_start = indicator_kind // 9 * 9
        return suit_start + (indicator_kind - suit_start + 1) % 9
    if indicator_kind <= 30:  # winds
        return 27 + (indicator_kind - 27 + 1) % 4
    return 31 + (indicator_kind - 31 + 1) % 3  # dragons

