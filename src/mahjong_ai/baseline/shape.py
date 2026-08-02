"""Value-free enumeration of non-overlapping whole-hand structures."""

from dataclasses import dataclass
from functools import lru_cache

from mahjong_ai.baseline.tiles import is_honor, tile_rank


@dataclass(frozen=True)
class ShapeFeatures:
    complete_meld: int = 0
    ryanmen_taatsu: int = 0
    kanchan_taatsu: int = 0
    penchan_taatsu: int = 0
    head_pair: int = 0
    extra_pair: int = 0
    unused_middle: int = 0
    unused_near_terminal: int = 0
    unused_terminal: int = 0
    unused_honor: int = 0

    def values(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()}


# meld, pair, ryanmen, kanchan, penchan, and four unused categories.
Signature = tuple[int, int, int, int, int, int, int, int, int]


def _add(signature: Signature, index: int) -> Signature:
    values = list(signature)
    values[index] += 1
    return tuple(values)  # type: ignore[return-value]


def _unused_index(kind: int) -> int:
    if is_honor(kind):
        return 8
    rank = tile_rank(kind)
    if rank in {1, 9}:
        return 7
    if rank in {2, 8}:
        return 6
    return 5


@lru_cache(maxsize=100_000)
def _enumerate(counts: tuple[int, ...]) -> tuple[Signature, ...]:
    """Enumerate unique structural signatures without assigning value."""
    try:
        kind = next(index for index, count in enumerate(counts) if count)
    except StopIteration:
        return ((0, 0, 0, 0, 0, 0, 0, 0, 0),)

    branches: list[tuple[tuple[int, ...], int]] = []

    def remove(kinds: tuple[int, ...], feature_index: int) -> None:
        remaining = list(counts)
        for tile in kinds:
            if remaining[tile] <= 0:
                return
            remaining[tile] -= 1
        branches.append((tuple(remaining), feature_index))

    if counts[kind] >= 3:
        remove((kind, kind, kind), 0)
    if kind < 27 and kind % 9 <= 6:
        remove((kind, kind + 1, kind + 2), 0)
    if counts[kind] >= 2:
        remove((kind, kind), 1)
    if kind < 27:
        rank = kind % 9
        if rank <= 7:
            remove((kind, kind + 1), 4 if rank in {0, 7} else 2)
        if rank <= 6:
            remove((kind, kind + 2), 3)
    remove((kind,), _unused_index(kind))

    signatures = {
        _add(child, feature_index)
        for remaining, feature_index in branches
        for child in _enumerate(remaining)
    }
    return tuple(sorted(signatures))


def enumerate_shape_features(counts: tuple[int, ...]) -> tuple[ShapeFeatures, ...]:
    """Return every unique disjoint feature vector in deterministic order."""
    candidates = []
    for melds, pairs, ryanmen, kanchan, penchan, middle, near, terminal, honor in _enumerate(counts):
        candidates.append(
            ShapeFeatures(
                complete_meld=melds,
                ryanmen_taatsu=ryanmen,
                kanchan_taatsu=kanchan,
                penchan_taatsu=penchan,
                head_pair=int(pairs > 0),
                extra_pair=max(0, pairs - 1),
                unused_middle=middle,
                unused_near_terminal=near,
                unused_terminal=terminal,
                unused_honor=honor,
            )
        )
    return tuple(sorted(set(candidates), key=lambda item: tuple(item.__dict__.values())))


def decompose_shape(counts: tuple[int, ...]) -> ShapeFeatures:
    """Compatibility helper returning a deterministic structure-first candidate."""
    return max(
        enumerate_shape_features(counts),
        key=lambda item: (
            item.complete_meld,
            item.head_pair,
            item.ryanmen_taatsu,
            item.extra_pair,
            item.kanchan_taatsu,
            item.penchan_taatsu,
            -sum((item.unused_middle, item.unused_near_terminal, item.unused_terminal, item.unused_honor)),
        ),
    )
