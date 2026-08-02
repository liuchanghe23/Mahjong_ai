"""Non-overlapping whole-hand shape decomposition."""

from dataclasses import dataclass
from functools import lru_cache

from mahjong_ai.baseline.tiles import is_honor, tile_rank


@dataclass(frozen=True)
class ShapeFeatures:
    complete_meld: int = 0
    required_ryanmen: int = 0
    required_kanchan: int = 0
    required_penchan: int = 0
    head_pair: int = 0
    extra_pair: int = 0
    unused_middle: int = 0
    unused_near_terminal: int = 0
    unused_terminal: int = 0
    unused_honor: int = 0
    shape_flexibility: float = 0.0

    def values(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()}


# A group stores its structural type and the exact 34-kind tiles it consumes.
Group = tuple[str, tuple[int, ...]]
Decomposition = tuple[Group, ...]


def _unused_categories(kinds: list[int]) -> tuple[int, int, int, int]:
    middle = near = terminal = honor = 0
    for kind in kinds:
        if is_honor(kind):
            honor += 1
        elif tile_rank(kind) in {1, 9}:
            terminal += 1
        elif tile_rank(kind) in {2, 8}:
            near += 1
        else:
            middle += 1
    return middle, near, terminal, honor


def _project(groups: Decomposition) -> tuple[ShapeFeatures, float]:
    melds = [group for group in groups if group[0] == "meld"]
    pairs = [group for group in groups if group[0] == "pair"]
    by_type = {
        name: [group for group in groups if group[0] == name]
        for name in ("ryanmen", "kanchan", "penchan")
    }
    singles = [kind for name, kinds in groups if name == "single" for kind in kinds]

    head = pairs[:1]
    extra_pairs = pairs[1:]
    slots = max(0, 4 - len(melds))
    selected_pairs = extra_pairs[:slots]
    slots -= len(selected_pairs)
    selected_ryanmen = by_type["ryanmen"][:slots]
    slots -= len(selected_ryanmen)
    selected_kanchan = by_type["kanchan"][:slots]
    slots -= len(selected_kanchan)
    selected_penchan = by_type["penchan"][:slots]

    selected_groups = head + selected_pairs + selected_ryanmen + selected_kanchan + selected_penchan + melds
    unselected_groups = list(groups)
    for group in selected_groups:
        unselected_groups.remove(group)
    unused_kinds = [kind for group in unselected_groups for kind in group[1]]
    unused_middle, unused_near, unused_terminal, unused_honor = _unused_categories(unused_kinds)
    unused_penalty = (
        unused_middle * 1.0 + unused_near * 1.3
        + unused_terminal * 1.6 + unused_honor * 1.8
    )
    score = (
        len(melds) * 100.0 + bool(head) * 15.0 + len(selected_pairs) * 9.0
        + len(selected_ryanmen) * 12.0 + len(selected_kanchan) * 7.0
        + len(selected_penchan) * 5.0 - unused_penalty * 2.0
    )
    return ShapeFeatures(
        complete_meld=len(melds),
        required_ryanmen=len(selected_ryanmen),
        required_kanchan=len(selected_kanchan),
        required_penchan=len(selected_penchan),
        head_pair=int(bool(head)),
        extra_pair=len(selected_pairs),
        unused_middle=unused_middle,
        unused_near_terminal=unused_near,
        unused_terminal=unused_terminal,
        unused_honor=unused_honor,
    ), score


@lru_cache(maxsize=100_000)
def _enumerate(counts: tuple[int, ...]) -> tuple[Decomposition, ...]:
    try:
        kind = next(index for index, count in enumerate(counts) if count)
    except StopIteration:
        return ((),)

    branches: list[tuple[tuple[int, ...], Group]] = []

    def remove(kinds: tuple[int, ...], group_type: str) -> None:
        remaining = list(counts)
        for tile in kinds:
            if remaining[tile] <= 0:
                return
            remaining[tile] -= 1
        branches.append((tuple(remaining), (group_type, kinds)))

    if counts[kind] >= 3:
        remove((kind, kind, kind), "meld")
    if kind < 27 and kind % 9 <= 6:
        remove((kind, kind + 1, kind + 2), "meld")
    if counts[kind] >= 2:
        remove((kind, kind), "pair")
    if kind < 27:
        rank = kind % 9
        if rank <= 7:
            remove((kind, kind + 1), "penchan" if rank in {0, 7} else "ryanmen")
        if rank <= 6:
            remove((kind, kind + 2), "kanchan")
    remove((kind,), "single")

    decompositions = {
        (group, *child)
        for remaining, group in branches
        for child in _enumerate(remaining)
    }
    ranked = sorted(decompositions, key=lambda item: _project(item)[1], reverse=True)
    # A small beam is sufficient for a 13/14-tile heuristic evaluator and
    # avoids spending simulation time on structurally poor decompositions.
    return tuple(ranked[:24])


def decompose_shape(counts: tuple[int, ...]) -> ShapeFeatures:
    """Return the best disjoint decomposition plus alternate-route flexibility."""
    projected: dict[tuple[int, ...], tuple[ShapeFeatures, float]] = {}
    for decomposition in _enumerate(counts):
        features, score = _project(decomposition)
        key = tuple(
            int(value) for name, value in features.__dict__.items()
            if name != "shape_flexibility"
        )
        projected[key] = (features, score)
    ranked = sorted(projected.values(), key=lambda item: item[1], reverse=True)
    best, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    flexibility = max(0.0, second_score / best_score) if best_score > 0 else 0.0
    return ShapeFeatures(**{**best.__dict__, "shape_flexibility": flexibility})
