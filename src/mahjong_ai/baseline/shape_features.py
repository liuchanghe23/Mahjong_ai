"""Shape feature extraction, including the isolated legacy evaluator."""

from mahjong_ai.baseline.shape import enumerate_shape_features
from mahjong_ai.baseline.tiles import counts34


def _legacy_overlap(hand: list[int], value_honor_kinds: frozenset[int]) -> dict[str, float]:
    counts = counts34(hand)
    ryanmen = kanchan = penchan = 0
    for suit_start in (0, 9, 18):
        for rank in range(1, 9):
            left = suit_start + rank - 1
            if counts[left] and counts[left + 1]:
                if rank in {1, 8}:
                    penchan += 1
                else:
                    ryanmen += 1
        for rank in range(1, 8):
            left = suit_start + rank - 1
            if counts[left] and counts[left + 2]:
                kanchan += 1
    return {
        "complete_meld": 0.0,
        "ryanmen_taatsu": float(ryanmen),
        "kanchan_taatsu": float(kanchan),
        "penchan_taatsu": float(penchan),
        "head_pair": float(sum(count >= 2 for count in counts)),
        "extra_pair": 0.0,
        "unused_middle": 0.0,
        "unused_near_terminal": 0.0,
        "unused_terminal": 0.0,
        "unused_honor": 0.0,
        "legacy_value_honor_pair": float(
            sum(counts[kind] >= 2 for kind in value_honor_kinds)
        ),
    }


def extract_shape_candidates(
    hand: list[int], value_honor_kinds: frozenset[int], mode: str
) -> tuple[dict[str, float], ...]:
    if mode == "legacy_overlap":
        return (_legacy_overlap(hand, value_honor_kinds),)
    candidates = []
    for shape in enumerate_shape_features(tuple(counts34(hand))):
        values = shape.values()
        values["legacy_value_honor_pair"] = 0.0
        candidates.append(values)
    return tuple(candidates)
