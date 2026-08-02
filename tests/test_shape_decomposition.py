from dataclasses import replace

from mahjong_ai.baseline.config import default_config_path, load_config
from mahjong_ai.baseline.feature_registry import normalize_features
from mahjong_ai.baseline.features import CandidateFeatures
from mahjong_ai.baseline.scoring import select_best_variant
from mahjong_ai.baseline.shape import _enumerate, decompose_shape
from mahjong_ai.baseline.shape import enumerate_shape_features


def counts_for(*kinds: int) -> tuple[int, ...]:
    counts = [0] * 34
    for kind in kinds:
        counts[kind] += 1
    return tuple(counts)


def test_composite_shape_never_reuses_a_tile() -> None:
    # 23456m can be one sequence plus one taatsu, not three simultaneous
    # adjacent taatsu and two simultaneous kanchan.
    shape = decompose_shape(counts_for(1, 2, 3, 4, 5))

    assert shape.complete_meld == 1
    assert shape.ryanmen_taatsu == 1
    assert shape.kanchan_taatsu == 0
    assert shape.penchan_taatsu == 0


def test_complete_sequence_is_not_also_counted_as_taatsu() -> None:
    shape = decompose_shape(counts_for(0, 1, 2))

    assert shape.complete_meld == 1
    assert shape.ryanmen_taatsu == 0
    assert shape.kanchan_taatsu == 0
    assert shape.penchan_taatsu == 0


def test_unused_tiles_are_classified_inside_the_decomposition() -> None:
    shape = decompose_shape(counts_for(0, 10, 22, 27))

    assert shape.unused_terminal == 1
    assert shape.unused_near_terminal == 1
    assert shape.unused_middle == 1
    assert shape.unused_honor == 1


def test_equal_score_decompositions_have_a_deterministic_order() -> None:
    counts = counts_for(1, 2, 3, 4, 5)
    first = _enumerate(counts)
    _enumerate.cache_clear()

    assert _enumerate(counts) == first


def test_trainable_weights_choose_the_structural_decomposition() -> None:
    config = load_config(default_config_path())
    candidates = tuple(
        CandidateFeatures(
            values=shape.values(),
            normalized_values=normalize_features(shape.values(), config.normalization),
            shanten=0,
            ukeire_kinds=(),
            ukeire_count=0,
            danger=0.0,
        )
        for shape in enumerate_shape_features(counts_for(0, 1, 2))
    )
    meld_weights = {name: 0.0 for name in config.weights}
    meld_weights["complete_meld"] = 10.0
    taatsu_weights = dict(meld_weights)
    taatsu_weights["complete_meld"] = 0.0
    taatsu_weights["penchan_taatsu"] = 10.0

    meld = select_best_variant(candidates, replace(config, weights=meld_weights))[2]
    taatsu = select_best_variant(candidates, replace(config, weights=taatsu_weights))[2]

    assert meld.values["complete_meld"] == 1
    assert taatsu.values["penchan_taatsu"] == 1
