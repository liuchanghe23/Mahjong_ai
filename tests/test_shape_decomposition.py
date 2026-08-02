from mahjong_ai.baseline.shape import decompose_shape


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
    assert shape.required_ryanmen == 1
    assert shape.required_kanchan == 0
    assert shape.required_penchan == 0


def test_complete_sequence_is_not_also_counted_as_taatsu() -> None:
    shape = decompose_shape(counts_for(0, 1, 2))

    assert shape.complete_meld == 1
    assert shape.required_ryanmen == 0
    assert shape.required_kanchan == 0
    assert shape.required_penchan == 0


def test_unused_tiles_are_classified_inside_the_decomposition() -> None:
    shape = decompose_shape(counts_for(0, 10, 22, 27))

    assert shape.unused_terminal == 1
    assert shape.unused_near_terminal == 1
    assert shape.unused_middle == 1
    assert shape.unused_honor == 1
