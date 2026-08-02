from riichienv import parse_hand, parse_tile

from mahjong_ai.baseline.config import default_config_path, load_config
from mahjong_ai.baseline.state import PublicState
from mahjong_ai.baseline.tiles import counts34
from mahjong_ai.baseline.yaku import YakuPotentialEvaluator, YakuState


def make_state(hand: list[int], *, player_id: int = 0, dealer: int = 0, round_wind: int = 0) -> PublicState:
    return PublicState(
        player_id=player_id,
        hand=tuple(hand),
        visible_counts=tuple(counts34(hand)),
        discards=((), (), (), ()),
        dora_indicators=(),
        riichi_declared=(False, False, False, False),
        round_wind=round_wind,
        dealer=dealer,
    )


def evaluator() -> YakuPotentialEvaluator:
    return YakuPotentialEvaluator(load_config(default_config_path()).yaku)


def by_name(items):
    return {item.yaku: item for item in items}


def test_breaking_double_wind_pair_loses_yakuhai_value() -> None:
    hand, _ = parse_hand("11z123m456m789p123s")
    state = make_state(hand, player_id=0, dealer=0, round_wind=0)
    after = list(hand)
    after.remove(parse_tile("1z"))

    delta = evaluator().compare(hand, after, state)
    before = by_name(delta.before)["yakuhai"]

    assert state.value_honor_han(27) == 2
    assert before.expected_han == 1.1
    assert delta.values["yaku_yakuhai_delta"] < 0


def test_discarding_last_terminal_improves_tanyao_potential() -> None:
    hand, _ = parse_hand("234m456m678p345s1m1p")
    state = make_state(hand)
    after = list(hand)
    after.remove(parse_tile("1m"))

    delta = evaluator().compare(hand, after, state)

    assert delta.values["yaku_tanyao_delta"] > 0
    assert by_name(delta.after)["tanyao"].probability > by_name(delta.before)["tanyao"].probability


def test_breaking_pair_reduces_chiitoitsu_potential() -> None:
    hand, _ = parse_hand("112233m445566p7s8s")
    state = make_state(hand)
    after = list(hand)
    after.remove(parse_tile("1m"))

    delta = evaluator().compare(hand, after, state)

    assert delta.values["yaku_chiitoitsu_delta"] < 0


def test_discarding_off_suit_tile_improves_flush_route() -> None:
    hand, _ = parse_hand("123456789m11z2p3p4s")
    state = make_state(hand)
    after = list(hand)
    after.remove(parse_tile("4s"))

    delta = evaluator().compare(hand, after, state)
    flush = by_name(delta.after)["flush"]

    assert delta.values["yaku_flush_delta"] > 0
    assert flush.state in {YakuState.POSSIBLE, YakuState.LIKELY, YakuState.GUARANTEED}


def test_open_terminal_meld_makes_tanyao_impossible() -> None:
    hand, _ = parse_hand("234m456m678p345s1p1p")
    state = PublicState(
        **{**make_state(hand).__dict__, "self_melds": ((0, 1, 2),)}
    )

    result = by_name(evaluator().evaluate(hand, state))["tanyao"]

    assert result.state == YakuState.IMPOSSIBLE
    assert result.potential == 0

