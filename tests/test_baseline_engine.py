from riichienv import ActionType, GameRule, Phase, RiichiEnv, calculate_shanten, parse_hand, parse_tile

from mahjong_ai.baseline import BaselineEngine
from mahjong_ai.baseline.features import extract_discard_features
from mahjong_ai.baseline.state import PublicState
from mahjong_ai.baseline.tiles import counts34
from mahjong_ai.baseline.config import default_config_path, load_config
from mahjong_ai.baseline.tiles import dora_kind
from mahjong_ai.simulator import run_match


def test_dora_indicator_wraps_suits_winds_and_dragons() -> None:
    assert dora_kind(8) == 0
    assert dora_kind(30) == 27
    assert dora_kind(33) == 31


def test_round_and_seat_winds_count_as_value_honor_pairs() -> None:
    hand, _ = parse_hand("11z22z123m123p789s9m")
    state = PublicState(
        player_id=1,
        hand=tuple(hand),
        visible_counts=tuple(counts34(hand)),
        discards=((), (), (), ()),
        dora_indicators=(),
        riichi_declared=(False, False, False, False),
        round_wind=0,
        dealer=0,
    )
    config = load_config(default_config_path())

    features = extract_discard_features(
        parse_tile("9m"), state, config.danger, shape_mode="legacy_overlap"
    )

    assert state.seat_wind == 1
    assert features.values["legacy_value_honor_pair"] == 2


def test_engine_records_ranked_discard_candidates() -> None:
    env = RiichiEnv(seed=11, rule=GameRule.default_mjsoul())
    observation = next(iter(env.reset().values()))
    engine = BaselineEngine()

    action = engine.act(observation)
    decision = engine.last_decision

    assert action.to_mjai() in {candidate.to_mjai() for candidate in observation.legal_actions()}
    assert decision is not None
    assert decision.policy == "linear_discard"
    assert len(decision.candidates) == len(observation.legal_actions())
    assert [(item.shanten, -item.score) for item in decision.candidates] == sorted(
        (item.shanten, -item.score) for item in decision.candidates
    )
    assert decision.selected == decision.candidates[0].action
    assert set(decision.candidates[0].group_contributions) == {
        "efficiency", "value", "shape", "risk"
    }
    assert decision.candidates[0].normalized_features["ukeire_types"] <= 1.0
    assert {item.yaku for item in decision.candidates[0].yaku_before} == {
        "yakuhai", "tanyao", "chiitoitsu", "flush"
    }
    assert {item.yaku for item in decision.candidates[0].yaku_after} == {
        "yakuhai", "tanyao", "chiitoitsu", "flush"
    }


def test_higher_shanten_candidates_are_pruned_before_expensive_features() -> None:
    observation = next(iter(RiichiEnv(seed=1).reset().values()))
    engine = BaselineEngine()

    engine.act(observation)
    decision = engine.last_decision

    assert decision is not None
    pruned = [candidate for candidate in decision.candidates if candidate.pruned]
    assert pruned
    assert all(candidate.prune_reason == "higher_shanten" for candidate in pruned)
    assert all(candidate.shanten > decision.candidates[0].shanten for candidate in pruned)


def test_shanten_has_dominant_weight() -> None:
    env = RiichiEnv(seed=12, rule=GameRule.default_mjsoul())
    observation = next(iter(env.reset().values()))
    engine = BaselineEngine()

    action = engine.act(observation)
    resulting_hand = list(observation.hand)
    resulting_hand.remove(action.tile)
    chosen_shanten = calculate_shanten(resulting_hand)
    all_shanten = []
    for candidate in observation.legal_actions():
        hand = list(observation.hand)
        hand.remove(candidate.tile)
        all_shanten.append(calculate_shanten(hand))

    assert chosen_shanten == min(all_shanten)


def test_late_round_risk_is_larger_after_normalization() -> None:
    hand, _ = parse_hand("123456m234p678s55z")
    base = dict(
        player_id=0,
        hand=tuple(hand),
        visible_counts=tuple(counts34(hand)),
        dora_indicators=(),
        riichi_declared=(False, True, False, False),
        round_wind=0,
        dealer=2,
    )
    early = PublicState(**base, discards=((), (), (), ()))
    late = PublicState(**base, discards=((parse_tile("1z"),) * 13, (), (), ()))
    config = load_config(default_config_path())
    tile = parse_tile("5z")

    early_features = extract_discard_features(
        tile, early, config.danger, config.yaku, config.normalization, config.risk_context
    )
    late_features = extract_discard_features(
        tile, late, config.danger, config.yaku, config.normalization, config.risk_context
    )

    assert late_features.values["discard_risk"] == early_features.values["discard_risk"]
    assert late_features.normalized_values["discard_risk"] > early_features.normalized_values["discard_risk"]


def test_baseline_completes_seeded_match() -> None:
    result = run_match(seed=2026, engine_factory=lambda _player, _seed: BaselineEngine())

    assert result.event_count > 0
    assert sorted(result.ranks) == [1, 2, 3, 4]


def test_riichi_strictly_discards_drawn_physical_tile() -> None:
    env = RiichiEnv(seed=21)
    env.reset()
    base_hand, _ = parse_hand("123m456m789p123s1z")
    drawn = parse_tile("9s")
    hands = env.hands
    hands[0] = [*base_hand, drawn]
    env.hands = hands
    env.current_player = 0
    env.active_players = [0]
    env.phase = Phase.WaitAct
    env.needs_tsumo = False
    env.drawn_tile = drawn
    declared = env.riichi_declared
    declared[0] = True
    env.riichi_declared = declared
    observation = env.get_observations([0])[0]

    action = BaselineEngine().act(observation)

    assert action.action_type == ActionType.DISCARD
    assert action.tile == drawn


def test_riichi_selects_only_wait_preserving_ankan_exposed_by_env() -> None:
    env = RiichiEnv(seed=42)
    env.reset()
    hands = env.hands
    hands[2] = [4, 5, 68, 69, 71, 73, 74, 75, 88, 89, 90, 97, 98, 70]
    hands[2].sort()
    env.hands = hands
    env.current_player = 2
    env.active_players = [2]
    env.phase = Phase.WaitAct
    env.needs_tsumo = False
    env.drawn_tile = 70
    declared = env.riichi_declared
    declared[2] = True
    env.riichi_declared = declared
    observation = env.get_observations([2])[2]

    legal_ankan = [a for a in observation.legal_actions() if a.action_type == ActionType.ANKAN]
    action = BaselineEngine().act(observation)

    assert len(legal_ankan) == 1
    assert action.action_type == ActionType.ANKAN
