from riichienv import GameRule, RiichiEnv

from mahjong_ai.baseline.config import default_config_path, load_config
from mahjong_ai.baseline.features import FeaturePipeline, remove_one_tile
from mahjong_ai.baseline.lookahead_features import extract_lookahead
from mahjong_ai.baseline.state import PublicState


def test_lookahead_probabilities_are_bounded_and_publicly_deterministic() -> None:
    env = RiichiEnv(seed=1201, rule=GameRule.default_mjsoul())
    observation = next(iter(env.reset().values()))
    state = PublicState.from_observation(observation)
    discard = observation.legal_actions()[0]
    hand = remove_one_tile(state.hand, discard.tile)

    first = extract_lookahead(hand, state)
    second = extract_lookahead(hand, state)

    assert first == second
    assert 0.0 <= first.win_probability <= 1.0
    assert 0.0 <= first.tenpai_probability <= 1.0
    assert 0.0 <= first.wide_wait_probability <= first.tenpai_probability
    assert 0.0 <= first.furiten_probability <= first.tenpai_probability
    assert first.expected_ukeire >= 0.0


def test_no_lookahead_ablation_skips_search() -> None:
    path = default_config_path().parent / "ablations" / "no-lookahead.yaml"
    pipeline = FeaturePipeline.from_config(load_config(path))

    assert pipeline.compute_lookahead is False
