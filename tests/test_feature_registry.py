from riichienv import GameRule, RiichiEnv

from mahjong_ai.baseline.config import default_config_path, load_config
from mahjong_ai.baseline.feature_registry import FEATURE_BY_NAME, FEATURE_GROUPS, feature_names
from mahjong_ai.baseline.features import FeaturePipeline
from mahjong_ai.baseline.efficiency_features import improving_kinds
from mahjong_ai.baseline.state import PublicState


def test_registry_is_the_single_source_for_configured_weights() -> None:
    config = load_config(default_config_path())

    assert set(config.weights) == set(FEATURE_BY_NAME)
    assert set(config.group_weights) == set(FEATURE_GROUPS)
    assert set().union(*(feature_names(group) for group in FEATURE_GROUPS)) == set(FEATURE_BY_NAME)


def test_pipeline_only_emits_registered_features() -> None:
    env = RiichiEnv(seed=901, rule=GameRule.default_mjsoul())
    observation = next(iter(env.reset().values()))
    state = PublicState.from_observation(observation)
    config = load_config(default_config_path())
    pipeline = FeaturePipeline.from_config(config)
    discard = next(action for action in observation.legal_actions() if action.tile is not None)

    features = pipeline.extract(discard.tile, state)

    assert set(features.values) <= set(FEATURE_BY_NAME)
    assert set(features.normalized_values) == set(features.values)


def test_no_shape_configuration_disables_shape_extractor() -> None:
    path = default_config_path().parent / "ablations" / "no-shape.yaml"
    pipeline = FeaturePipeline.from_config(load_config(path))

    assert pipeline.compute_shape is False


def test_improving_kinds_are_cached_by_hand_structure() -> None:
    counts = (1, 1, 1, 0, 0, 0, 0, 0, 0) + (0,) * 25
    improving_kinds.cache_clear()

    first = improving_kinds(counts)
    second = improving_kinds(counts)

    assert second == first
    assert improving_kinds.cache_info().hits == 1
