from pathlib import Path

from mahjong_ai.training.search import sample_candidates
from mahjong_ai.training.schema import load_training_spec


def spec_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "training" / "stage1-search.yaml"


def test_candidate_sampling_is_deterministic_and_starts_at_baseline() -> None:
    first = sample_candidates(spec_path(), 5, 77)
    second = sample_candidates(spec_path(), 5, 77)
    spec = load_training_spec(spec_path())

    assert first == second
    assert first[0].values == spec.initial_values
    assert len({candidate.candidate_id for candidate in first}) == 5


def test_sampled_candidates_respect_parameter_bounds() -> None:
    spec = load_training_spec(spec_path())

    for candidate in sample_candidates(spec_path(), 20, 88):
        for name, value in candidate.values.items():
            parameter = spec.parameters[name]
            assert parameter.minimum <= value <= parameter.maximum
