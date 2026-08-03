from pathlib import Path

import pytest

from mahjong_ai.training.validation import prepare_parameter_candidate


ROOT = Path(__file__).parents[1]
SPEC = ROOT / "configs" / "training" / "stage1-search.yaml"


def test_prepare_value_candidate_uses_independent_selection_seeds() -> None:
    spec, candidate, seeds = prepare_parameter_candidate(
        SPEC, "value_group", 0.5, "selection"
    )

    assert candidate.group_weights == {
        "efficiency": 1.0, "shape": 1.0, "value": 0.5, "risk": 1.0,
    }
    assert seeds.base_seed == 110000
    assert seeds.matches == 1002
    assert spec.seed_sets["train"].stop_seed < seeds.base_seed


def test_prepare_parameter_candidate_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="Unknown trainable parameter"):
        prepare_parameter_candidate(SPEC, "missing", 1.0, "selection")
