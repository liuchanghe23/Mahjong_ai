from pathlib import Path

from mahjong_ai.training.ablation import build_ablation_candidates


ROOT = Path(__file__).parents[1]


def test_group_weight_ablation_builds_baseline_and_one_factor_candidates() -> None:
    spec_path, budget, samples, seed, definitions = build_ablation_candidates(
        ROOT / "configs" / "training" / "group-weight-ablation.yaml"
    )

    assert spec_path.name == "stage1-search.yaml"
    assert (budget, samples, seed) == (300, 2000, 20260810)
    assert len(definitions) == 13
    assert definitions[0][0:2] == ("baseline", None)
    assert definitions[0][2].values == {
        "shape_group": 1.0, "value_group": 1.0, "risk_group": 1.0,
    }
    for axis, level, candidate in definitions[1:]:
        changed = {
            name for name, value in candidate.values.items()
            if value != definitions[0][2].values[name]
        }
        assert changed == {axis}
        assert candidate.values[axis] == level
