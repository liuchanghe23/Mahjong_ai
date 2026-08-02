from pathlib import Path

import pytest

from mahjong_ai.baseline.config import default_config_path, load_config


def test_default_config_is_complete() -> None:
    config = load_config(default_config_path())

    assert config.version == 4
    assert "shanten" not in config.weights
    assert config.weights["ukeire_count"] > 0
    assert set(config.group_weights) == {"efficiency", "value", "shape", "risk"}
    assert all(0 <= value <= 1 for value in config.danger.values())


def test_config_rejects_unknown_weight(tmp_path: Path) -> None:
    raw = default_config_path().read_text(encoding="utf-8")
    path = tmp_path / "invalid.yaml"
    path.write_text(raw.replace("  ukeire_count: 6.0", "  ukeire_count: 6.0\n  typo: 1"), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown=.*typo"):
        load_config(path)


def test_ablation_overlay_only_changes_declared_weights() -> None:
    baseline = load_config(default_config_path())
    ablation = load_config(default_config_path().parent / "ablations" / "no-risk.yaml")

    assert ablation.name == "baseline-no-risk"
    assert ablation.weights["discard_risk"] == 0.0
    assert {
        key: value for key, value in ablation.weights.items() if key != "discard_risk"
    } == {
        key: value for key, value in baseline.weights.items() if key != "discard_risk"
    }
