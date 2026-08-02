"""Strict configuration loading for trainable heuristic weights."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from mahjong_ai.baseline.feature_registry import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    NORMALIZATION_NAMES,
)


REQUIRED_WEIGHTS = set(FEATURE_NAMES)
REQUIRED_GROUPS = set(FEATURE_GROUPS)
REQUIRED_NORMALIZATION = set(NORMALIZATION_NAMES)
REQUIRED_RISK_CONTEXT = {
    "early", "middle", "late", "two_threats", "three_threats",
    "dealer_threat", "self_tenpai", "self_one_shanten", "high_value_hand",
}

REQUIRED_DANGER = {
    "genbutsu",
    "honor_three_visible",
    "honor_two_visible",
    "honor_one_visible",
    "honor_live",
    "terminal",
    "near_terminal",
    "middle",
}


@dataclass(frozen=True)
class PolicyConfig:
    always_win: bool
    always_riichi: bool
    conservative_calls: bool


@dataclass(frozen=True)
class YakuConfig:
    shanten_probability: dict[str, float]
    yakuhai_pair_probability: float
    yakuhai_single_probability: float
    tanyao_forbidden_tile_penalty: float
    flush_off_suit_penalty: float
    flush_honor_penalty_for_chinitsu: float


@dataclass(frozen=True)
class BaselineConfig:
    version: int
    name: str
    policy: PolicyConfig
    weights: dict[str, float]
    danger: dict[str, float]
    yaku: YakuConfig
    group_weights: dict[str, float]
    normalization: dict[str, float]
    risk_context: dict[str, float]
    shape_mode: str


def _strict_numeric_map(raw: Any, required: set[str], section: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"{section} must be a mapping")
    keys = set(raw)
    missing, unknown = required - keys, keys - required
    if missing or unknown:
        raise ValueError(f"Invalid {section} keys; missing={sorted(missing)}, unknown={sorted(unknown)}")
    values = {key: float(value) for key, value in raw.items()}
    if not all(isfinite(value) for value in values.values()):
        raise ValueError(f"{section} values must be finite")
    return values


def load_config(path: Path) -> BaselineConfig:
    with path.open(encoding="utf-8") as input_file:
        raw = yaml.safe_load(input_file)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    # Ablation files are intentionally small overlays.  Keeping the full
    # baseline in one place prevents unrelated parameters drifting between
    # experiments.
    if set(raw) <= {"version", "name", "base", "weight_overrides", "shape_mode"} and "base" in raw:
        if raw.get("version") != 4:
            raise ValueError(f"Unsupported baseline configuration version: {raw.get('version')!r}")
        overrides = raw.get("weight_overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("weight_overrides must be a mapping")
        unknown_weights = set(overrides) - REQUIRED_WEIGHTS
        if unknown_weights:
            raise ValueError(f"Unknown weight overrides: {sorted(unknown_weights)}")
        base_path = (path.parent / str(raw["base"])).resolve()
        base = load_config(base_path)
        weights = dict(base.weights)
        weights.update({key: float(value) for key, value in overrides.items()})
        if not all(isfinite(value) for value in weights.values()):
            raise ValueError("weight_overrides values must be finite")
        return BaselineConfig(
            version=4,
            name=str(raw.get("name", path.stem)),
            policy=base.policy,
            weights=weights,
            danger=base.danger,
            yaku=base.yaku,
            group_weights=base.group_weights,
            normalization=base.normalization,
            risk_context=base.risk_context,
            shape_mode=str(raw.get("shape_mode", base.shape_mode)),
        )

    allowed = {
        "version", "name", "policy", "weights", "danger", "yaku",
        "group_weights", "normalization", "risk_context",
        "shape_mode",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    if raw.get("version") != 4:
        raise ValueError(f"Unsupported baseline configuration version: {raw.get('version')!r}")

    policy_raw = raw.get("policy")
    if not isinstance(policy_raw, dict):
        raise ValueError("policy must be a mapping")
    policy_keys = {"always_win", "always_riichi", "conservative_calls"}
    if set(policy_raw) != policy_keys:
        raise ValueError("policy must contain exactly always_win, always_riichi, conservative_calls")

    danger = _strict_numeric_map(raw.get("danger"), REQUIRED_DANGER, "danger")
    if any(not 0.0 <= value <= 1.0 for value in danger.values()):
        raise ValueError("danger values must be between 0 and 1")

    yaku_raw = raw.get("yaku")
    if not isinstance(yaku_raw, dict) or set(yaku_raw) != {
        "shanten_probability", "yakuhai", "tanyao", "flush"
    }:
        raise ValueError("yaku must contain exactly shanten_probability, yakuhai, tanyao, flush")
    shanten_probability = _strict_numeric_map(
        yaku_raw["shanten_probability"], {"ready", "one", "two", "three", "far"}, "yaku.shanten_probability"
    )
    yakuhai = _strict_numeric_map(
        yaku_raw["yakuhai"], {"pair_probability", "single_probability"}, "yaku.yakuhai"
    )
    tanyao = _strict_numeric_map(
        yaku_raw["tanyao"], {"forbidden_tile_penalty"}, "yaku.tanyao"
    )
    flush = _strict_numeric_map(
        yaku_raw["flush"], {"off_suit_penalty", "honor_penalty_for_chinitsu"}, "yaku.flush"
    )
    probabilities = [*shanten_probability.values(), *yakuhai.values()]
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("yaku probability values must be between 0 and 1")

    group_weights = _strict_numeric_map(raw.get("group_weights"), REQUIRED_GROUPS, "group_weights")
    normalization = _strict_numeric_map(raw.get("normalization"), REQUIRED_NORMALIZATION, "normalization")
    risk_context = _strict_numeric_map(raw.get("risk_context"), REQUIRED_RISK_CONTEXT, "risk_context")
    if any(value <= 0 for value in normalization.values()):
        raise ValueError("normalization values must be positive")
    if any(value <= 0 for value in risk_context.values()):
        raise ValueError("risk_context values must be positive")
    shape_mode = str(raw.get("shape_mode", "decomposition"))
    if shape_mode not in {"decomposition", "legacy_overlap"}:
        raise ValueError("shape_mode must be decomposition or legacy_overlap")

    return BaselineConfig(
        version=4,
        name=str(raw.get("name", "baseline")),
        policy=PolicyConfig(**{key: bool(policy_raw[key]) for key in policy_keys}),
        weights=_strict_numeric_map(raw.get("weights"), REQUIRED_WEIGHTS, "weights"),
        danger=danger,
        yaku=YakuConfig(
            shanten_probability=shanten_probability,
            yakuhai_pair_probability=yakuhai["pair_probability"],
            yakuhai_single_probability=yakuhai["single_probability"],
            tanyao_forbidden_tile_penalty=tanyao["forbidden_tile_penalty"],
            flush_off_suit_penalty=flush["off_suit_penalty"],
            flush_honor_penalty_for_chinitsu=flush["honor_penalty_for_chinitsu"],
        ),
        group_weights=group_weights,
        normalization=normalization,
        risk_context=risk_context,
        shape_mode=shape_mode,
    )


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "baseline.yaml"
