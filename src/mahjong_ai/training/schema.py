"""Strict schemas, constraints, and reproducible candidate configuration."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from mahjong_ai.baseline.config import BaselineConfig, load_config
from mahjong_ai.baseline.feature_registry import FEATURE_GROUPS, FEATURE_NAMES


TRAINABLE_PATHS = frozenset(
    [*(f"weights.{name}" for name in FEATURE_NAMES)]
    + [*(f"group_weights.{name}" for name in FEATURE_GROUPS)]
    + [
        "risk_context.early", "risk_context.middle", "risk_context.late",
        "risk_context.two_threats", "risk_context.three_threats",
        "risk_context.dealer_threat", "risk_context.self_tenpai",
        "risk_context.self_one_shanten", "risk_context.high_value_hand",
    ]
)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    path: str
    initial: float
    minimum: float
    maximum: float
    scale: str

    def validate(self, value: float) -> float:
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"Parameter {self.name} must be finite")
        if not self.minimum <= value <= self.maximum:
            raise ValueError(
                f"Parameter {self.name}={value} outside [{self.minimum}, {self.maximum}]"
            )
        if self.scale == "log" and value <= 0:
            raise ValueError(f"Log-scale parameter {self.name} must be positive")
        return value


@dataclass(frozen=True)
class SeedSet:
    name: str
    base_seed: int
    matches: int

    @property
    def stop_seed(self) -> int:
        return self.base_seed + self.matches


@dataclass(frozen=True)
class EvaluationSpec:
    game_mode: str
    objective: str
    budgets: tuple[int, ...]


@dataclass(frozen=True)
class TrainingSpec:
    version: int
    name: str
    source_path: Path
    base_config_path: Path
    base_config: BaselineConfig
    parameters: dict[str, ParameterSpec]
    seed_sets: dict[str, SeedSet]
    evaluation: EvaluationSpec

    @property
    def initial_values(self) -> dict[str, float]:
        return {name: parameter.initial for name, parameter in self.parameters.items()}


def _strict_keys(raw: Any, required: set[str], section: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{section} must be a mapping")
    missing, unknown = required - set(raw), set(raw) - required
    if missing or unknown:
        raise ValueError(f"Invalid {section}; missing={sorted(missing)}, unknown={sorted(unknown)}")
    return raw


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        raw = yaml.safe_load(input_file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} root must be a mapping")
    return raw


def _load_seed_sets(path: Path) -> dict[str, SeedSet]:
    raw = _strict_keys(_read_yaml(path), {"version", "sets"}, "seed sets")
    if raw["version"] != 1:
        raise ValueError(f"Unsupported seed-set version: {raw['version']!r}")
    if not isinstance(raw["sets"], dict) or set(raw["sets"]) != {"train", "selection", "validation"}:
        raise ValueError("seed sets must contain exactly train, selection, validation")
    output = {}
    for name, item in raw["sets"].items():
        item = _strict_keys(item, {"base_seed", "matches"}, f"seed set {name}")
        seed_set = SeedSet(name, int(item["base_seed"]), int(item["matches"]))
        if seed_set.base_seed < 0 or seed_set.matches < 1 or seed_set.matches % 6:
            raise ValueError(f"Seed set {name} requires non-negative seed and matches divisible by 6")
        output[name] = seed_set
    ordered = sorted(output.values(), key=lambda item: item.base_seed)
    for left, right in zip(ordered, ordered[1:]):
        if left.stop_seed > right.base_seed:
            raise ValueError(f"Seed sets {left.name} and {right.name} overlap")
    return output


def _value_at_path(config: BaselineConfig, path: str) -> float:
    section, key = path.split(".", 1)
    return float(getattr(config, section)[key])


def load_training_spec(path: Path) -> TrainingSpec:
    path = path.resolve()
    raw = _strict_keys(
        _read_yaml(path),
        {"version", "name", "base_config", "seed_sets", "evaluation", "parameters"},
        "training spec",
    )
    if raw["version"] != 1:
        raise ValueError(f"Unsupported training-spec version: {raw['version']!r}")
    base_path = (path.parent / str(raw["base_config"])).resolve()
    base = load_config(base_path)
    parameters_raw = raw["parameters"]
    if not isinstance(parameters_raw, dict) or not parameters_raw:
        raise ValueError("parameters must be a non-empty mapping")
    parameters = {}
    used_paths = set()
    for name, item in parameters_raw.items():
        item = _strict_keys(item, {"path", "initial", "min", "max", "scale"}, f"parameter {name}")
        parameter = ParameterSpec(
            name=str(name), path=str(item["path"]), initial=float(item["initial"]),
            minimum=float(item["min"]), maximum=float(item["max"]), scale=str(item["scale"]),
        )
        if parameter.path not in TRAINABLE_PATHS:
            raise ValueError(f"Parameter {name} uses non-trainable path {parameter.path!r}")
        if parameter.path in used_paths:
            raise ValueError(f"Duplicate trainable path: {parameter.path}")
        if parameter.scale not in {"linear", "log"} or parameter.minimum > parameter.maximum:
            raise ValueError(f"Invalid range or scale for parameter {name}")
        parameter.validate(parameter.initial)
        if _value_at_path(base, parameter.path) != parameter.initial:
            raise ValueError(f"Initial value for {name} does not match base configuration")
        parameters[name] = parameter
        used_paths.add(parameter.path)
    evaluation_raw = _strict_keys(
        raw["evaluation"], {"game_mode", "objective", "budgets"}, "evaluation"
    )
    budgets = tuple(int(value) for value in evaluation_raw["budgets"])
    if not budgets or any(value < 1 or value % 6 for value in budgets) or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("evaluation budgets must be unique, increasing, and divisible by 6")
    seed_sets = _load_seed_sets((path.parent / str(raw["seed_sets"])).resolve())
    if budgets[-1] > seed_sets["train"].matches:
        raise ValueError("Largest evaluation budget exceeds the train seed set")
    if str(evaluation_raw["objective"]) not in {"average_rank_improvement"}:
        raise ValueError(f"Unsupported objective: {evaluation_raw['objective']!r}")
    return TrainingSpec(
        version=1, name=str(raw["name"]), source_path=path,
        base_config_path=base_path, base_config=base, parameters=parameters,
        seed_sets=seed_sets,
        evaluation=EvaluationSpec(
            game_mode=str(evaluation_raw["game_mode"]),
            objective=str(evaluation_raw["objective"]), budgets=budgets,
        ),
    )


def apply_parameters(spec: TrainingSpec, values: dict[str, float]) -> BaselineConfig:
    missing, unknown = set(spec.parameters) - set(values), set(values) - set(spec.parameters)
    if missing or unknown:
        raise ValueError(f"Candidate parameters missing={sorted(missing)}, unknown={sorted(unknown)}")
    sections = {
        "weights": dict(spec.base_config.weights),
        "group_weights": dict(spec.base_config.group_weights),
        "risk_context": dict(spec.base_config.risk_context),
    }
    for name, raw_value in values.items():
        parameter = spec.parameters[name]
        section, key = parameter.path.split(".", 1)
        sections[section][key] = parameter.validate(raw_value)
    candidate = replace(spec.base_config, **sections)
    return replace(candidate, name=f"{spec.name}-{config_hash(candidate)[:12]}")


def config_mapping(config: BaselineConfig) -> dict[str, Any]:
    return {
        "version": config.version,
        "name": config.name,
        "shape_mode": config.shape_mode,
        "policy": asdict(config.policy),
        "weights": dict(config.weights),
        "group_weights": dict(config.group_weights),
        "normalization": dict(config.normalization),
        "risk_context": dict(config.risk_context),
        "yaku": {
            "shanten_probability": dict(config.yaku.shanten_probability),
            "yakuhai": {
                "pair_probability": config.yaku.yakuhai_pair_probability,
                "single_probability": config.yaku.yakuhai_single_probability,
            },
            "tanyao": {
                "forbidden_tile_penalty": config.yaku.tanyao_forbidden_tile_penalty,
            },
            "flush": {
                "off_suit_penalty": config.yaku.flush_off_suit_penalty,
                "honor_penalty_for_chinitsu": config.yaku.flush_honor_penalty_for_chinitsu,
            },
        },
        "danger": dict(config.danger),
    }


def config_hash(config: BaselineConfig) -> str:
    mapping = config_mapping(config)
    mapping.pop("name")
    payload = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_config_snapshot(config: BaselineConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_mapping(config), sort_keys=True, allow_unicode=True), encoding="utf-8")
    return path
