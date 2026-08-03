"""Independent seed-set validation for one trained parameter candidate."""

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from mahjong_ai.baseline import BaselineEngine
from mahjong_ai.evaluation import run_comparison
from mahjong_ai.training.schema import apply_parameters, config_hash, load_training_spec, write_config_snapshot
from mahjong_ai.training.search import _git_state, _spec_fingerprint, _write_json


def prepare_parameter_candidate(
    spec_path: Path, parameter: str, value: float, seed_set: str,
) -> tuple[Any, Any, Any]:
    spec = load_training_spec(spec_path.resolve())
    if parameter not in spec.parameters:
        raise ValueError(f"Unknown trainable parameter: {parameter}")
    if seed_set not in spec.seed_sets:
        raise ValueError(f"Unknown seed set: {seed_set}")
    values = spec.initial_values
    values[parameter] = spec.parameters[parameter].validate(value)
    return spec, apply_parameters(spec, values), spec.seed_sets[seed_set]


def run_parameter_validation(
    *, spec_path: Path, output_dir: Path, parameter: str, value: float,
    seed_set: str = "selection", matches: int | None = None,
    bootstrap_samples: int = 5000, bootstrap_seed: int = 20260811,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    output_dir = output_dir.resolve()
    spec, candidate, seeds = prepare_parameter_candidate(spec_path, parameter, value, seed_set)
    match_count = seeds.matches if matches is None else matches
    if match_count < 1 or match_count % 6 or match_count > seeds.matches:
        raise ValueError("matches must be divisible by 6 and within the selected seed set")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    workspace = spec_path.parents[2]
    git_commit, git_dirty = _git_state(workspace)
    if git_dirty and not allow_dirty:
        raise ValueError("Working tree is dirty; commit changes or pass allow_dirty for a smoke run")
    candidate_id = config_hash(candidate)[:16]
    settings = {
        "spec_path": str(spec_path), "spec_fingerprint": _spec_fingerprint(spec),
        "git_commit": git_commit, "git_dirty": git_dirty,
        "parameter": parameter, "value": float(value), "candidate_id": candidate_id,
        "seed_set": seed_set, "base_seed": seeds.base_seed, "matches": match_count,
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
    }
    manifest_path = output_dir / "manifest.json"
    result_path = output_dir / "result.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [key for key, expected in settings.items() if manifest.get(key) != expected]
        if mismatches:
            raise ValueError(f"Existing validation manifest differs in: {', '.join(mismatches)}")
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
    else:
        _write_json(manifest_path, {
            "version": 1, "created_at": datetime.now(timezone.utc).isoformat(), **settings,
            "parameters": {**spec.initial_values, parameter: float(value)},
        })
        write_config_snapshot(candidate, output_dir / f"candidate-{candidate_id}.yaml")

    started = perf_counter()
    result = run_comparison(
        engine_a_name="candidate",
        engine_a_factory=lambda _pid, _seed: BaselineEngine(config=candidate),
        engine_b_name="training-baseline",
        engine_b_factory=lambda _pid, _seed: BaselineEngine(config=spec.base_config),
        match_count=match_count,
        base_seed=seeds.base_seed,
        game_mode=spec.evaluation.game_mode,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    payload = result.to_dict()
    payload["validation_metadata"] = {**settings, "elapsed_seconds": perf_counter() - started}
    _write_json(result_path, payload)
    result.write_markdown(output_dir / "result.md")
    rank = result.paired_deltas["平均顺位改善"]
    point = result.paired_deltas["平均点差改善"]
    summary = {
        "candidate_id": candidate_id, "parameter": parameter, "value": float(value),
        "seed_set": seed_set, "matches": match_count,
        "average_rank_improvement": asdict_like(rank),
        "average_point_improvement": asdict_like(point),
        "result_path": str(result_path),
    }
    _write_json(output_dir / "summary.json", summary)
    return payload


def asdict_like(value: dict[str, float]) -> dict[str, float]:
    return {key: float(item) for key, item in value.items()}
