"""Deterministic one-factor-at-a-time parameter ablation."""

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mahjong_ai.training.schema import apply_parameters, config_hash, load_training_spec, write_config_snapshot
from mahjong_ai.training.search import (
    CandidateDefinition,
    _evaluate_candidate,
    _git_state,
    _spec_fingerprint,
    _write_json,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Ablation config must be a mapping")
    required = {
        "version", "name", "training_spec", "budget", "bootstrap_samples",
        "bootstrap_seed", "levels",
    }
    if set(raw) != required:
        raise ValueError(
            f"Ablation config keys missing={sorted(required - set(raw))}, "
            f"unknown={sorted(set(raw) - required)}"
        )
    if raw["version"] != 1:
        raise ValueError(f"Unsupported ablation config version: {raw['version']!r}")
    return raw


def build_ablation_candidates(config_path: Path) -> tuple[
    Path, int, int, int, tuple[tuple[str, float | None, CandidateDefinition], ...]
]:
    config_path = config_path.resolve()
    raw = _load_yaml(config_path)
    spec_path = (config_path.parent / str(raw["training_spec"])).resolve()
    spec = load_training_spec(spec_path)
    levels = raw["levels"]
    if not isinstance(levels, dict) or set(levels) != set(spec.parameters):
        raise ValueError("Ablation levels must exactly match training-spec parameters")

    definitions: list[tuple[str, float | None, CandidateDefinition]] = []
    baseline_values = spec.initial_values
    baseline_config = apply_parameters(spec, baseline_values)
    definitions.append(("baseline", None, CandidateDefinition(
        config_hash(baseline_config)[:16], baseline_values,
    )))
    for parameter_name in spec.parameters:
        raw_levels = levels[parameter_name]
        if not isinstance(raw_levels, list) or len(raw_levels) < 3:
            raise ValueError(f"{parameter_name} requires at least three ablation levels")
        numeric_levels = [spec.parameters[parameter_name].validate(value) for value in raw_levels]
        if numeric_levels != sorted(set(numeric_levels)):
            raise ValueError(f"{parameter_name} levels must be unique and increasing")
        initial = baseline_values[parameter_name]
        if initial not in numeric_levels:
            raise ValueError(f"{parameter_name} levels must include initial value {initial}")
        for value in numeric_levels:
            if value == initial:
                continue
            values = dict(baseline_values)
            values[parameter_name] = value
            candidate_config = apply_parameters(spec, values)
            definitions.append((parameter_name, value, CandidateDefinition(
                config_hash(candidate_config)[:16], values,
            )))
    return (
        spec_path, int(raw["budget"]), int(raw["bootstrap_samples"]),
        int(raw["bootstrap_seed"]), tuple(definitions),
    )


def run_ablation(
    *, config_path: Path, output_dir: Path, workers: int, allow_dirty: bool = False,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    spec_path, budget, bootstrap_samples, bootstrap_seed, definitions = (
        build_ablation_candidates(config_path)
    )
    spec = load_training_spec(spec_path)
    if budget < 1 or budget % 6 or budget > spec.seed_sets["train"].matches:
        raise ValueError("Ablation budget must be divisible by 6 and within train seeds")
    if bootstrap_samples < 1 or workers < 1:
        raise ValueError("bootstrap_samples and workers must be positive")

    workspace = spec_path.parents[2]
    git_commit, git_dirty = _git_state(workspace)
    if git_dirty and not allow_dirty:
        raise ValueError("Working tree is dirty; commit changes or pass allow_dirty for a smoke run")
    config_fingerprint = hashlib.sha256(config_path.read_bytes()).hexdigest()
    settings = {
        "config_path": str(config_path), "config_fingerprint": config_fingerprint,
        "spec_path": str(spec_path), "spec_fingerprint": _spec_fingerprint(spec),
        "git_commit": git_commit, "git_dirty": git_dirty, "budget": budget,
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in settings.items() if manifest.get(key) != value]
        if mismatches:
            raise ValueError(f"Existing ablation manifest differs in: {', '.join(mismatches)}")
    else:
        manifest = {
            "version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            **settings, "workers": workers,
            "candidates": [
                {"axis": axis, "level": level, **asdict(candidate)}
                for axis, level, candidate in definitions
            ],
        }
        _write_json(manifest_path, manifest)
        for _, _, candidate in definitions:
            write_config_snapshot(
                apply_parameters(spec, candidate.values),
                output_dir / "candidates" / f"{candidate.candidate_id}.yaml",
            )

    result_by_id = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(definitions))) as executor:
        futures = {
            executor.submit(
                _evaluate_candidate, str(spec_path), candidate, budget, str(output_dir),
                bootstrap_samples, bootstrap_seed,
            ): (axis, level, candidate)
            for axis, level, candidate in definitions
        }
        for future in as_completed(futures):
            axis, level, candidate = futures[future]
            result = future.result()
            result_by_id[candidate.candidate_id] = result
            print(
                f"axis={axis} level={level} candidate={candidate.candidate_id} "
                f"rank={result.objective:+.4f} elapsed={result.elapsed_seconds:.1f}s"
                + (" resumed" if result.resumed else ""), flush=True,
            )

    rows = []
    for axis, level, candidate in definitions:
        rows.append({"axis": axis, "level": level, **asdict(result_by_id[candidate.candidate_id])})
    curves = {
        name: sorted(
            [row for row in rows if row["axis"] in {"baseline", name}],
            key=lambda row: spec.initial_values[name] if row["level"] is None else row["level"],
        )
        for name in spec.parameters
    }
    best = max(rows, key=lambda row: (row["objective"], row["point_improvement"]))
    summary = {"manifest": str(manifest_path), "rows": rows, "curves": curves, "best": best}
    _write_json(output_dir / "summary.json", summary)
    _write_markdown(output_dir / "summary.md", curves, spec.initial_values)
    return summary


def _write_markdown(
    path: Path, curves: dict[str, list[dict[str, Any]]], initial_values: dict[str, float],
) -> None:
    lines = [
        "# 组权重单变量消融", "",
        "> 每张表只改变标题所示参数，其他参数保持基线值。正数表示优于训练基线。", "",
    ]
    for parameter, rows in curves.items():
        lines.extend([
            f"## {parameter}", "",
            "| 参数值 | 平均顺位改善 | 95% CI | 平均点差改善 |",
            "|---:|---:|---:|---:|",
        ])
        for row in rows:
            level = initial_values[parameter] if row["level"] is None else row["level"]
            marker = "（基线）" if row["axis"] == "baseline" else ""
            lines.append(
                f"| {level:.4g}{marker} | {row['objective']:+.4f} | "
                f"[{row['rank_ci_low']:+.4f}, {row['rank_ci_high']:+.4f}] | "
                f"{row['point_improvement']:+.1f} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
