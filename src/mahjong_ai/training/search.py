"""Deterministic candidate generation and parallel successive halving."""

import hashlib
import json
import math
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from mahjong_ai.baseline import BaselineEngine
from mahjong_ai.evaluation import run_comparison
from mahjong_ai.training.schema import (
    apply_parameters,
    config_hash,
    load_training_spec,
    write_config_snapshot,
)


@dataclass(frozen=True)
class CandidateDefinition:
    candidate_id: str
    values: dict[str, float]


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    budget: int
    objective: float
    rank_ci_low: float
    rank_ci_high: float
    point_improvement: float
    result_path: str
    elapsed_seconds: float
    resumed: bool = False


def sample_candidates(spec_path: Path, count: int, search_seed: int) -> tuple[CandidateDefinition, ...]:
    if count < 1:
        raise ValueError("candidate count must be at least 1")
    spec = load_training_spec(spec_path)
    rng = random.Random(search_seed)
    value_sets = [spec.initial_values]
    while len(value_sets) < count:
        values = {}
        for name, parameter in spec.parameters.items():
            if parameter.scale == "log":
                values[name] = math.exp(
                    rng.uniform(math.log(parameter.minimum), math.log(parameter.maximum))
                )
            else:
                values[name] = rng.uniform(parameter.minimum, parameter.maximum)
        value_sets.append(values)
    output = []
    seen = set()
    for values in value_sets:
        candidate = apply_parameters(spec, values)
        candidate_id = config_hash(candidate)[:16]
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        output.append(CandidateDefinition(candidate_id, values))
    return tuple(output)


def _git_state(workspace: Path) -> tuple[str, bool]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace,
        check=False, capture_output=True, text=True,
    )
    commit = completed.stdout.strip() if completed.returncode == 0 else "unknown"
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=workspace,
        check=False, capture_output=True, text=True,
    )
    return commit, bool(status.stdout.strip()) if status.returncode == 0 else True


def _spec_fingerprint(spec: Any) -> str:
    value = {
        "name": spec.name,
        "base_config_hash": config_hash(spec.base_config),
        "parameters": {name: asdict(item) for name, item in spec.parameters.items()},
        "seed_sets": {name: asdict(item) for name, item in spec.seed_sets.items()},
        "evaluation": asdict(spec.evaluation),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_cached_result(path: Path, candidate_id: str, budget: int) -> CandidateResult | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    metadata = raw.get("training_metadata", {})
    if metadata.get("candidate_id") != candidate_id or metadata.get("budget") != budget:
        return None
    rank = raw["paired_deltas"]["平均顺位改善"]
    point = raw["paired_deltas"]["平均点差改善"]
    return CandidateResult(
        candidate_id, budget, float(rank["estimate"]), float(rank["ci_low"]),
        float(rank["ci_high"]), float(point["estimate"]), str(path),
        float(metadata.get("elapsed_seconds", 0.0)), resumed=True,
    )


def _evaluate_candidate(
    spec_path_text: str,
    candidate: CandidateDefinition,
    budget: int,
    output_dir_text: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> CandidateResult:
    spec_path = Path(spec_path_text)
    output_dir = Path(output_dir_text)
    result_path = output_dir / "results" / f"n{budget}" / f"{candidate.candidate_id}.json"
    cached = _read_cached_result(result_path, candidate.candidate_id, budget)
    if cached is not None:
        return cached

    candidate_config = apply_parameters(load_training_spec(spec_path), candidate.values)
    spec = load_training_spec(spec_path)
    baseline_config = spec.base_config
    started = perf_counter()
    result = run_comparison(
        engine_a_name="candidate",
        engine_a_factory=lambda _pid, _seed: BaselineEngine(config=candidate_config),
        engine_b_name="training-baseline",
        engine_b_factory=lambda _pid, _seed: BaselineEngine(config=baseline_config),
        match_count=budget,
        base_seed=spec.seed_sets["train"].base_seed,
        game_mode=spec.evaluation.game_mode,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    elapsed = perf_counter() - started
    payload = result.to_dict()
    payload["training_metadata"] = {
        "candidate_id": candidate.candidate_id,
        "parameters": candidate.values,
        "budget": budget,
        "elapsed_seconds": elapsed,
    }
    _write_json(result_path, payload)
    markdown_path = result_path.with_suffix(".md")
    result.write_markdown(markdown_path)
    rank = result.paired_deltas["平均顺位改善"]
    point = result.paired_deltas["平均点差改善"]
    return CandidateResult(
        candidate.candidate_id, budget, float(rank["estimate"]),
        float(rank["ci_low"]), float(rank["ci_high"]),
        float(point["estimate"]), str(result_path), elapsed,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_search(
    *,
    spec_path: Path,
    output_dir: Path,
    candidate_count: int,
    search_seed: int,
    workers: int,
    budgets: tuple[int, ...] | None = None,
    keep_ratio: float = 0.25,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20260803,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    output_dir = output_dir.resolve()
    spec = load_training_spec(spec_path)
    selected_budgets = budgets or spec.evaluation.budgets
    if (
        not selected_budgets
        or tuple(sorted(set(selected_budgets))) != selected_budgets
        or any(value < 1 or value % 6 for value in selected_budgets)
        or selected_budgets[-1] > spec.seed_sets["train"].matches
    ):
        raise ValueError("budgets must be unique, increasing, divisible by 6, and within train seeds")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    workspace = spec_path.parents[2]
    git_commit, git_dirty = _git_state(workspace)
    if git_dirty and not allow_dirty:
        raise ValueError("Working tree is dirty; commit changes or pass allow_dirty for a smoke run")
    fingerprint = _spec_fingerprint(spec)
    run_settings = {
        "spec_path": str(spec_path), "spec_fingerprint": fingerprint,
        "git_commit": git_commit, "git_dirty": git_dirty,
        "search_seed": search_seed, "candidate_count": candidate_count,
        "budgets": list(selected_budgets), "keep_ratio": keep_ratio,
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
    }

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidates = tuple(
            CandidateDefinition(item["candidate_id"], item["values"])
            for item in manifest["candidates"]
        )
        mismatches = [key for key, value in run_settings.items() if manifest.get(key) != value]
        if mismatches:
            raise ValueError(f"Existing run manifest differs in: {', '.join(mismatches)}")
    else:
        candidates = sample_candidates(spec_path, candidate_count, search_seed)
        manifest = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "spec_path": str(spec_path),
            "spec_name": spec.name,
            **run_settings,
            "workers": workers,
            "candidates": [asdict(candidate) for candidate in candidates],
        }
        _write_json(manifest_path, manifest)
        for candidate in candidates:
            config = apply_parameters(spec, candidate.values)
            write_config_snapshot(config, output_dir / "candidates" / f"{candidate.candidate_id}.yaml")

    active = list(candidates)
    stage_summaries = []
    for stage_index, budget in enumerate(selected_budgets):
        results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(active))) as executor:
            futures = {
                executor.submit(
                    _evaluate_candidate, str(spec_path), candidate, budget, str(output_dir),
                    bootstrap_samples, bootstrap_seed + stage_index,
                ): candidate
                for candidate in active
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"stage={stage_index + 1} budget={budget} candidate={result.candidate_id} "
                    f"rank={result.objective:+.4f} elapsed={result.elapsed_seconds:.1f}s"
                    + (" resumed" if result.resumed else ""),
                    flush=True,
                )
        results.sort(key=lambda item: (-item.objective, -item.point_improvement, item.candidate_id))
        keep_count = len(results) if stage_index == len(selected_budgets) - 1 else max(
            1, math.ceil(len(results) * keep_ratio)
        )
        active_ids = {item.candidate_id for item in results[:keep_count]}
        active = [candidate for candidate in active if candidate.candidate_id in active_ids]
        summary = {
            "stage": stage_index + 1,
            "budget": budget,
            "keep_count": keep_count,
            "results": [asdict(result) for result in results],
            "survivors": [candidate.candidate_id for candidate in active],
        }
        stage_summaries.append(summary)
        _write_json(output_dir / f"stage-{stage_index + 1}-summary.json", summary)

    final = stage_summaries[-1]["results"][0]
    summary = {"manifest": str(manifest_path), "stages": stage_summaries, "best": final}
    _write_json(output_dir / "summary.json", summary)
    return summary
