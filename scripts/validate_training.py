"""Validate a training specification and materialize its initial candidate."""

import argparse
from pathlib import Path

from mahjong_ai.training import apply_parameters, config_hash, load_training_spec
from mahjong_ai.training.schema import write_config_snapshot


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--spec", type=Path, default=Path("configs/training/stage1-search.yaml"))
parser.add_argument("--output", type=Path, default=Path("artifacts/training/initial-config.yaml"))
args = parser.parse_args()

spec = load_training_spec(args.spec)
candidate = apply_parameters(spec, spec.initial_values)
write_config_snapshot(candidate, args.output)
print(f"training={spec.name}")
print(f"parameters={len(spec.parameters)}")
print(f"config_hash={config_hash(candidate)}")
print(f"snapshot={args.output}")
