"""CLI for building training manifests from rollout results with V2 matching."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.dataset_builder_v2 import RolloutDatasetBuilderV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to a rollout results directory, e.g. results/xvla_calvin_base_260422",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where success_only.jsonl, triplets.jsonl, and summary.json will be written.",
    )
    parser.add_argument("--success-segment-cap", type=int, default=30)
    parser.add_argument("--resample-points", type=int, default=41)
    parser.add_argument("--align-threshold", type=float, default=0.04)
    parser.add_argument("--align-run", type=int, default=3)
    parser.add_argument("--divergence-delta", type=float, default=0.015)
    parser.add_argument("--divergence-run", type=int, default=2)
    parser.add_argument("--triplet-window", type=int, default=5)
    parser.add_argument("--gripper-persistence-steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = RolloutDatasetBuilderV2(
        args.results_dir,
        success_segment_cap=args.success_segment_cap,
        resample_points=args.resample_points,
        align_threshold=args.align_threshold,
        align_run=args.align_run,
        divergence_delta=args.divergence_delta,
        divergence_run=args.divergence_run,
        triplet_window=args.triplet_window,
        gripper_persistence_steps=args.gripper_persistence_steps,
    )
    summary = builder.build(Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
