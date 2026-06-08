"""CLI for building LIBERO same-initial-state triplet manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.libero_triplet_dataset_builder import LiberoTripletDatasetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-root",
        action="append",
        required=True,
        help="Benchmark output directory for one seed. Repeat for multiple seeds.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write manifests.")
    parser.add_argument("--action-key", default="env_action", choices=["env_action", "server_action"])
    parser.add_argument("--future-chunks", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=1, help="Triplet chunks per success/failure pair.")
    parser.add_argument("--state-weight", type=float, default=0.4)
    parser.add_argument("--action-weight", type=float, default=0.4)
    parser.add_argument("--future-weight", type=float, default=0.2)
    parser.add_argument(
        "--success-ratio",
        type=float,
        default=1.5,
        help="Target success-only samples as a multiple of triplet samples.",
    )
    parser.add_argument(
        "--success-samples-per-rollout",
        type=int,
        default=3,
        help="Max candidate success-only chunk samples to collect per successful rollout.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = LiberoTripletDatasetBuilder(
        rollout_roots=[Path(p) for p in args.rollout_root],
        output_dir=Path(args.output_dir),
        action_key=args.action_key,
        future_chunks=args.future_chunks,
        top_k=args.top_k,
        state_weight=args.state_weight,
        action_weight=args.action_weight,
        future_weight=args.future_weight,
        success_ratio=args.success_ratio,
        success_samples_per_rollout=args.success_samples_per_rollout,
        random_seed=args.random_seed,
    )
    summary = builder.build()
    print(f"Wrote {summary['triplets']} triplets")
    print(f"Wrote {summary['success_only']} success-only samples")
    print(f"Summary: {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
