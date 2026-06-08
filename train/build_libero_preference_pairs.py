"""CLI for building GR00T/LIBERO chunk preference pair manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.libero_preference_pair_builder import LiberoPreferencePairBuilder


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
    parser.add_argument("--default-chunk-size", type=int, default=16)
    parser.add_argument("--min-target-offset", type=int, default=1)
    parser.add_argument("--max-target-offset", type=int, default=None)
    parser.add_argument(
        "--target-window-size",
        type=int,
        default=3,
        help="Contiguous target actions required inside the selected action chunk.",
    )
    parser.add_argument("--local-future-steps", type=int, default=4)
    parser.add_argument("--policy-state-weight", type=float, default=0.5)
    parser.add_argument("--policy-future-weight", type=float, default=0.4)
    parser.add_argument("--policy-action-weight", type=float, default=0.1)
    parser.add_argument("--target-growth-weight", type=float, default=0.35)
    parser.add_argument("--target-step-growth-weight", type=float, default=0.25)
    parser.add_argument("--target-future-weight", type=float, default=0.2)
    parser.add_argument("--target-action-weight", type=float, default=0.15)
    parser.add_argument("--target-earliness-weight", type=float, default=0.05)
    parser.add_argument("--state-pos-weight", type=float, default=1.0)
    parser.add_argument("--state-rot-weight", type=float, default=1.0)
    parser.add_argument("--state-grip-weight", type=float, default=0.5)
    parser.add_argument("--action-pos-weight", type=float, default=0.45)
    parser.add_argument("--action-rot-weight", type=float, default=0.35)
    parser.add_argument("--action-grip-weight", type=float, default=0.2)
    parser.add_argument("--action-grip-cap", type=float, default=1.0)
    parser.add_argument(
        "--success-ratio",
        type=float,
        default=1.5,
        help="Target success-only samples as a multiple of preference pair samples.",
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
    builder = LiberoPreferencePairBuilder(
        rollout_roots=[Path(p) for p in args.rollout_root],
        output_dir=Path(args.output_dir),
        action_key=args.action_key,
        default_chunk_size=args.default_chunk_size,
        min_target_offset=args.min_target_offset,
        max_target_offset=args.max_target_offset,
        target_window_size=args.target_window_size,
        local_future_steps=args.local_future_steps,
        policy_state_weight=args.policy_state_weight,
        policy_future_weight=args.policy_future_weight,
        policy_action_weight=args.policy_action_weight,
        target_growth_weight=args.target_growth_weight,
        target_step_growth_weight=args.target_step_growth_weight,
        target_future_weight=args.target_future_weight,
        target_action_weight=args.target_action_weight,
        target_earliness_weight=args.target_earliness_weight,
        state_pos_weight=args.state_pos_weight,
        state_rot_weight=args.state_rot_weight,
        state_grip_weight=args.state_grip_weight,
        action_pos_weight=args.action_pos_weight,
        action_rot_weight=args.action_rot_weight,
        action_grip_weight=args.action_grip_weight,
        action_grip_cap=args.action_grip_cap,
        success_ratio=args.success_ratio,
        success_samples_per_rollout=args.success_samples_per_rollout,
        random_seed=args.random_seed,
    )
    summary = builder.build()
    print(f"Wrote {summary['preference_pairs']} preference pairs")
    print(f"Wrote {summary['success_only']} success-only samples")
    print(f"Summary: {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
