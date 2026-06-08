"""Analyze GR00T/LIBERO preference pair manifests before LoRA training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _trajectory_path(ref: dict[str, Any]) -> Path:
    return Path(str(ref["root"])) / Path(str(ref["trajectory"]))


def _load_window(ref: dict[str, Any], *, action_key: str, horizon: int) -> np.ndarray:
    with np.load(_trajectory_path(ref), allow_pickle=False) as traj:
        step = int(ref["action_step"])
        actions = np.asarray(traj[action_key], dtype=np.float32)
        return actions[step : min(len(actions), step + horizon)]


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _input_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    ref = row["vla_input"]
    return (
        str(ref["root"]),
        str(ref["trajectory"]),
        int(ref["obs_step"]),
        int(ref.get("state_step", ref["obs_step"])),
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(Path(args.preference_pairs))
    suite_counts = Counter(str(r.get("suite", "unknown")) for r in rows)
    task_counts = Counter(f"{r.get('suite', 'unknown')}:{r.get('task_id', 'unknown')}:{r.get('task', '')}" for r in rows)
    offsets = [int(r.get("action_offset", 0)) for r in rows]
    target_windows = [int(r.get("target_window_size", 0)) for r in rows]

    input_groups: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        input_groups[_input_key(row)].append(idx)
    dup_group_sizes = [len(v) for v in input_groups.values() if len(v) > 1]

    action_gaps: list[float] = []
    action_pos_gaps: list[float] = []
    action_rot_gaps: list[float] = []
    action_grip_gaps: list[float] = []
    weak_pairs = 0
    if args.compute_action_gaps:
        for row in rows:
            horizon = int(row.get("target_window_size", 1))
            key = str(row.get("action_key", args.action_key))
            pos = _load_window(row["positive"], action_key=key, horizon=horizon)
            neg = _load_window(row["negative"], action_key=key, horizon=horizon)
            n = min(len(pos), len(neg))
            if n <= 0:
                continue
            diff = pos[:n] - neg[:n]
            gap = float(np.linalg.norm(diff) / max(1, n))
            action_gaps.append(gap)
            action_pos_gaps.append(float(np.linalg.norm(diff[:, 0:3]) / max(1, n)))
            action_rot_gaps.append(float(np.linalg.norm(diff[:, 3:6]) / max(1, n)))
            action_grip_gaps.append(float(np.linalg.norm(diff[:, 6:7]) / max(1, n)))
            if gap <= args.weak_action_gap:
                weak_pairs += 1

    summary = {
        "num_pairs": len(rows),
        "suite_counts": dict(suite_counts),
        "top_tasks": dict(task_counts.most_common(args.top_k)),
        "action_offset": {
            "counts": dict(Counter(offsets)),
            "stats": _percentiles([float(x) for x in offsets]),
        },
        "target_window_size": {
            "counts": dict(Counter(target_windows)),
            "stats": _percentiles([float(x) for x in target_windows]),
        },
        "duplicate_vla_input": {
            "num_unique_inputs": len(input_groups),
            "num_duplicate_groups": len(dup_group_sizes),
            "num_rows_in_duplicate_groups": int(sum(dup_group_sizes)),
            "group_size_stats": _percentiles([float(x) for x in dup_group_sizes]),
        },
    }
    if args.compute_action_gaps:
        summary["positive_negative_action_gap"] = {
            "weak_action_gap_threshold": args.weak_action_gap,
            "weak_pair_count": weak_pairs,
            "weak_pair_ratio": weak_pairs / max(1, len(action_gaps)),
            "all": _percentiles(action_gaps),
            "position": _percentiles(action_pos_gaps),
            "rotation": _percentiles(action_rot_gaps),
            "gripper": _percentiles(action_grip_gaps),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference-pairs", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--compute-action-gaps", action="store_true")
    parser.add_argument("--action-key", default="env_action")
    parser.add_argument("--weak-action-gap", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Wrote preference pair analysis: {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
