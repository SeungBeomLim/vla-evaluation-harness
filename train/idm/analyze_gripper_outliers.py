"""Analyze gripper outliers from IDM evaluation samples."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SAMPLES = Path("train/idm/runs/idm_groot-n16_libero_h1_260526_1430/eval_h1/samples.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=5)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sign(value: float) -> int:
    return 1 if float(value) >= 0.0 else -1


def _safe_slice(arr: np.ndarray, step: int, window: int) -> tuple[int, np.ndarray]:
    start = max(0, int(step) - int(window))
    end = min(len(arr), int(step) + int(window) + 1)
    return start, arr[start:end]


def _neighbor_info(traj_path: Path, step: int, window: int) -> dict[str, Any]:
    with np.load(traj_path, allow_pickle=False) as traj:
        env = np.asarray(traj["env_action"], dtype=np.float32)
        server = np.asarray(traj["server_action"], dtype=np.float32) if "server_action" in traj.files else None
        start, grip = _safe_slice(env[:, 6], step, window)
        grip_sign = np.asarray([_sign(x) for x in grip], dtype=np.int8)
        sign_changes = np.flatnonzero(grip_sign[1:] != grip_sign[:-1]) + 1
        info: dict[str, Any] = {
            "neighbor_start": int(start),
            "neighbor_env_gripper": grip.astype(float).tolist(),
            "neighbor_env_gripper_sign": grip_sign.astype(int).tolist(),
            "gripper_sign_change_offsets": sign_changes.astype(int).tolist(),
            "gripper_switch_near": bool(len(sign_changes) > 0),
        }
        if "action_is_chunk_start" in traj.files:
            _, starts = _safe_slice(traj["action_is_chunk_start"].astype(bool), step, window)
            info["neighbor_chunk_start"] = starts.astype(bool).tolist()
            info["chunk_start_near"] = bool(starts.any())
            info["is_chunk_start"] = bool(traj["action_is_chunk_start"][step])
        if server is not None:
            _, server_grip = _safe_slice(server[:, 6], step, window)
            info["neighbor_server_gripper"] = server_grip.astype(float).tolist()
            info["env_server_gripper_max_abs_diff"] = float(np.max(np.abs(env[:, 6] - server[:, 6])))
    return info


def main() -> None:
    args = parse_args()
    rows = _read_jsonl(args.samples)
    output_dir = args.output_dir or args.samples.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzed: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row in rows:
        pred = np.asarray(row.get("pred_action", []), dtype=np.float32)
        target = np.asarray(row.get("target_action", []), dtype=np.float32)
        if len(pred) < 7 or len(target) < 7:
            continue
        grip_abs = float(abs(pred[6] - target[6]))
        pred_sign = _sign(float(pred[6]))
        target_sign = _sign(float(target[6]))
        record = {
            "trajectory": row.get("trajectory"),
            "step": int(row.get("step", -1)),
            "goal_step": int(row.get("goal_step", -1)),
            "pred_gripper": float(pred[6]),
            "target_gripper": float(target[6]),
            "pred_gripper_sign": pred_sign,
            "target_gripper_sign": target_sign,
            "gripper_sign_match": pred_sign == target_sign,
            "grip_abs": grip_abs,
            "action_l1": row.get("action_l1"),
            "pos_l2": row.get("pos_l2"),
            "rot_l2": row.get("rot_l2"),
            "current_to_goal_state_l2": row.get("current_to_goal_state_l2"),
        }
        if grip_abs >= args.threshold and row.get("trajectory"):
            record.update(_neighbor_info(Path(str(row["trajectory"])), int(row["step"]), int(args.window)))
            analyzed.append(record)
        if pred_sign == target_sign:
            counters["sign_match"] += 1
        else:
            counters["sign_flip"] += 1
        if grip_abs >= args.threshold:
            counters["outlier"] += 1
        if abs(float(target[6])) < 0.9:
            counters["soft_target"] += 1

    for record in analyzed:
        if record.get("gripper_switch_near"):
            counters["outlier_switch_near"] += 1
        if record.get("chunk_start_near"):
            counters["outlier_chunk_start_near"] += 1
        if record.get("is_chunk_start"):
            counters["outlier_is_chunk_start"] += 1

    summary = {
        "samples": len(rows),
        "threshold": float(args.threshold),
        "window": int(args.window),
        "sign_match": int(counters["sign_match"]),
        "sign_flip": int(counters["sign_flip"]),
        "sign_flip_ratio": float(counters["sign_flip"] / max(1, counters["sign_match"] + counters["sign_flip"])),
        "outliers": int(counters["outlier"]),
        "soft_targets_abs_lt_0_9": int(counters["soft_target"]),
        "outlier_switch_near": int(counters["outlier_switch_near"]),
        "outlier_chunk_start_near": int(counters["outlier_chunk_start_near"]),
        "outlier_is_chunk_start": int(counters["outlier_is_chunk_start"]),
    }
    (output_dir / "gripper_outliers.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in analyzed)
    )
    (output_dir / "gripper_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote gripper outliers to {output_dir / 'gripper_outliers.jsonl'}")


if __name__ == "__main__":
    main()
