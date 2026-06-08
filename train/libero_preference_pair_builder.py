"""Build LIBERO chunk preference/recovery pair manifests for GR00T training.

This builder replaces the earlier single-step triplet selection with a
two-step selection:

1. choose a chunk-start policy observation step `c`;
2. choose a target offset `k` inside that chunk, where success/failure states
   and actions have actually diverged.

The resulting sample is designed for chunk-policy training:

    VLA input:   failure observation/state/language at `c`
    NCE target: success/failure actions at `t = c + k`
    IDM input:  failure/goal image observations at `t`
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class EpisodeRecord:
    root: Path
    rollout_dir: Path
    trajectory_path: Path
    metadata_path: Path
    suite: str
    task_id: int
    episode_idx: int
    task: str
    seed: int | None
    success: bool
    steps: int | None

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.suite, self.task_id, self.episode_idx)

    @property
    def rollout_rel(self) -> str:
        return str(self.rollout_dir)

    @property
    def trajectory_rel(self) -> str:
        return str(self.trajectory_path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _infer_seed(root: Path, fallback: int | None = None) -> int | None:
    match = re.search(r"seed(\d+)", root.name)
    if match:
        return int(match.group(1))
    return fallback


def _episode_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*LIBEROBenchmark_*_episodes.jsonl"))
    preferred = [p for p in files if p.name.startswith("groot_libero_base_collect")]
    return preferred or [p for p in files if p.name.startswith("LIBEROBenchmark_")]


def _high_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return values
    if n == 1:
        return np.ones(1, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64) / float(n - 1)
    return ranks


def _low_rank(values: np.ndarray) -> np.ndarray:
    return 1.0 - _high_rank(values)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _norm(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(arr.astype(np.float64), axis=axis)


def _component_state_distances(states_s: np.ndarray, states_f: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "pos": _norm(states_s[:, 0:3] - states_f[:, 0:3]),
        "rot": _norm(states_s[:, 3:6] - states_f[:, 3:6]),
        "grip": _norm(states_s[:, 6:] - states_f[:, 6:]),
    }


def _component_action_distances(
    actions_s: np.ndarray,
    actions_f: np.ndarray,
    *,
    grip_cap: float,
) -> dict[str, np.ndarray]:
    grip = np.abs(actions_s[:, 6] - actions_f[:, 6]).astype(np.float64)
    return {
        "pos": _norm(actions_s[:, 0:3] - actions_f[:, 0:3]),
        "rot": _norm(actions_s[:, 3:6] - actions_f[:, 3:6]),
        "grip": np.minimum(grip, float(grip_cap)),
        "grip_raw": grip,
    }


def _weighted_minmax_scalar(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    vals = []
    total = 0.0
    for key, weight in weights.items():
        if key not in components or weight <= 0:
            continue
        vals.append(_minmax(components[key]) * float(weight))
        total += float(weight)
    if not vals or total <= 0:
        first = next(iter(components.values()))
        return np.zeros_like(first, dtype=np.float64)
    return np.sum(vals, axis=0) / total


def _chunk_start_steps(traj: np.lib.npyio.NpzFile, length: int) -> list[int]:
    if "action_is_chunk_start" in traj.files:
        starts = np.flatnonzero(traj["action_is_chunk_start"].astype(bool))
        if len(starts):
            return [int(x) for x in starts if int(x) < length]
    if "model_action_chunk_start_step" in traj.files:
        starts = sorted({int(x) for x in traj["model_action_chunk_start_step"]})
        if starts:
            return [x for x in starts if 0 <= x < length]
    if "action_chunk_start_step" in traj.files:
        starts = sorted({int(x) for x in traj["action_chunk_start_step"]})
        if starts:
            return [x for x in starts if 0 <= x < length]
    return list(range(0, length, 16))


def _step_chunk_start(traj: np.lib.npyio.NpzFile, step: int) -> int:
    if "action_chunk_start_step" in traj.files and step < len(traj["action_chunk_start_step"]):
        return int(traj["action_chunk_start_step"][step])
    starts = _chunk_start_steps(traj, step + 1)
    starts = [x for x in starts if x <= step]
    return starts[-1] if starts else step


def _step_chunk_size(traj: np.lib.npyio.NpzFile, step: int, default: int = 16) -> int:
    if "action_chunk_size" in traj.files and step < len(traj["action_chunk_size"]):
        size = int(traj["action_chunk_size"][step])
        if size > 0:
            return size
    if "model_action_chunk_length" in traj.files and "model_action_chunk_start_step" in traj.files:
        starts = traj["model_action_chunk_start_step"].astype(int)
        matches = np.flatnonzero(starts == int(step))
        if len(matches):
            size = int(traj["model_action_chunk_length"][matches[0]])
            if size > 0:
                return size
    return int(default)


class LiberoPreferencePairBuilder:
    def __init__(
        self,
        *,
        rollout_roots: list[Path],
        output_dir: Path,
        action_key: str = "env_action",
        default_chunk_size: int = 16,
        min_target_offset: int = 1,
        max_target_offset: int | None = None,
        target_window_size: int = 3,
        local_future_steps: int = 4,
        policy_state_weight: float = 0.5,
        policy_future_weight: float = 0.4,
        policy_action_weight: float = 0.1,
        target_growth_weight: float = 0.35,
        target_step_growth_weight: float = 0.25,
        target_future_weight: float = 0.2,
        target_action_weight: float = 0.15,
        target_earliness_weight: float = 0.05,
        state_pos_weight: float = 1.0,
        state_rot_weight: float = 1.0,
        state_grip_weight: float = 0.5,
        action_pos_weight: float = 0.45,
        action_rot_weight: float = 0.35,
        action_grip_weight: float = 0.2,
        action_grip_cap: float = 1.0,
        success_ratio: float = 1.5,
        success_samples_per_rollout: int = 3,
        random_seed: int = 0,
    ) -> None:
        self.rollout_roots = [p.resolve() for p in rollout_roots]
        self.output_dir = output_dir
        self.action_key = action_key
        self.default_chunk_size = max(1, int(default_chunk_size))
        self.min_target_offset = max(1, int(min_target_offset))
        self.max_target_offset = max_target_offset if max_target_offset is None else max(1, int(max_target_offset))
        self.target_window_size = max(1, int(target_window_size))
        self.local_future_steps = max(1, int(local_future_steps))
        self.policy_state_weight = float(policy_state_weight)
        self.policy_future_weight = float(policy_future_weight)
        self.policy_action_weight = float(policy_action_weight)
        self.target_growth_weight = float(target_growth_weight)
        self.target_step_growth_weight = float(target_step_growth_weight)
        self.target_future_weight = float(target_future_weight)
        self.target_action_weight = float(target_action_weight)
        self.target_earliness_weight = float(target_earliness_weight)
        self.state_weights = {
            "pos": float(state_pos_weight),
            "rot": float(state_rot_weight),
            "grip": float(state_grip_weight),
        }
        self.action_weights = {
            "pos": float(action_pos_weight),
            "rot": float(action_rot_weight),
            "grip": float(action_grip_weight),
        }
        self.action_grip_cap = float(action_grip_cap)
        self.success_ratio = max(0.0, float(success_ratio))
        self.success_samples_per_rollout = max(0, int(success_samples_per_rollout))
        self.random_seed = int(random_seed)

    def build(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = self._load_records()
        by_key: dict[tuple[str, int, int], list[EpisodeRecord]] = defaultdict(list)
        for record in records:
            by_key[record.key].append(record)

        pairs: list[dict[str, Any]] = []
        skipped = Counter()
        for key in sorted(by_key):
            successes = [r for r in by_key[key] if r.success]
            failures = [r for r in by_key[key] if not r.success]
            if not successes or not failures:
                continue
            for success in successes:
                for failure in failures:
                    row, reason = self._select_pair(success, failure)
                    if row is None:
                        skipped[reason or "no_pair"] += 1
                    else:
                        pairs.append(row)

        success_rows = self._build_success_only(records, target_count=round(len(pairs) * self.success_ratio))

        pair_count = _write_jsonl(self.output_dir / "preference_pairs.jsonl", pairs)
        success_count = _write_jsonl(self.output_dir / "success_only.jsonl", success_rows)
        summary = self._summary(records, by_key, pairs, success_rows, skipped)
        summary["preference_pairs"] = pair_count
        summary["success_only"] = success_count
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

    def _load_records(self) -> list[EpisodeRecord]:
        records: list[EpisodeRecord] = []
        for root in self.rollout_roots:
            if not root.is_dir():
                raise FileNotFoundError(root)
            for path in _episode_files(root):
                result_seed = None
                result_path = path.with_name(path.name.removesuffix("_episodes.jsonl") + ".json")
                if result_path.exists():
                    result_seed = _read_json(result_path).get("seed")
                seed = _infer_seed(root, result_seed)
                with path.open() as f:
                    for line in f:
                        if not line.strip():
                            continue
                        episode = json.loads(line)
                        artifacts = episode.get("artifacts") or episode.get("files") or {}
                        rollout_dir = artifacts.get("rollout_dir")
                        trajectory = artifacts.get("trajectory")
                        if rollout_dir is None and trajectory:
                            rollout_dir = str(Path(trajectory).parent)
                        if rollout_dir is None:
                            raise ValueError(f"Episode lacks rollout_dir/trajectory in {path}")
                        rollout_rel = Path(rollout_dir)
                        metadata_path = root / rollout_rel / "metadata.json"
                        metadata = _read_json(metadata_path)
                        task_meta = metadata["task_metadata"]
                        trajectory_rel = Path(trajectory) if trajectory else rollout_rel / "trajectory.npz"
                        records.append(
                            EpisodeRecord(
                                root=root,
                                rollout_dir=rollout_rel,
                                trajectory_path=trajectory_rel,
                                metadata_path=rollout_rel / "metadata.json",
                                suite=str(task_meta["suite"]),
                                task_id=int(task_meta["task_id"]),
                                episode_idx=int(task_meta["episode_idx"]),
                                task=str(task_meta["name"]),
                                seed=seed,
                                success=bool(episode.get("metrics", {}).get("success")),
                                steps=episode.get("steps") or metadata.get("steps"),
                            )
                        )
        return records

    def _select_pair(self, success: EpisodeRecord, failure: EpisodeRecord) -> tuple[dict[str, Any] | None, str | None]:
        with np.load(success.root / success.trajectory_path) as traj_s, np.load(
            failure.root / failure.trajectory_path
        ) as traj_f:
            required = {"state", self.action_key}
            if not required.issubset(traj_s.files) or not required.issubset(traj_f.files):
                return None, "missing_state_or_action"

            max_len = min(len(traj_s["state"]), len(traj_f["state"]), len(traj_s[self.action_key]), len(traj_f[self.action_key]))
            if max_len <= self.min_target_offset + 1:
                return None, "too_short"

            states_s = traj_s["state"][:max_len].astype(np.float32)
            states_f = traj_f["state"][:max_len].astype(np.float32)
            actions_s = traj_s[self.action_key][:max_len].astype(np.float32)
            actions_f = traj_f[self.action_key][:max_len].astype(np.float32)

            state_components = _component_state_distances(states_s, states_f)
            action_components = _component_action_distances(actions_s, actions_f, grip_cap=self.action_grip_cap)
            state_scalar = _weighted_minmax_scalar(state_components, self.state_weights)
            action_scalar = _weighted_minmax_scalar(action_components, self.action_weights)

            starts_s = set(_chunk_start_steps(traj_s, max_len))
            starts_f = set(_chunk_start_steps(traj_f, max_len))
            starts = sorted(starts_s & starts_f)
            candidates = []
            for c in starts:
                chunk_size = min(
                    _step_chunk_size(traj_s, c, self.default_chunk_size),
                    _step_chunk_size(traj_f, c, self.default_chunk_size),
                )
                max_offset = min(chunk_size - self.target_window_size, max_len - c - self.target_window_size)
                if self.max_target_offset is not None:
                    max_offset = min(max_offset, self.max_target_offset)
                if max_offset < self.min_target_offset:
                    continue
                target_steps = [
                    c + k
                    for k in range(self.min_target_offset, max_offset + 1)
                    if _step_chunk_start(traj_s, c + k) == c
                    and _step_chunk_start(traj_f, c + k) == c
                    and _step_chunk_start(traj_s, c + k + self.target_window_size - 1) == c
                    and _step_chunk_start(traj_f, c + k + self.target_window_size - 1) == c
                ]
                if not target_steps:
                    continue
                hi = min(max_len, c + chunk_size)
                future_growth = float(np.max(state_scalar[c + 1 : hi]) - state_scalar[c]) if c + 1 < hi else 0.0
                action_saliency = float(np.max(action_scalar[target_steps]))
                candidates.append(
                    {
                        "chunk_start": c,
                        "chunk_size": chunk_size,
                        "target_steps": target_steps,
                        "current_gap": float(state_scalar[c]),
                        "future_growth": future_growth,
                        "action_saliency": action_saliency,
                    }
                )

            if not candidates:
                return None, "no_valid_chunk_candidate"

            current = np.asarray([x["current_gap"] for x in candidates], dtype=np.float64)
            future = np.asarray([x["future_growth"] for x in candidates], dtype=np.float64)
            action = np.asarray([x["action_saliency"] for x in candidates], dtype=np.float64)
            policy_scores = (
                self.policy_state_weight * _low_rank(current)
                + self.policy_future_weight * _high_rank(future)
                + self.policy_action_weight * _high_rank(action)
            )
            policy_idx = int(np.argsort(-policy_scores, kind="mergesort")[0])
            policy = candidates[policy_idx]
            c = int(policy["chunk_start"])
            target = self._select_target_step(
                c=c,
                target_steps=policy["target_steps"],
                state_scalar=state_scalar,
                action_scalar=action_scalar,
                max_len=max_len,
            )
            if target is None:
                return None, "no_valid_target_step"
            t, target_debug = target
            offset = int(t - c)

            state_pos = float(state_components["pos"][t])
            state_rot = float(state_components["rot"][t])
            state_grip = float(state_components["grip"][t])
            action_pos = float(action_components["pos"][t])
            action_rot = float(action_components["rot"][t])
            action_grip = float(action_components["grip_raw"][t])
            non_grip_action = action_pos + action_rot
            gripper_only = bool(action_grip > self.action_grip_cap * 0.5 and non_grip_action < 1e-3)

            row = {
                "sample_type": "chunk_preference_pair",
                "selection": "chunk_internal_rank_top1",
                "suite": success.suite,
                "task_id": success.task_id,
                "task": success.task,
                "episode_idx": success.episode_idx,
                "success_seed": success.seed,
                "failure_seed": failure.seed,
                "action_key": self.action_key,
                "policy_obs_step": c,
                "chunk_start_step": c,
                "target_step": int(t),
                "action_offset": offset,
                "chunk_size": int(policy["chunk_size"]),
                "target_window_size": self.target_window_size,
                "vla_input": self._sample_ref(failure, obs_step=c, state_step=c),
                "positive": self._sample_ref(success, obs_step=t, state_step=t, action_step=t),
                "negative": self._sample_ref(failure, obs_step=t, state_step=t, action_step=t),
                "idm_input": {
                    "failure_obs_step": int(t),
                    "success_obs_step": int(t),
                    "failure_state_step": int(t),
                    "success_state_step": int(t),
                    "failure": self._sample_ref(failure, obs_step=t, state_step=t),
                    "success": self._sample_ref(success, obs_step=t, state_step=t),
                },
                "selection_debug": {
                    "policy_score": float(policy_scores[policy_idx]),
                    "policy_current_gap": float(policy["current_gap"]),
                    "policy_future_growth": float(policy["future_growth"]),
                    "policy_action_saliency": float(policy["action_saliency"]),
                    "policy_candidate_count": int(len(candidates)),
                    "target_score": float(target_debug["score"]),
                    "target_candidate_count": int(len(policy["target_steps"])),
                    "target_state_growth": float(target_debug["state_growth"]),
                    "target_step_growth": float(target_debug["step_growth"]),
                    "target_local_future_growth": float(target_debug["local_future_growth"]),
                    "target_action_saliency": float(target_debug["action_saliency"]),
                    "state_gap_at_policy": float(state_scalar[c]),
                    "state_gap_at_target": float(state_scalar[t]),
                    "action_gap_at_target": float(action_scalar[t]),
                    "state_pos_distance": state_pos,
                    "state_rot_distance": state_rot,
                    "state_grip_distance": state_grip,
                    "action_pos_distance": action_pos,
                    "action_rot_distance": action_rot,
                    "action_grip_distance": action_grip,
                    "gripper_only_action_gap": gripper_only,
                },
            }
            return row, None

    def _select_target_step(
        self,
        *,
        c: int,
        target_steps: list[int],
        state_scalar: np.ndarray,
        action_scalar: np.ndarray,
        max_len: int,
    ) -> tuple[int, dict[str, float]] | None:
        if not target_steps:
            return None
        state_growth = np.asarray([state_scalar[t] - state_scalar[c] for t in target_steps], dtype=np.float64)
        step_growth = np.asarray(
            [state_scalar[t] - state_scalar[max(c, t - 1)] for t in target_steps],
            dtype=np.float64,
        )
        local_future_growth = []
        for t in target_steps:
            hi = min(max_len, t + self.local_future_steps + 1)
            local_future_growth.append(float(np.max(state_scalar[t:hi]) - state_scalar[c]) if t < hi else 0.0)
        local_future_growth_arr = np.asarray(local_future_growth, dtype=np.float64)
        action_saliency = np.asarray([action_scalar[t] for t in target_steps], dtype=np.float64)
        offset = np.asarray([t - c for t in target_steps], dtype=np.float64)
        scores = (
            self.target_growth_weight * _high_rank(state_growth)
            + self.target_step_growth_weight * _high_rank(step_growth)
            + self.target_future_weight * _high_rank(local_future_growth_arr)
            + self.target_action_weight * _high_rank(action_saliency)
            + self.target_earliness_weight * _low_rank(offset)
        )
        idx = int(np.argsort(-scores, kind="mergesort")[0])
        return int(target_steps[idx]), {
            "score": float(scores[idx]),
            "state_growth": float(state_growth[idx]),
            "step_growth": float(step_growth[idx]),
            "local_future_growth": float(local_future_growth_arr[idx]),
            "action_saliency": float(action_saliency[idx]),
        }

    def _build_success_only(self, records: list[EpisodeRecord], *, target_count: int) -> list[dict[str, Any]]:
        if target_count <= 0 or self.success_samples_per_rollout <= 0:
            return []

        candidates_by_task: dict[tuple[str, int], deque[dict[str, Any]]] = defaultdict(deque)
        rng = random.Random(self.random_seed)
        for record in records:
            if not record.success:
                continue
            steps = self._success_steps(record)
            rng.shuffle(steps)
            for step in sorted(steps[: self.success_samples_per_rollout]):
                candidates_by_task[(record.suite, record.task_id)].append(
                    {
                        "sample_type": "success_only",
                        "suite": record.suite,
                        "task_id": record.task_id,
                        "task": record.task,
                        "episode_idx": record.episode_idx,
                        "seed": record.seed,
                        "action_key": self.action_key,
                        "policy_obs_step": int(step),
                        "chunk_start_step": int(step),
                        "action_offset": 0,
                        "vla_input": self._sample_ref(record, obs_step=step, state_step=step),
                        "target": self._sample_ref(record, obs_step=step, state_step=step, action_step=step),
                    }
                )

        keys = sorted(candidates_by_task)
        rng.shuffle(keys)
        selected: list[dict[str, Any]] = []
        while keys and len(selected) < target_count:
            next_keys = []
            for key in keys:
                queue = candidates_by_task[key]
                if queue and len(selected) < target_count:
                    selected.append(queue.popleft())
                if queue:
                    next_keys.append(key)
            keys = next_keys
        return selected

    def _success_steps(self, record: EpisodeRecord) -> list[int]:
        with np.load(record.root / record.trajectory_path) as traj:
            length = min(len(traj["state"]), len(traj[self.action_key])) if "state" in traj.files and self.action_key in traj.files else 0
            starts = _chunk_start_steps(traj, length)
            if not starts:
                return []
            if len(starts) <= self.success_samples_per_rollout:
                return starts
            positions = np.linspace(0, len(starts) - 1, num=self.success_samples_per_rollout)
            return sorted({int(starts[int(round(p))]) for p in positions})

    def _sample_ref(
        self,
        record: EpisodeRecord,
        *,
        obs_step: int,
        state_step: int | None = None,
        action_step: int | None = None,
    ) -> dict[str, Any]:
        ref = {
            "root": str(record.root),
            "rollout_dir": record.rollout_rel,
            "trajectory": record.trajectory_rel,
            "metadata": str(record.metadata_path),
            "obs_step": int(obs_step),
            "state_step": int(obs_step if state_step is None else state_step),
            "seed": record.seed,
            "success": record.success,
        }
        if action_step is not None:
            ref["action_step"] = int(action_step)
        return ref

    def _summary(
        self,
        records: list[EpisodeRecord],
        by_key: dict[tuple[str, int, int], list[EpisodeRecord]],
        pairs: list[dict[str, Any]],
        success_rows: list[dict[str, Any]],
        skipped: Counter,
    ) -> dict[str, Any]:
        seed_counts = Counter(r.seed for r in records)
        seed_success = Counter(r.seed for r in records if r.success)
        key_distribution = Counter()
        pair_candidates = 0
        for rows in by_key.values():
            s = sum(r.success for r in rows)
            f = len(rows) - s
            key_distribution[f"S{s}_F{f}"] += 1
            pair_candidates += s * f

        pairs_by_suite = Counter(row["suite"] for row in pairs)
        pairs_by_task = Counter((row["suite"], row["task_id"], row["task"]) for row in pairs)
        offsets = Counter(row["action_offset"] for row in pairs)
        gripper_only = sum(1 for row in pairs if row["selection_debug"].get("gripper_only_action_gap"))
        state_growth = np.asarray([row["selection_debug"]["target_state_growth"] for row in pairs], dtype=np.float64)
        action_gap = np.asarray([row["selection_debug"]["action_gap_at_target"] for row in pairs], dtype=np.float64)

        def stats(values: np.ndarray) -> dict[str, float] | None:
            if len(values) == 0:
                return None
            return {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p10": float(np.percentile(values, 10)),
                "p90": float(np.percentile(values, 90)),
            }

        return {
            "rollout_roots": [str(p) for p in self.rollout_roots],
            "action_key": self.action_key,
            "records": len(records),
            "episode_keys": len(by_key),
            "seed_counts": {str(k): v for k, v in sorted(seed_counts.items())},
            "seed_success": {str(k): v for k, v in sorted(seed_success.items())},
            "key_success_failure_distribution": dict(sorted(key_distribution.items())),
            "pair_candidates": pair_candidates,
            "skipped": dict(skipped),
            "selection": {
                "default_chunk_size": self.default_chunk_size,
                "min_target_offset": self.min_target_offset,
                "max_target_offset": self.max_target_offset,
                "target_window_size": self.target_window_size,
                "local_future_steps": self.local_future_steps,
                "policy_weights": {
                    "state": self.policy_state_weight,
                    "future": self.policy_future_weight,
                    "action": self.policy_action_weight,
                },
                "target_weights": {
                    "growth": self.target_growth_weight,
                    "step_growth": self.target_step_growth_weight,
                    "future": self.target_future_weight,
                    "action": self.target_action_weight,
                    "earliness": self.target_earliness_weight,
                },
                "state_component_weights": self.state_weights,
                "action_component_weights": self.action_weights,
                "action_grip_cap": self.action_grip_cap,
            },
            "pairs_by_suite": dict(sorted(pairs_by_suite.items())),
            "top_pair_tasks": [
                {"suite": suite, "task_id": task_id, "task": task, "count": count}
                for (suite, task_id, task), count in pairs_by_task.most_common(25)
            ],
            "action_offset_distribution": {str(k): v for k, v in sorted(offsets.items())},
            "gripper_only_action_gap": {
                "count": gripper_only,
                "ratio": float(gripper_only / len(pairs)) if pairs else 0.0,
            },
            "target_state_growth_stats": stats(state_growth),
            "target_action_gap_stats": stats(action_gap),
            "success_ratio": self.success_ratio,
            "success_samples_per_rollout": self.success_samples_per_rollout,
            "random_seed": self.random_seed,
        }
