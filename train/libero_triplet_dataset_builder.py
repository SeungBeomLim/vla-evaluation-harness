"""Build LIBERO same-initial-state triplet and success-only manifests.

The builder consumes benchmark rollout directories collected from multiple
stochastic GR00T seeds. Rollouts are grouped by `(suite, task_id, episode_idx)`.
For each group, every success rollout is paired with every failure rollout.

For each success/failure pair, the divergence chunk is selected with a
threshold-free rank score:

    score = state_weight * rank_low(d_state)
          + action_weight * rank_high(d_action)
          + future_weight * rank_high(future_gap)

where `future_gap` is the increase in state divergence over the next N chunks.
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


def _chunk_start_steps(traj: np.lib.npyio.NpzFile) -> list[int]:
    if "action_is_chunk_start" in traj.files:
        starts = np.flatnonzero(traj["action_is_chunk_start"].astype(bool))
        if len(starts):
            return [int(x) for x in starts]
    if "action_chunk_start_step" in traj.files:
        return sorted({int(x) for x in traj["action_chunk_start_step"]})
    return list(range(0, len(traj["state"]), 16))


def _component_distances(values_s: np.ndarray, values_f: np.ndarray) -> dict[str, np.ndarray]:
    state_s = values_s[:, :8]
    state_f = values_f[:, :8]
    return {
        "pos": _norm(state_s[:, 0:3] - state_f[:, 0:3]),
        "rot": _norm(state_s[:, 3:6] - state_f[:, 3:6]),
        "grip": _norm(state_s[:, 6:] - state_f[:, 6:]),
    }


def _action_component_distances(actions_s: np.ndarray, actions_f: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "pos": _norm(actions_s[:, 0:3] - actions_f[:, 0:3]),
        "rot": _norm(actions_s[:, 3:6] - actions_f[:, 3:6]),
        "grip": np.abs(actions_s[:, 6] - actions_f[:, 6]).astype(np.float64),
    }


def _avg_rank_low(components: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean([_low_rank(v) for v in components.values()], axis=0)


def _avg_rank_high(components: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean([_high_rank(v) for v in components.values()], axis=0)


def _avg_minmax(components: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean([_minmax(v) for v in components.values()], axis=0)


class LiberoTripletDatasetBuilder:
    def __init__(
        self,
        *,
        rollout_roots: list[Path],
        output_dir: Path,
        action_key: str = "env_action",
        future_chunks: int = 3,
        top_k: int = 1,
        state_weight: float = 0.4,
        action_weight: float = 0.4,
        future_weight: float = 0.2,
        success_ratio: float = 1.5,
        success_samples_per_rollout: int = 3,
        random_seed: int = 0,
    ) -> None:
        self.rollout_roots = [p.resolve() for p in rollout_roots]
        self.output_dir = output_dir
        self.action_key = action_key
        self.future_chunks = max(0, int(future_chunks))
        self.top_k = max(1, int(top_k))
        self.state_weight = float(state_weight)
        self.action_weight = float(action_weight)
        self.future_weight = float(future_weight)
        self.success_ratio = max(0.0, float(success_ratio))
        self.success_samples_per_rollout = max(0, int(success_samples_per_rollout))
        self.random_seed = random_seed

    def build(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = self._load_records()
        by_key: dict[tuple[str, int, int], list[EpisodeRecord]] = defaultdict(list)
        for record in records:
            by_key[record.key].append(record)

        triplets: list[dict[str, Any]] = []
        skipped = Counter()
        for key in sorted(by_key):
            successes = [r for r in by_key[key] if r.success]
            failures = [r for r in by_key[key] if not r.success]
            if not successes or not failures:
                continue
            for success in successes:
                for failure in failures:
                    selected = self._select_triplet_chunks(success, failure)
                    if not selected:
                        skipped["no_candidate_chunks"] += 1
                        continue
                    triplets.extend(selected)

        success_rows = self._build_success_only(records, target_count=round(len(triplets) * self.success_ratio))

        triplet_count = _write_jsonl(self.output_dir / "triplets.jsonl", triplets)
        success_count = _write_jsonl(self.output_dir / "success_only.jsonl", success_rows)

        summary = self._summary(records, by_key, triplets, success_rows, skipped)
        summary["triplets"] = triplet_count
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

    def _select_triplet_chunks(self, success: EpisodeRecord, failure: EpisodeRecord) -> list[dict[str, Any]]:
        with np.load(success.root / success.trajectory_path) as traj_s, np.load(
            failure.root / failure.trajectory_path
        ) as traj_f:
            if "state" not in traj_s.files or "state" not in traj_f.files:
                return []
            if self.action_key not in traj_s.files or self.action_key not in traj_f.files:
                return []

            starts_s = set(_chunk_start_steps(traj_s))
            starts_f = set(_chunk_start_steps(traj_f))
            max_len = min(len(traj_s["state"]), len(traj_f["state"]))
            starts = sorted(t for t in starts_s & starts_f if 0 <= t < max_len)
            if not starts:
                return []

            idx = np.asarray(starts, dtype=np.int64)
            states_s = traj_s["state"][idx]
            states_f = traj_f["state"][idx]
            actions_s = traj_s[self.action_key][idx]
            actions_f = traj_f[self.action_key][idx]

            state_components = _component_distances(states_s, states_f)
            action_components = _action_component_distances(actions_s, actions_f)

            state_rank = _avg_rank_low(state_components)
            action_rank = _avg_rank_high(action_components)
            state_scalar = _avg_minmax(state_components)
            future_gap = self._future_gap(state_scalar)
            future_rank = _high_rank(future_gap)

            score = (
                self.state_weight * state_rank
                + self.action_weight * action_rank
                + self.future_weight * future_rank
            )
            order = np.argsort(-score, kind="mergesort")[: self.top_k]
            max_score = max(1e-12, self.state_weight + self.action_weight + self.future_weight)

            rows = []
            for rank_idx, pos in enumerate(order):
                step = int(idx[pos])
                rows.append(
                    {
                        "sample_type": "triplet",
                        "selection": "rank_top1" if self.top_k == 1 else "rank_topk",
                        "suite": success.suite,
                        "task_id": success.task_id,
                        "task": success.task,
                        "episode_idx": success.episode_idx,
                        "success_seed": success.seed,
                        "failure_seed": failure.seed,
                        "anchor": self._sample_ref(failure, obs_step=step, action_step=step),
                        "positive": self._sample_ref(success, obs_step=step, action_step=step),
                        "negative": self._sample_ref(failure, obs_step=step, action_step=step),
                        "action_key": self.action_key,
                        "divergence": {
                            "chunk_start_step": step,
                            "rank_order": int(rank_idx),
                            "score": float(score[pos]),
                            "confidence": float(score[pos] / max_score),
                            "state_rank": float(state_rank[pos]),
                            "action_rank": float(action_rank[pos]),
                            "future_rank": float(future_rank[pos]),
                            "future_gap": float(future_gap[pos]),
                            "state_pos_distance": float(state_components["pos"][pos]),
                            "state_rot_distance": float(state_components["rot"][pos]),
                            "state_grip_distance": float(state_components["grip"][pos]),
                            "action_pos_distance": float(action_components["pos"][pos]),
                            "action_rot_distance": float(action_components["rot"][pos]),
                            "action_grip_distance": float(action_components["grip"][pos]),
                            "candidate_count": int(len(idx)),
                            "future_chunks": int(self.future_chunks),
                        },
                    }
                )
            return rows

    def _future_gap(self, state_scalar: np.ndarray) -> np.ndarray:
        n = len(state_scalar)
        out = np.zeros(n, dtype=np.float64)
        if self.future_chunks <= 0:
            return out
        for i in range(n):
            lo = i + 1
            hi = min(n, lo + self.future_chunks)
            if lo >= hi:
                out[i] = 0.0
            else:
                out[i] = float(np.mean(state_scalar[lo:hi]) - state_scalar[i])
        return out

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
                        "sample": self._sample_ref(record, obs_step=step, action_step=step),
                        "action_key": self.action_key,
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
            starts = _chunk_start_steps(traj)
            if not starts:
                return []
            if len(starts) <= self.success_samples_per_rollout:
                return starts
            positions = np.linspace(0, len(starts) - 1, num=self.success_samples_per_rollout)
            return sorted({int(starts[int(round(p))]) for p in positions})

    def _sample_ref(self, record: EpisodeRecord, *, obs_step: int, action_step: int) -> dict[str, Any]:
        return {
            "root": str(record.root),
            "rollout_dir": record.rollout_rel,
            "trajectory": record.trajectory_rel,
            "metadata": str(record.metadata_path),
            "obs_step": int(obs_step),
            "action_step": int(action_step),
            "seed": record.seed,
            "success": record.success,
        }

    def _summary(
        self,
        records: list[EpisodeRecord],
        by_key: dict[tuple[str, int, int], list[EpisodeRecord]],
        triplets: list[dict[str, Any]],
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

        triplets_by_suite = Counter(row["suite"] for row in triplets)
        triplets_by_task = Counter((row["suite"], row["task_id"], row["task"]) for row in triplets)

        return {
            "rollout_roots": [str(p) for p in self.rollout_roots],
            "action_key": self.action_key,
            "future_chunks": self.future_chunks,
            "top_k": self.top_k,
            "score_weights": {
                "state": self.state_weight,
                "action": self.action_weight,
                "future": self.future_weight,
            },
            "records": len(records),
            "episode_keys": len(by_key),
            "seed_counts": {str(k): v for k, v in sorted(seed_counts.items())},
            "seed_success": {str(k): v for k, v in sorted(seed_success.items())},
            "key_success_failure_distribution": dict(sorted(key_distribution.items())),
            "pair_candidates": pair_candidates,
            "skipped": dict(skipped),
            "triplets_by_suite": dict(sorted(triplets_by_suite.items())),
            "top_triplet_tasks": [
                {"suite": suite, "task_id": task_id, "task": task, "count": count}
                for (suite, task_id, task), count in triplets_by_task.most_common(25)
            ],
            "success_ratio": self.success_ratio,
            "success_samples_per_rollout": self.success_samples_per_rollout,
            "random_seed": self.random_seed,
        }
