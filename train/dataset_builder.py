"""Build success-only and triplet training manifests from rollout artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUCCESS_PROGRESS_POINTS = (0.3, 0.6, 0.85)


@dataclass(frozen=True)
class RolloutRecord:
    """Metadata for one rollout directory."""

    rollout_dir: Path
    trajectory_path: Path
    task: str
    success: bool
    completed_subtasks: int
    current_subtask_table: tuple[str, ...]
    language_table: tuple[str, ...]
    episode_steps: int


@dataclass(frozen=True)
class SegmentRef:
    """One contiguous subtask segment inside a rollout."""

    rollout: RolloutRecord
    subtask_name: str
    subtask_idx: int
    raw_indices: tuple[int, ...]


@dataclass(frozen=True)
class TripletMatch:
    """Matched success/failure segments plus divergence metadata."""

    success_segment: SegmentRef
    failure_segment: SegmentRef
    success_resampled: tuple[int, ...]
    failure_resampled: tuple[int, ...]
    aligned_start: int
    min_index: int
    divergence_index: int
    min_distance: float


class RolloutDatasetBuilder:
    """Construct training manifests from rollout artifacts."""

    def __init__(
        self,
        results_dir: str | Path,
        *,
        success_segment_cap: int = 30,
        success_progress_points: tuple[float, ...] = SUCCESS_PROGRESS_POINTS,
        resample_points: int = 41,
        align_threshold: float = 0.04,
        align_run: int = 3,
        divergence_delta: float = 0.015,
        divergence_run: int = 2,
        triplet_window: int = 5,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.rollouts_dir = self.results_dir / "rollouts"
        self.success_segment_cap = success_segment_cap
        self.success_progress_points = success_progress_points
        self.resample_points = resample_points
        self.align_threshold = align_threshold
        self.align_run = align_run
        self.divergence_delta = divergence_delta
        self.divergence_run = divergence_run
        self.triplet_window = triplet_window

        self._npz_cache: dict[Path, Any] = {}
        self._array_cache: dict[tuple[Path, str], np.ndarray] = {}

    def build(self, output_dir: str | Path) -> dict[str, Any]:
        """Build success-only and triplet manifests and write them to *output_dir*."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        rollouts = self._load_rollouts()
        segments = self._extract_segments(rollouts)

        success_entries, success_stats = self._build_success_only_entries(segments)
        triplet_entries, triplet_stats = self._build_triplet_entries(segments)

        self._write_jsonl(output_path / "success_only.jsonl", success_entries)
        self._write_jsonl(output_path / "triplets.jsonl", triplet_entries)

        summary = {
            "results_dir": str(self.results_dir.resolve()),
            "num_rollouts": len(rollouts),
            "config": {
                "success_segment_cap": self.success_segment_cap,
                "success_progress_points": list(self.success_progress_points),
                "resample_points": self.resample_points,
                "align_threshold": self.align_threshold,
                "align_run": self.align_run,
                "divergence_delta": self.divergence_delta,
                "divergence_run": self.divergence_run,
                "triplet_window": self.triplet_window,
            },
            "success_only": success_stats,
            "triplets": triplet_stats,
        }
        (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def _load_rollouts(self) -> list[RolloutRecord]:
        records: list[RolloutRecord] = []
        for metadata_path in sorted(self.rollouts_dir.glob("*/metadata.json")):
            metadata = json.loads(metadata_path.read_text())
            rollout_dir = metadata_path.parent
            trajectory_path = rollout_dir / "trajectory.npz"
            if not trajectory_path.exists():
                continue
            metrics = metadata.get("metrics", {})
            records.append(
                RolloutRecord(
                    rollout_dir=rollout_dir,
                    trajectory_path=trajectory_path,
                    task=str(metadata.get("task", rollout_dir.name)),
                    success=bool(metrics.get("success", False)),
                    completed_subtasks=int(metrics.get("completed_subtasks", 0)),
                    current_subtask_table=tuple(metadata.get("current_subtask_table", [])),
                    language_table=tuple(metadata.get("language_table", [])),
                    episode_steps=int(metadata.get("steps", 0)),
                )
            )
        return records

    def _extract_segments(self, rollouts: list[RolloutRecord]) -> dict[str, list[SegmentRef]]:
        by_subtask: dict[str, list[SegmentRef]] = {}
        for rollout in rollouts:
            subtask_idx = self._load_array(rollout.trajectory_path, "subtask_idx")
            for idx, subtask_name in enumerate(rollout.current_subtask_table):
                raw_indices = np.where(subtask_idx == idx)[0]
                if len(raw_indices) == 0:
                    continue
                ref = SegmentRef(
                    rollout=rollout,
                    subtask_name=subtask_name,
                    subtask_idx=idx,
                    raw_indices=tuple(int(x) for x in raw_indices.tolist()),
                )
                by_subtask.setdefault(subtask_name, []).append(ref)
        return by_subtask

    def _build_success_only_entries(self, segments: dict[str, list[SegmentRef]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        per_subtask: dict[str, dict[str, int]] = {}

        for subtask_name, refs in sorted(segments.items()):
            success_refs = [ref for ref in refs if ref.rollout.success]
            if not success_refs:
                continue
            selected = self._select_evenly_spaced(success_refs, min(len(success_refs), self.success_segment_cap))
            step_count = 0
            for ref in selected:
                for raw_step in self._sample_progress_steps(ref.raw_indices, self.success_progress_points):
                    entries.append(
                        {
                            "sample_type": "success_only",
                            "subtask": subtask_name,
                            "anchor_rollout": str(ref.rollout.rollout_dir),
                            "anchor_step": raw_step,
                            "positive_rollout": str(ref.rollout.rollout_dir),
                            "positive_step": raw_step,
                            "negative_rollout": None,
                            "negative_step": None,
                            "progress_points": list(self.success_progress_points),
                        }
                    )
                    step_count += 1
            per_subtask[subtask_name] = {
                "available_segments": len(success_refs),
                "selected_segments": len(selected),
                "num_samples": step_count,
            }

        stats = {
            "num_samples": len(entries),
            "num_subtasks": len(per_subtask),
            "per_subtask": per_subtask,
        }
        return entries, stats

    def _build_triplet_entries(self, segments: dict[str, list[SegmentRef]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        matched_failures = 0
        unmatched_failures: list[str] = []
        per_subtask: dict[str, int] = {}

        failure_segments = []
        for refs in segments.values():
            for ref in refs:
                if ref.rollout.success:
                    continue
                if ref.subtask_idx == ref.rollout.completed_subtasks:
                    failure_segments.append(ref)

        for failure_ref in sorted(failure_segments, key=lambda ref: ref.rollout.rollout_dir.name):
            success_candidates = [ref for ref in segments.get(failure_ref.subtask_name, []) if ref.rollout.success]
            if not success_candidates:
                unmatched_failures.append(failure_ref.rollout.rollout_dir.name)
                continue

            match = self._find_best_triplet_match(failure_ref, success_candidates)
            if match is None:
                unmatched_failures.append(failure_ref.rollout.rollout_dir.name)
                continue

            matched_failures += 1
            window_indices = self._window_indices(match.divergence_index, len(match.success_resampled), self.triplet_window)
            step_entries = 0
            seen_step_pairs: set[tuple[int, int]] = set()
            for norm_idx in window_indices:
                success_step = match.success_resampled[norm_idx]
                failure_step = match.failure_resampled[norm_idx]
                step_pair = (success_step, failure_step)
                if step_pair in seen_step_pairs:
                    continue
                seen_step_pairs.add(step_pair)
                entries.append(
                    {
                        "sample_type": "triplet",
                        "subtask": failure_ref.subtask_name,
                        "anchor_rollout": str(match.success_segment.rollout.rollout_dir),
                        "anchor_step": success_step,
                        "positive_rollout": str(match.success_segment.rollout.rollout_dir),
                        "positive_step": success_step,
                        "negative_rollout": str(match.failure_segment.rollout.rollout_dir),
                        "negative_step": failure_step,
                        "match_rollout_success": match.success_segment.rollout.rollout_dir.name,
                        "match_rollout_failure": match.failure_segment.rollout.rollout_dir.name,
                        "aligned_start_index": match.aligned_start,
                        "min_index": match.min_index,
                        "divergence_index": match.divergence_index,
                        "divergence_distance": round(float(self._distance_at(match, norm_idx)), 6),
                        "window_size": self.triplet_window,
                    }
                )
                step_entries += 1
            per_subtask[failure_ref.subtask_name] = per_subtask.get(failure_ref.subtask_name, 0) + step_entries

        stats = {
            "num_samples": len(entries),
            "num_failed_segments": len(failure_segments),
            "matched_failed_segments": matched_failures,
            "unmatched_failed_segments": len(unmatched_failures),
            "unmatched_rollouts": unmatched_failures,
            "per_subtask": per_subtask,
        }
        return entries, stats

    def _find_best_triplet_match(self, failure_ref: SegmentRef, success_candidates: list[SegmentRef]) -> TripletMatch | None:
        best: TripletMatch | None = None
        best_score: tuple[float, float] | None = None
        for success_ref in success_candidates:
            match = self._match_segments(success_ref, failure_ref)
            if match is None:
                continue
            score = (match.min_distance, float(abs(match.divergence_index - match.min_index)))
            if best_score is None or score < best_score:
                best = match
                best_score = score
        return best

    def _match_segments(self, success_ref: SegmentRef, failure_ref: SegmentRef) -> TripletMatch | None:
        success_positions = self._eef_positions(success_ref)
        failure_positions = self._eef_positions(failure_ref)
        success_resampled = self._resample_segment_indices(success_ref.raw_indices, self.resample_points)
        failure_resampled = self._resample_segment_indices(failure_ref.raw_indices, self.resample_points)

        success_curve = success_positions[np.asarray(success_resampled, dtype=np.int32)]
        failure_curve = failure_positions[np.asarray(failure_resampled, dtype=np.int32)]
        distances = np.linalg.norm(success_curve - failure_curve, axis=1)

        aligned_start = self._find_aligned_start(distances)
        if aligned_start is None:
            return None

        min_index = aligned_start + int(np.argmin(distances[aligned_start:]))
        divergence_index = self._find_divergence_index(distances, min_index)
        if divergence_index is None:
            return None

        return TripletMatch(
            success_segment=success_ref,
            failure_segment=failure_ref,
            success_resampled=tuple(int(x) for x in success_resampled),
            failure_resampled=tuple(int(x) for x in failure_resampled),
            aligned_start=aligned_start,
            min_index=min_index,
            divergence_index=divergence_index,
            min_distance=float(distances[min_index]),
        )

    def _find_aligned_start(self, distances: np.ndarray) -> int | None:
        max_start = len(distances) - self.align_run + 1
        for start in range(max_start):
            window = distances[start : start + self.align_run]
            if np.all(window <= self.align_threshold):
                return start
        return None

    def _find_divergence_index(self, distances: np.ndarray, min_index: int) -> int | None:
        limit = len(distances) - self.divergence_run + 1
        baseline = float(distances[min_index])
        for idx in range(min_index + 1, limit):
            window = distances[idx : idx + self.divergence_run]
            if not np.all(np.diff(window) >= 0):
                continue
            if float(window[-1]) - baseline >= self.divergence_delta:
                return idx
        return None

    def _sample_progress_steps(self, raw_indices: tuple[int, ...], progress_points: tuple[float, ...]) -> list[int]:
        last = len(raw_indices) - 1
        chosen: list[int] = []
        seen: set[int] = set()
        for progress in progress_points:
            offset = int(round(progress * last))
            raw_step = int(raw_indices[min(max(offset, 0), last)])
            if raw_step in seen:
                continue
            seen.add(raw_step)
            chosen.append(raw_step)
        return chosen

    def _resample_segment_indices(self, raw_indices: tuple[int, ...], num_points: int) -> list[int]:
        if len(raw_indices) == 1:
            return [int(raw_indices[0])] * num_points
        positions = np.linspace(0, len(raw_indices) - 1, num_points)
        sampled = [int(raw_indices[int(round(pos))]) for pos in positions]
        return sampled

    def _window_indices(self, center: int, length: int, size: int) -> list[int]:
        radius = size // 2
        raw = list(range(center - radius, center + radius + 1))
        clipped = [min(max(idx, 0), length - 1) for idx in raw]
        unique: list[int] = []
        for idx in clipped:
            if idx not in unique:
                unique.append(idx)
        return unique

    def _select_evenly_spaced(self, refs: list[SegmentRef], count: int) -> list[SegmentRef]:
        if count >= len(refs):
            return list(refs)
        positions = np.linspace(0, len(refs) - 1, count)
        seen: set[int] = set()
        selected: list[SegmentRef] = []
        for pos in positions:
            idx = int(round(pos))
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(refs[idx])
        return selected

    def _distance_at(self, match: TripletMatch, norm_idx: int) -> float:
        success_positions = self._eef_positions(match.success_segment)
        failure_positions = self._eef_positions(match.failure_segment)
        success_step = match.success_resampled[norm_idx]
        failure_step = match.failure_resampled[norm_idx]
        return float(np.linalg.norm(success_positions[success_step] - failure_positions[failure_step]))

    def _eef_positions(self, segment: SegmentRef) -> np.ndarray:
        states = self._load_array(segment.rollout.trajectory_path, "state")
        return states[:, :3]

    def _load_array(self, trajectory_path: Path, key: str) -> np.ndarray:
        cache_key = (trajectory_path, key)
        if cache_key not in self._array_cache:
            npz = self._load_npz(trajectory_path)
            self._array_cache[cache_key] = np.asarray(npz[key])
        return self._array_cache[cache_key]

    def _load_npz(self, trajectory_path: Path) -> Any:
        if trajectory_path not in self._npz_cache:
            self._npz_cache[trajectory_path] = np.load(trajectory_path, allow_pickle=False)
        return self._npz_cache[trajectory_path]

    def _write_jsonl(self, path: Path, entries: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")
