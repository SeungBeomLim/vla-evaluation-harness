"""Build training manifests with gripper-aware triplet divergence detection.

This variant keeps the original EE-distance alignment logic, then augments
divergence detection with a gripper-based candidate:

1. Find an aligned region using EE position distance.
2. Find the minimum-distance point after alignment.
3. Mark a gripper mismatch at a resampled match point as a divergence candidate.
4. Confirm that candidate only if the mismatch persists for the next
   ``gripper_persistence_steps`` raw segment steps.
5. Use the earliest confirmed divergence from either EE-distance growth
   or persistent gripper mismatch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from train.legacy_xvla_calvin.dataset_builder import (
    SUCCESS_PROGRESS_POINTS,
    RolloutDatasetBuilder,
    RolloutRecord,
    SegmentRef,
    TripletMatch,
)


@dataclass(frozen=True)
class TripletMatchV2(TripletMatch):
    """Matched success/failure segments plus divergence attribution."""

    divergence_reason: str


class RolloutDatasetBuilderV2(RolloutDatasetBuilder):
    """Construct manifests with gripper-aware triplet divergence detection."""

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
        gripper_persistence_steps: int = 3,
    ) -> None:
        super().__init__(
            results_dir,
            success_segment_cap=success_segment_cap,
            success_progress_points=success_progress_points,
            resample_points=resample_points,
            align_threshold=align_threshold,
            align_run=align_run,
            divergence_delta=divergence_delta,
            divergence_run=divergence_run,
            triplet_window=triplet_window,
        )
        self.gripper_persistence_steps = gripper_persistence_steps

    def build(self, output_dir: str | Path) -> dict[str, Any]:
        summary = super().build(output_dir)
        summary["config"]["gripper_persistence_steps"] = self.gripper_persistence_steps
        Path(output_dir).joinpath("summary.json").write_text(__import__("json").dumps(summary, indent=2))
        return summary

    def _build_triplet_entries(self, segments: dict[str, list[SegmentRef]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        matched_failures = 0
        unmatched_failures: list[str] = []
        per_subtask: dict[str, int] = {}
        divergence_reason_counts: Counter[str] = Counter()

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
            divergence_reason_counts[match.divergence_reason] += 1
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
                        "divergence_reason": match.divergence_reason,
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
            "divergence_reason_counts": dict(divergence_reason_counts),
        }
        return entries, stats

    def _match_segments(self, success_ref: SegmentRef, failure_ref: SegmentRef) -> TripletMatchV2 | None:
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
        gripper_mismatch = self._gripper_mismatch_sequence(success_ref, failure_ref, success_resampled, failure_resampled)
        divergence_index, divergence_reason = self._find_divergence_index_v2(
            distances,
            gripper_mismatch,
            min_index,
            success_ref,
            failure_ref,
            success_resampled,
            failure_resampled,
        )
        if divergence_index is None or divergence_reason is None:
            return None

        return TripletMatchV2(
            success_segment=success_ref,
            failure_segment=failure_ref,
            success_resampled=tuple(int(x) for x in success_resampled),
            failure_resampled=tuple(int(x) for x in failure_resampled),
            aligned_start=aligned_start,
            min_index=min_index,
            divergence_index=divergence_index,
            min_distance=float(distances[min_index]),
            divergence_reason=divergence_reason,
        )

    def _find_divergence_index_v2(
        self,
        distances: np.ndarray,
        gripper_mismatch: np.ndarray,
        min_index: int,
        success_ref: SegmentRef,
        failure_ref: SegmentRef,
        success_resampled: list[int],
        failure_resampled: list[int],
    ) -> tuple[int | None, str | None]:
        distance_index = self._find_divergence_index(distances, min_index)
        gripper_index = self._find_gripper_divergence_index(
            gripper_mismatch,
            min_index,
            success_ref,
            failure_ref,
            success_resampled,
            failure_resampled,
        )

        if distance_index is None and gripper_index is None:
            return None, None
        if distance_index is None:
            return gripper_index, "gripper"
        if gripper_index is None:
            return distance_index, "distance"
        if gripper_index < distance_index:
            return gripper_index, "gripper"
        if distance_index < gripper_index:
            return distance_index, "distance"
        return distance_index, "both"

    def _find_gripper_divergence_index(
        self,
        gripper_mismatch: np.ndarray,
        min_index: int,
        success_ref: SegmentRef,
        failure_ref: SegmentRef,
        success_resampled: list[int],
        failure_resampled: list[int],
    ) -> int | None:
        """Return the first resampled mismatch that persists across raw segment steps."""
        for idx in range(min_index + 1, len(gripper_mismatch)):
            if not bool(gripper_mismatch[idx]):
                continue
            success_step = success_resampled[idx]
            failure_step = failure_resampled[idx]
            if self._raw_gripper_mismatch_persists(success_ref, failure_ref, success_step, failure_step):
                return idx
        return None

    def _raw_gripper_mismatch_persists(
        self,
        success_ref: SegmentRef,
        failure_ref: SegmentRef,
        success_step: int,
        failure_step: int,
    ) -> bool:
        """Check whether a mismatch persists for the next N raw steps in both segments."""
        success_indices = success_ref.raw_indices
        failure_indices = failure_ref.raw_indices
        try:
            success_offset = success_indices.index(success_step)
            failure_offset = failure_indices.index(failure_step)
        except ValueError:
            return False

        required = self.gripper_persistence_steps + 1
        if success_offset + required > len(success_indices):
            return False
        if failure_offset + required > len(failure_indices):
            return False

        success_gripper = self._gripper_binary(success_ref)
        failure_gripper = self._gripper_binary(failure_ref)
        for rel in range(required):
            s_idx = success_indices[success_offset + rel]
            f_idx = failure_indices[failure_offset + rel]
            if bool(success_gripper[s_idx]) == bool(failure_gripper[f_idx]):
                return False
        return True

    def _gripper_mismatch_sequence(
        self,
        success_ref: SegmentRef,
        failure_ref: SegmentRef,
        success_resampled: list[int],
        failure_resampled: list[int],
    ) -> np.ndarray:
        success_gripper = self._gripper_binary(success_ref)[np.asarray(success_resampled, dtype=np.int32)]
        failure_gripper = self._gripper_binary(failure_ref)[np.asarray(failure_resampled, dtype=np.int32)]
        return success_gripper != failure_gripper

    def _gripper_binary(self, segment: SegmentRef) -> np.ndarray:
        """Return the CALVIN binary open/close state from the recorded 8-D state."""
        states = self._load_array(segment.rollout.trajectory_path, "state")
        # The second gripper entry is already a binary sign in recorded artifacts.
        return np.asarray(states[:, 7] > 0.0, dtype=np.bool_)
