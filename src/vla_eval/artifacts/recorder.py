"""Step-level rollout artifact recording."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.types import Action, Observation, Task

_SAFE_NAME_RE = re.compile(r"[^\w\-.]")


def _safe_name(value: str) -> str:
    return _SAFE_NAME_RE.sub("_", value)[:180]


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value)
    except Exception:
        return None


def _stack(values: list[np.ndarray], *, dtype: Any | None = None) -> np.ndarray:
    arr = np.stack(values, axis=0)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _pad_2d_chunks(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad variable-length 2-D chunks into a dense archive-friendly array."""
    arrays = [np.asarray(v, dtype=np.float32) for v in values]
    lengths = np.asarray([a.shape[0] if a.ndim > 0 else 1 for a in arrays], dtype=np.int32)
    dims = np.asarray([a.shape[1] if a.ndim > 1 else 1 for a in arrays], dtype=np.int32)
    max_len = int(lengths.max(initial=0))
    max_dim = int(dims.max(initial=0))
    padded = np.full((len(arrays), max_len, max_dim), np.nan, dtype=np.float32)
    for i, arr in enumerate(arrays):
        arr2d = arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)
        padded[i, : arr2d.shape[0], : arr2d.shape[1]] = arr2d
    return padded, lengths, dims


def _pad_video_frames(frames: np.ndarray, block: int = 16) -> np.ndarray:
    """Pad video frames to codec-friendly dimensions without resizing content."""
    h, w = frames.shape[1:3]
    pad_h = (-h) % block
    pad_w = (-w) % block
    if pad_h == 0 and pad_w == 0:
        return frames
    return np.pad(frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _write_video(path: Path, frames: np.ndarray, fps: int = 10) -> str | None:
    """Write RGB uint8 frames to mp4. Returns an error string on failure."""
    if len(frames) == 0:
        return "no frames"

    frames = np.ascontiguousarray(_pad_video_frames(frames), dtype=np.uint8)

    # Prefer H.264 for VS Code/browser compatibility. OpenCV commonly falls
    # back to MPEG-4 Part 2 (mp4v), which many HTML5 video viewers reject.
    try:
        import imageio.v2 as imageio

        imageio.mimsave(
            path,
            frames,
            fps=fps,
            codec="libx264",
            macro_block_size=16,
            output_params=["-vf", "format=yuv420p", "-movflags", "+faststart"],
        )
        return None
    except Exception as imageio_exc:
        try:
            import cv2

            h, w = frames.shape[1:3]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open")
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return None
        except Exception as cv2_exc:
            return f"imageio: {imageio_exc}; cv2: {cv2_exc}"


class EpisodeArtifactRecorder:
    """Collect and persist one episode's step-level rollout artifacts."""

    def __init__(
        self,
        *,
        output_dir: Path,
        benchmark_name: str,
        task_name: str,
        episode_id: int,
        task: Task,
        server_info: dict[str, Any],
        fps: int = 10,
        save_video: bool = True,
        save_trajectory: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.benchmark_name = benchmark_name
        self.task_name = task_name
        self.episode_id = episode_id
        self.task = task
        self.server_info = server_info
        self.fps = fps
        self.save_video = save_video
        self.save_trajectory = save_trajectory

        rollout_name = f"{_safe_name(benchmark_name)}_{_safe_name(task_name)}_ep{episode_id}"
        self.rollout_dir = output_dir / "rollouts" / rollout_name
        self.rollout_dir.mkdir(parents=True, exist_ok=True)

        self.images: dict[str, list[np.ndarray]] = {}
        self.states: list[np.ndarray] = []
        self.server_actions: list[np.ndarray] = []
        self.env_actions: list[np.ndarray] = []
        self.model_native_actions: list[np.ndarray] = []
        self.action_chunk_id: list[int] = []
        self.action_chunk_start_step: list[int] = []
        self.action_chunk_offset: list[int] = []
        self.action_chunk_size: list[int] = []
        self.action_is_chunk_start: list[bool] = []
        self.action_from_buffer: list[bool] = []
        self.action_model_output_index: list[int] = []
        self.model_action_chunks: list[np.ndarray] = []
        self.model_native_action_chunks: list[np.ndarray] = []
        self.model_action_chunk_ids: list[int] = []
        self.model_action_chunk_start_steps: list[int] = []
        self.model_native_action_chunk_ids: list[int] = []
        self.model_native_action_chunk_start_steps: list[int] = []
        self.step_idx: list[int] = []
        self.language_ids: list[int] = []
        self.subtask_idx: list[int] = []
        self.subtask_step: list[int] = []
        self.completed_subtasks: list[int] = []
        self.subtask_done: list[bool] = []
        self.episode_done: list[bool] = []
        self.current_subtask_ids: list[int] = []

        self.language_table: list[str] = []
        self._language_to_id: dict[str, int] = {}
        self.current_subtask_table: list[str] = []
        self._subtask_to_id: dict[str, int] = {}
        self._prev_completed = 0
        self.video_errors: dict[str, str] = {}

    def _id_for(self, table: list[str], mapping: dict[str, int], value: str) -> int:
        if value not in mapping:
            mapping[value] = len(table)
            table.append(value)
        return mapping[value]

    def record_step(
        self,
        *,
        step: int,
        obs: Observation,
        action: Action,
        artifact_state: dict[str, Any],
        done: bool,
    ) -> None:
        """Record one transition: observation before action, result after action."""
        self.step_idx.append(step)

        for view, image in (obs.get("images") or {}).items():
            arr = _array_or_none(image)
            if arr is None:
                continue
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            self.images.setdefault(str(view), []).append(arr.copy())

        state = _array_or_none(obs.get("state", obs.get("states")))
        if state is not None:
            self.states.append(state.astype(np.float32, copy=True).reshape(-1))

        server_action = _array_or_none(action.get("actions", action.get("action")))
        if server_action is not None:
            self.server_actions.append(server_action.astype(np.float32, copy=True).reshape(-1))

        env_action = _array_or_none(artifact_state.get("env_action"))
        if env_action is None:
            env_action = server_action
        if env_action is not None:
            self.env_actions.append(env_action.astype(np.float32, copy=True).reshape(-1))

        native_action = _array_or_none(action.get("model_native_action", action.get("model_native_actions")))
        if native_action is not None:
            self.model_native_actions.append(native_action.astype(np.float32, copy=True).reshape(-1))

        action_metadata = action.get("action_metadata") or {}
        chunk_id = int(action_metadata.get("chunk_id", -1))
        chunk_start_step = int(action_metadata.get("chunk_start_step", step))
        chunk_offset = int(action_metadata.get("chunk_offset", 0))
        chunk_size = int(action_metadata.get("chunk_size", 1))
        is_chunk_start = bool(action_metadata.get("is_chunk_start", chunk_offset == 0))
        self.action_chunk_id.append(chunk_id)
        self.action_chunk_start_step.append(chunk_start_step)
        self.action_chunk_offset.append(chunk_offset)
        self.action_chunk_size.append(chunk_size)
        self.action_is_chunk_start.append(is_chunk_start)
        self.action_from_buffer.append(bool(action_metadata.get("from_buffer", False)))
        self.action_model_output_index.append(int(action_metadata.get("model_output_index", chunk_offset)))

        model_action_chunk = _array_or_none(action.get("model_action_chunk"))
        if model_action_chunk is not None and is_chunk_start:
            self.model_action_chunks.append(model_action_chunk.astype(np.float32, copy=True))
            self.model_action_chunk_ids.append(chunk_id)
            self.model_action_chunk_start_steps.append(chunk_start_step)

        model_native_action_chunk = _array_or_none(action.get("model_native_action_chunk"))
        if model_native_action_chunk is not None and is_chunk_start:
            self.model_native_action_chunks.append(model_native_action_chunk.astype(np.float32, copy=True))
            self.model_native_action_chunk_ids.append(chunk_id)
            self.model_native_action_chunk_start_steps.append(chunk_start_step)

        language = str(obs.get("task_description", ""))
        self.language_ids.append(self._id_for(self.language_table, self._language_to_id, language))

        completed = int(artifact_state.get("completed_subtasks", 0))
        self.completed_subtasks.append(completed)
        self.subtask_done.append(completed > self._prev_completed)
        self._prev_completed = completed
        self.subtask_idx.append(int(artifact_state.get("subtask_idx", -1)))
        self.subtask_step.append(int(artifact_state.get("subtask_step", -1)))
        self.episode_done.append(bool(done))

        current_subtask = str(artifact_state.get("current_subtask") or "")
        self.current_subtask_ids.append(self._id_for(self.current_subtask_table, self._subtask_to_id, current_subtask))

    def _trajectory_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {
            "step_idx": np.asarray(self.step_idx, dtype=np.int32),
            "language_id": np.asarray(self.language_ids, dtype=np.int32),
            "subtask_idx": np.asarray(self.subtask_idx, dtype=np.int32),
            "subtask_step": np.asarray(self.subtask_step, dtype=np.int32),
            "completed_subtasks": np.asarray(self.completed_subtasks, dtype=np.int32),
            "subtask_done": np.asarray(self.subtask_done, dtype=np.bool_),
            "episode_done": np.asarray(self.episode_done, dtype=np.bool_),
            "current_subtask_id": np.asarray(self.current_subtask_ids, dtype=np.int32),
            "action_chunk_id": np.asarray(self.action_chunk_id, dtype=np.int32),
            "action_chunk_start_step": np.asarray(self.action_chunk_start_step, dtype=np.int32),
            "action_chunk_offset": np.asarray(self.action_chunk_offset, dtype=np.int32),
            "action_chunk_size": np.asarray(self.action_chunk_size, dtype=np.int32),
            "action_is_chunk_start": np.asarray(self.action_is_chunk_start, dtype=np.bool_),
            "action_from_buffer": np.asarray(self.action_from_buffer, dtype=np.bool_),
            "action_model_output_index": np.asarray(self.action_model_output_index, dtype=np.int32),
        }
        for view, frames in self.images.items():
            arrays[f"image_{view}"] = _stack(frames, dtype=np.uint8)
        if self.states:
            arrays["state"] = _stack(self.states, dtype=np.float32)
        if self.server_actions:
            arrays["server_action"] = _stack(self.server_actions, dtype=np.float32)
        if self.env_actions:
            arrays["env_action"] = _stack(self.env_actions, dtype=np.float32)
        if self.model_native_actions:
            arrays["model_native_action"] = _stack(self.model_native_actions, dtype=np.float32)
        if self.model_action_chunks:
            chunk_arr, chunk_len, chunk_dim = _pad_2d_chunks(self.model_action_chunks)
            arrays["model_action_chunk"] = chunk_arr
            arrays["model_action_chunk_length"] = chunk_len
            arrays["model_action_chunk_dim"] = chunk_dim
            arrays["model_action_chunk_id"] = np.asarray(self.model_action_chunk_ids, dtype=np.int32)
            arrays["model_action_chunk_start_step"] = np.asarray(
                self.model_action_chunk_start_steps,
                dtype=np.int32,
            )
        if self.model_native_action_chunks:
            native_chunk_arr, native_chunk_len, native_chunk_dim = _pad_2d_chunks(self.model_native_action_chunks)
            arrays["model_native_action_chunk"] = native_chunk_arr
            arrays["model_native_action_chunk_length"] = native_chunk_len
            arrays["model_native_action_chunk_dim"] = native_chunk_dim
            arrays["model_native_action_chunk_id"] = np.asarray(self.model_native_action_chunk_ids, dtype=np.int32)
            arrays["model_native_action_chunk_start_step"] = np.asarray(
                self.model_native_action_chunk_start_steps,
                dtype=np.int32,
            )
        return arrays

    def finalize(self, episode_result: dict[str, Any]) -> dict[str, Any]:
        """Write trajectory, metadata, and per-view videos. Returns relative paths."""
        arrays = self._trajectory_arrays()

        files: dict[str, Any] = {}
        if self.save_trajectory:
            trajectory_path = self.rollout_dir / "trajectory.npz"
            np.savez_compressed(trajectory_path, **arrays)
            files["trajectory"] = str(trajectory_path.relative_to(self.output_dir))

        video_paths: dict[str, str] = {}
        if self.save_video:
            for view, frames in self.images.items():
                video_path = self.rollout_dir / f"video_{_safe_name(view)}.mp4"
                err = _write_video(video_path, _stack(frames, dtype=np.uint8), fps=self.fps)
                if err is None:
                    video_paths[view] = str(video_path.relative_to(self.output_dir))
                else:
                    self.video_errors[view] = err
            files["videos"] = video_paths

        action_spec = self.server_info.get("action_spec", {})
        metadata = {
            "benchmark": self.benchmark_name,
            "task": self.task_name,
            "episode_id": self.episode_id,
            "task_metadata": {
                k: v for k, v in self.task.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            },
            "metrics": episode_result.get("metrics", {}),
            "steps": episode_result.get("steps", len(self.step_idx)),
            "elapsed_sec": episode_result.get("elapsed_sec"),
            "language_table": self.language_table,
            "current_subtask_table": self.current_subtask_table,
            "observation_views": sorted(self.images),
            "action_spaces": {
                "model_native_action": {
                    "source": "model server native action before benchmark conversion",
                    "dims": int(arrays["model_native_action"].shape[-1]) if "model_native_action" in arrays else None,
                },
                "server_action": {
                    "source": "action payload returned by model server and consumed by benchmark",
                    "dims": int(arrays["server_action"].shape[-1]) if "server_action" in arrays else None,
                    "spec": action_spec,
                },
                "env_action": {
                    "source": "benchmark action after benchmark-specific conversion, when provided",
                    "dims": int(arrays["env_action"].shape[-1]) if "env_action" in arrays else None,
                },
                "model_action_chunk": {
                    "source": "full server action chunk attached at chunk-start steps, padded with NaN",
                    "shape": list(arrays["model_action_chunk"].shape) if "model_action_chunk" in arrays else None,
                },
                "model_native_action_chunk": {
                    "source": "full native model action chunk attached at chunk-start steps, padded with NaN",
                    "shape": (
                        list(arrays["model_native_action_chunk"].shape)
                        if "model_native_action_chunk" in arrays
                        else None
                    ),
                },
            },
            "action_provenance": {
                "source": "model server chunk metadata recorded per environment step",
                "fields": {
                    "action_chunk_id": "monotonic chunk/inference id within each episode",
                    "action_chunk_start_step": "environment step whose observation triggered the model inference",
                    "action_chunk_offset": "index inside the emitted action chunk",
                    "action_chunk_size": "number of actions in the emitted chunk",
                    "action_is_chunk_start": "true when this step triggered a fresh model inference",
                    "action_from_buffer": "true when the action was served from a previous chunk buffer",
                    "action_model_output_index": "index in the model output action sequence",
                },
            },
            "files": files,
        }
        if self.video_errors:
            metadata["video_errors"] = self.video_errors

        metadata_path = self.rollout_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

        return {
            "rollout_dir": str(self.rollout_dir.relative_to(self.output_dir)),
            "metadata": str(metadata_path.relative_to(self.output_dir)),
            **files,
            **({"video_errors": self.video_errors} if self.video_errors else {}),
        }
