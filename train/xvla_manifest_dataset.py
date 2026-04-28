"""Dataset utilities for X-VLA LoRA fine-tuning from rollout manifests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


GRIPPER_INDICES_20D = (9, 19)
CACHED_ARRAY_KEYS = frozenset({"model_native_action", "state", "language_id", "subtask_idx"})


def _matrix_to_rot6d_interleaved(mat: np.ndarray) -> np.ndarray:
    return mat[:, :2].reshape(6).astype(np.float32, copy=True)


def euler_xyz_to_rot6d_interleaved(euler: np.ndarray) -> np.ndarray:
    """Extrinsic XYZ Euler angles to interleaved 6D rotation."""
    x, y, z = float(euler[0]), float(euler[1]), float(euler[2])

    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float32,
    )
    ry = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    rz = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    mat = rz @ ry @ rx
    return _matrix_to_rot6d_interleaved(mat)


def _state_to_xvla_proprio(state: np.ndarray, dim: int = 20) -> np.ndarray:
    """Convert CALVIN 8-D state to X-VLA 20-D proprio."""
    proprio = np.zeros(dim, dtype=np.float32)
    if len(state) >= 6:
        proprio[:3] = state[:3]
        proprio[3:9] = euler_xyz_to_rot6d_interleaved(state[3:6]).astype(np.float32)
    return proprio


def _binarize_gripper(action: np.ndarray) -> np.ndarray:
    """Match official X-VLA CALVIN targets: gripper channels are binary."""
    out = action.astype(np.float32, copy=True)
    for idx in GRIPPER_INDICES_20D:
        if idx < out.shape[-1]:
            out[..., idx] = (out[..., idx] > 0.5).astype(np.float32)
    return out


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class ActionStats:
    """Normalization statistics for the first 10 action dimensions."""

    pos_mean: tuple[float, float, float]
    pos_std: tuple[float, float, float]
    rot_mean: tuple[float, ...]
    rot_std: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pos_mean": list(self.pos_mean),
            "pos_std": list(self.pos_std),
            "rot_mean": list(self.rot_mean),
            "rot_std": list(self.rot_std),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionStats":
        return cls(
            pos_mean=tuple(float(x) for x in payload["pos_mean"]),
            pos_std=tuple(float(x) for x in payload["pos_std"]),
            rot_mean=tuple(float(x) for x in payload["rot_mean"]),
            rot_std=tuple(float(x) for x in payload["rot_std"]),
        )


class XVLAManifestDataset(Dataset):
    """Map-style dataset that reads manifest entries and rollout artifacts."""

    def __init__(
        self,
        manifest_paths: list[str | Path],
        *,
        num_actions: int,
        domain_id: int = 2,
        repo_root: str | Path | None = None,
        zero_right_arm: bool = True,
    ) -> None:
        self.manifest_paths = [Path(path) for path in manifest_paths]
        self.num_actions = int(num_actions)
        self.domain_id = int(domain_id)
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.zero_right_arm = zero_right_arm

        self.entries: list[dict[str, Any]] = []
        for manifest_path in self.manifest_paths:
            self.entries.extend(_load_jsonl(manifest_path))

        self._npz_cache: dict[Path, Any] = {}
        self._array_cache: dict[tuple[Path, str], np.ndarray] = {}
        self._metadata_cache: dict[Path, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        anchor_rollout = self._resolve_rollout_dir(entry["anchor_rollout"])
        positive_rollout = self._resolve_rollout_dir(entry["positive_rollout"])
        negative_rollout = (
            self._resolve_rollout_dir(entry["negative_rollout"])
            if entry["negative_rollout"] is not None
            else None
        )

        anchor_step = int(entry["anchor_step"])
        positive_step = int(entry["positive_step"])
        negative_step = int(entry["negative_step"]) if entry["negative_step"] is not None else None

        anchor_traj = anchor_rollout / "trajectory.npz"
        positive_traj = positive_rollout / "trajectory.npz"

        images = [
            self._load_array(anchor_traj, "image_rgb_static")[anchor_step],
            self._load_array(anchor_traj, "image_rgb_gripper")[anchor_step],
        ]
        state = self._load_array(anchor_traj, "state")[anchor_step]
        proprio = _state_to_xvla_proprio(state)

        language = self._language_at(anchor_rollout, anchor_step)
        positive_seq = self._positive_action_sequence(positive_traj, positive_step)
        positive_first = positive_seq[0, :10].copy()

        has_negative = negative_rollout is not None and negative_step is not None
        if has_negative:
            negative_action = self._load_array(negative_rollout / "trajectory.npz", "model_native_action")[negative_step]
            negative_action = _binarize_gripper(negative_action)
            if self.zero_right_arm:
                negative_action[10:] = 0.0
            negative_first = negative_action[:10].copy()
        else:
            negative_first = np.zeros(10, dtype=np.float32)

        return {
            "sample_type": entry["sample_type"],
            "subtask": entry["subtask"],
            "language_instruction": language,
            "images": images,
            "proprio": torch.from_numpy(proprio).float(),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "positive_action_seq": torch.from_numpy(positive_seq).float(),
            "positive_action_first": torch.from_numpy(positive_first).float(),
            "negative_action_first": torch.from_numpy(negative_first).float(),
            "has_negative": torch.tensor(has_negative, dtype=torch.bool),
        }

    def compute_action_stats(self, indices: list[int] | None = None) -> ActionStats:
        """Compute normalization stats from positive first-step actions."""
        selected_indices = indices if indices is not None else list(range(len(self.entries)))
        pos_values: list[np.ndarray] = []
        rot_values: list[np.ndarray] = []
        for idx in selected_indices:
            entry = self.entries[idx]
            positive_rollout = self._resolve_rollout_dir(entry["positive_rollout"])
            positive_step = int(entry["positive_step"])
            action = self._load_array(positive_rollout / "trajectory.npz", "model_native_action")[positive_step]
            action = _binarize_gripper(action)
            if self.zero_right_arm:
                action = action.copy()
                action[10:] = 0.0
            pos_values.append(action[:3].astype(np.float32))
            rot_values.append(action[3:9].astype(np.float32))

        pos_stack = np.stack(pos_values, axis=0)
        rot_stack = np.stack(rot_values, axis=0)

        pos_std = np.maximum(pos_stack.std(axis=0), 1e-6)
        rot_std = np.maximum(rot_stack.std(axis=0), 1e-6)

        return ActionStats(
            pos_mean=tuple(float(x) for x in pos_stack.mean(axis=0)),
            pos_std=tuple(float(x) for x in pos_std),
            rot_mean=tuple(float(x) for x in rot_stack.mean(axis=0)),
            rot_std=tuple(float(x) for x in rot_std),
        )

    def save_action_stats(self, path: str | Path, indices: list[int] | None = None) -> ActionStats:
        stats = self.compute_action_stats(indices=indices)
        Path(path).write_text(json.dumps(stats.as_dict(), indent=2))
        return stats

    @staticmethod
    def load_action_stats(path: str | Path) -> ActionStats:
        return ActionStats.from_dict(json.loads(Path(path).read_text()))

    def _resolve_rollout_dir(self, rollout_ref: str) -> Path:
        rollout_path = Path(rollout_ref)
        if rollout_path.is_absolute():
            return rollout_path
        return (self.repo_root / rollout_path).resolve()

    def _load_npz(self, trajectory_path: Path) -> Any:
        if trajectory_path not in self._npz_cache:
            self._npz_cache[trajectory_path] = np.load(trajectory_path, allow_pickle=False)
        return self._npz_cache[trajectory_path]

    def _load_array(self, trajectory_path: Path, key: str) -> np.ndarray:
        cache_key = (trajectory_path, key)
        if key not in CACHED_ARRAY_KEYS:
            return self._load_npz(trajectory_path)[key]
        if cache_key not in self._array_cache:
            self._array_cache[cache_key] = self._load_npz(trajectory_path)[key]
        return self._array_cache[cache_key]

    def _load_metadata(self, rollout_dir: Path) -> dict[str, Any]:
        if rollout_dir not in self._metadata_cache:
            self._metadata_cache[rollout_dir] = json.loads((rollout_dir / "metadata.json").read_text())
        return self._metadata_cache[rollout_dir]

    def _language_at(self, rollout_dir: Path, step: int) -> str:
        metadata = self._load_metadata(rollout_dir)
        language_table = metadata.get("language_table", [])
        traj = rollout_dir / "trajectory.npz"
        language_ids = self._load_array(traj, "language_id")
        lang_idx = int(language_ids[step])
        if 0 <= lang_idx < len(language_table):
            return str(language_table[lang_idx])
        return str(metadata.get("task", rollout_dir.name))

    def _positive_action_sequence(self, trajectory_path: Path, start_step: int) -> np.ndarray:
        actions = self._load_array(trajectory_path, "model_native_action")
        start = int(start_step)
        end = min(start + self.num_actions, actions.shape[0])
        seq = actions[start:end].astype(np.float32, copy=True)
        seq = _binarize_gripper(seq)
        if end - start < self.num_actions:
            pad = np.repeat(seq[-1:, :], self.num_actions - seq.shape[0], axis=0)
            seq = np.concatenate([seq, pad], axis=0)
        if self.zero_right_arm and seq.shape[-1] >= 20:
            seq[:, 10:] = 0.0
        return seq


class XVLAManifestCollator:
    """Collate manifest samples into processor/model inputs."""

    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [sample["images"] for sample in batch]
        languages = [sample["language_instruction"] for sample in batch]
        encoded = self.processor(images=images, language_instruction=languages)

        return {
            "input_ids": encoded["input_ids"],
            "image_input": encoded["image_input"],
            "image_mask": encoded["image_mask"],
            "language_instruction": languages,
            "proprio": torch.stack([sample["proprio"] for sample in batch], dim=0),
            "domain_id": torch.stack([sample["domain_id"] for sample in batch], dim=0),
            "positive_action_seq": torch.stack([sample["positive_action_seq"] for sample in batch], dim=0),
            "positive_action_first": torch.stack([sample["positive_action_first"] for sample in batch], dim=0),
            "negative_action_first": torch.stack([sample["negative_action_first"] for sample in batch], dim=0),
            "has_negative": torch.stack([sample["has_negative"] for sample in batch], dim=0),
            "sample_type": [sample["sample_type"] for sample in batch],
            "subtask": [sample["subtask"] for sample in batch],
        }
