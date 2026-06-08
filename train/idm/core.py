"""Goal-conditioned image inverse dynamics utilities.

The default preset targets the current LIBERO + GR00T rollout schema, but the
dataset/model code is intentionally benchmark-agnostic. Add a preset for a new
benchmark/model pair when trajectory key names differ.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_IMAGE_KEYS = ("image_agentview", "image_wrist")

IDM_DATA_PRESETS: dict[str, dict[str, object]] = {
    "libero:groot": {
        "trajectory_glob": "rollouts/*/trajectory.npz",
        "image_keys": DEFAULT_IMAGE_KEYS,
        "state_key": "state",
        "action_key": "env_action",
    },
}


def resolve_idm_data_config(
    preset: str = "libero:groot",
    *,
    trajectory_glob: str | None = None,
    image_keys: tuple[str, ...] | None = None,
    state_key: str | None = None,
    action_key: str | None = None,
) -> dict[str, object]:
    if preset not in IDM_DATA_PRESETS:
        known = ", ".join(sorted(IDM_DATA_PRESETS))
        raise KeyError(f"Unknown IDM preset {preset!r}. Known presets: {known}")
    config = dict(IDM_DATA_PRESETS[preset])
    if trajectory_glob is not None:
        config["trajectory_glob"] = trajectory_glob
    if image_keys is not None:
        config["image_keys"] = tuple(image_keys)
    if state_key is not None:
        config["state_key"] = state_key
    if action_key is not None:
        config["action_key"] = action_key
    config["image_keys"] = tuple(config["image_keys"])  # type: ignore[arg-type]
    return config


def load_idm_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read YAML config files") from exc
        data = yaml.safe_load(path.read_text())
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise TypeError(f"Expected mapping in config: {path}")
        return data
    raise ValueError(f"Unsupported config extension for {path}. Use .yaml, .yml, or .json")


def _json_hash(data: dict[str, object]) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class GoalImageIDMStats:
    state_mean: np.ndarray
    state_std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "state_mean": self.state_mean.astype(float).tolist(),
            "state_std": self.state_std.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "GoalImageIDMStats":
        return cls(
            state_mean=np.asarray(data["state_mean"], dtype=np.float32),
            state_std=np.asarray(data["state_std"], dtype=np.float32),
        )


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.size == 0:
            return
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        if self.mean is None or self.m2 is None:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        self.m2 = self.m2 + batch_m2 + delta**2 * self.count * batch_count / total
        self.count = total

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.m2 is None or self.count == 0:
            raise ValueError("No values were added to RunningStats")
        var = self.m2 / max(1, self.count - 1)
        return self.mean.astype(np.float32), np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)


class NpzLRU:
    def __init__(self, max_open: int = 32) -> None:
        self.max_open = max(1, int(max_open))
        self._cache: OrderedDict[Path, Any] = OrderedDict()

    def get(self, path: Path) -> Any:
        path = path.resolve()
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        if len(self._cache) >= self.max_open:
            self._cache.popitem(last=False)
        with np.load(path, allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in npz.files}
        self._cache[path] = arrays
        return arrays

    def close(self) -> None:
        self._cache.clear()


def _npz_files(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        return {Path(name).stem for name in zf.namelist() if name.endswith(".npy")}


def _npz_shape(path: Path, key: str) -> tuple[int, ...]:
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as zf:
        with zf.open(member) as fp:
            version = np.lib.format.read_magic(fp)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(fp)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(fp)
            else:
                shape, _, _ = np.lib.format._read_array_header(fp, version)
    return tuple(int(x) for x in shape)


def _resize_chw_uint8(image: np.ndarray, image_size: int) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(image)
    if pil.size != (image_size, image_size):
        pil = pil.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def _load_image_stack(
    traj: Any,
    *,
    image_keys: tuple[str, ...],
    step: int,
    image_size: int,
) -> np.ndarray:
    images: list[np.ndarray] = []
    for key in image_keys:
        if key not in traj:
            raise KeyError(f"{key} not found in trajectory")
        arr = traj[key]
        idx = max(0, min(int(step), len(arr) - 1))
        images.append(_resize_chw_uint8(arr[idx], image_size))
    return np.stack(images, axis=0).astype(np.float32)


class TrajectoryGoalImageIDMDataset(Dataset[dict[str, torch.Tensor]]):
    """Build goal-conditioned visual IDM samples.

    Each sample maps `(obs_t, goal_obs_{t+K})` to `action[t:t+K]`.
    At cache time the same model is applied as:
    `(failure_obs_at_target, success_obs_at_target) -> recovery action`.
    """

    def __init__(
        self,
        *,
        rollout_roots: Iterable[Path],
        trajectory_glob: str = "rollouts/*/trajectory.npz",
        horizon: int = 3,
        action_key: str = "env_action",
        state_key: str = "state",
        image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS,
        image_size: int = 128,
        include_state: bool = False,
        chunk_start_only: bool = False,
        max_samples: int | None = None,
        max_trajectories: int | None = None,
        shuffle_trajectories: bool = False,
        index_cache_path: Path | None = None,
        rebuild_index_cache: bool = False,
        seed: int = 0,
        cache_size: int = 32,
        stats: GoalImageIDMStats | None = None,
    ) -> None:
        self.rollout_roots = [Path(p).resolve() for p in rollout_roots]
        self.trajectory_glob = trajectory_glob
        self.horizon = int(horizon)
        self.action_key = action_key
        self.state_key = state_key
        self.image_keys = tuple(image_keys)
        self.image_size = int(image_size)
        self.include_state = bool(include_state)
        self.chunk_start_only = bool(chunk_start_only)
        self.max_samples = max_samples
        self.max_trajectories = max_trajectories
        self.shuffle_trajectories = bool(shuffle_trajectories)
        self.index_cache_path = Path(index_cache_path) if index_cache_path else None
        self.rebuild_index_cache = bool(rebuild_index_cache)
        self.seed = int(seed)
        self.cache = NpzLRU(cache_size)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

        cached = None if self.rebuild_index_cache else self._load_index_cache()
        if cached is not None:
            self.samples, cached_stats = cached
            self.stats = stats or cached_stats or self._compute_stats()
        else:
            self.samples = self._build_index()
            if not self.samples:
                raise ValueError("No valid goal-image IDM samples were found")
            self.stats = stats or self._compute_stats()
            self._save_index_cache()

    def _trajectory_paths(self) -> list[Path]:
        paths: list[Path] = []
        for root in self.rollout_roots:
            paths.extend(sorted(root.glob(self.trajectory_glob)))
        if not paths:
            raise FileNotFoundError(
                f"No trajectories matching {self.trajectory_glob!r} under: {self.rollout_roots}"
            )
        if self.shuffle_trajectories:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(paths)
        if self.max_trajectories is not None:
            paths = paths[: int(self.max_trajectories)]
        return paths

    def _index_cache_metadata(self) -> dict[str, object]:
        return {
            "rollout_roots": [str(p) for p in self.rollout_roots],
            "trajectory_glob": self.trajectory_glob,
            "horizon": self.horizon,
            "action_key": self.action_key,
            "state_key": self.state_key,
            "image_keys": list(self.image_keys),
            "include_state": self.include_state,
            "chunk_start_only": self.chunk_start_only,
            "max_samples": self.max_samples,
            "max_trajectories": self.max_trajectories,
            "shuffle_trajectories": self.shuffle_trajectories,
            "seed": self.seed,
        }

    def _load_index_cache(self) -> tuple[list[tuple[str, int]], GoalImageIDMStats | None] | None:
        if self.index_cache_path is None or not self.index_cache_path.exists():
            return None
        with np.load(self.index_cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"].item()))
            expected = self._index_cache_metadata()
            if metadata.get("hash") != _json_hash(expected):
                print(f"IDM index cache metadata mismatch, rebuilding: {self.index_cache_path}", flush=True)
                return None
            paths = [str(p) for p in cache["paths"].tolist()]
            steps = cache["steps"].astype(np.int64)
            samples = list(zip(paths, [int(s) for s in steps]))
            stats = None
            if "state_mean" in cache.files and "state_std" in cache.files:
                stats = GoalImageIDMStats(
                    state_mean=np.asarray(cache["state_mean"], dtype=np.float32),
                    state_std=np.asarray(cache["state_std"], dtype=np.float32),
                )
        print(f"Loaded IDM index cache: {self.index_cache_path} ({len(samples)} samples)", flush=True)
        return samples, stats

    def _save_index_cache(self) -> None:
        if self.index_cache_path is None:
            return
        self.index_cache_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._index_cache_metadata()
        metadata["hash"] = _json_hash(metadata)
        paths = np.asarray([path for path, _ in self.samples], dtype=str)
        steps = np.asarray([step for _, step in self.samples], dtype=np.int64)
        np.savez_compressed(
            self.index_cache_path,
            paths=paths,
            steps=steps,
            state_mean=self.stats.state_mean,
            state_std=self.stats.state_std,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True, ensure_ascii=False)),
        )
        print(f"Saved IDM index cache: {self.index_cache_path} ({len(self.samples)} samples)", flush=True)

    def _candidate_steps(self, traj: Any, length: int) -> np.ndarray:
        if self.chunk_start_only and "action_is_chunk_start" in traj.files:
            steps = np.flatnonzero(traj["action_is_chunk_start"].astype(bool))
        else:
            steps = np.arange(length, dtype=np.int64)
        return steps[steps + self.horizon < length]

    def _build_index(self) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        paths = self._trajectory_paths()
        iterator = paths
        progress = None
        if tqdm is not None:
            progress = tqdm(paths, desc="index IDM trajectories", dynamic_ncols=True, leave=True)
            iterator = progress
        for path in iterator:
            files = _npz_files(Path(path))
            required = {self.action_key, *self.image_keys}
            if self.include_state:
                required.add(self.state_key)
            if not required.issubset(files):
                continue
            length = _npz_shape(Path(path), self.action_key)[0]
            if self.include_state:
                length = min(length, _npz_shape(Path(path), self.state_key)[0])
            for key in self.image_keys:
                length = min(length, _npz_shape(Path(path), key)[0])
            if self.chunk_start_only:
                with np.load(path, allow_pickle=False) as traj:
                    steps = self._candidate_steps(traj, length)
            else:
                steps = np.arange(length, dtype=np.int64)
                steps = steps[steps + self.horizon < length]
            samples.extend((str(path), int(step)) for step in steps)
            if progress is not None:
                progress.set_postfix(samples=len(samples))
        if self.max_samples is not None and len(samples) > self.max_samples:
            samples = samples[: int(self.max_samples)]
        return samples

    def _compute_stats(self) -> GoalImageIDMStats:
        if not self.include_state:
            return GoalImageIDMStats(
                state_mean=np.zeros(1, dtype=np.float32),
                state_std=np.ones(1, dtype=np.float32),
            )
        state_stats = RunningStats()
        iterator = self.samples
        if tqdm is not None:
            iterator = tqdm(self.samples, desc="compute IDM state stats", dynamic_ncols=True, leave=True)
        for path, step in iterator:
            with np.load(path, allow_pickle=False) as traj:
                state_stats.update(traj[self.state_key][step].astype(np.float32))
                state_stats.update(traj[self.state_key][step + self.horizon].astype(np.float32))
        mean, std = state_stats.finalize()
        return GoalImageIDMStats(state_mean=mean, state_std=np.maximum(std, 1e-6))

    def __len__(self) -> int:
        return len(self.samples)

    def infer_dims(self) -> tuple[int, int]:
        path, step = self.samples[0]
        action_dim = int(_npz_shape(Path(path), self.action_key)[-1])
        if self.include_state:
            state_dim = int(_npz_shape(Path(path), self.state_key)[-1])
        else:
            state_dim = int(self.stats.state_mean.shape[-1])
        return state_dim, action_dim

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, step = self.samples[idx]
        try:
            traj = self.cache.get(Path(path))
            goal_step = step + self.horizon
            current_images = _load_image_stack(
                traj,
                image_keys=self.image_keys,
                step=step,
                image_size=self.image_size,
            )
            goal_images = _load_image_stack(
                traj,
                image_keys=self.image_keys,
                step=goal_step,
                image_size=self.image_size,
            )
            action = np.asarray(traj[self.action_key][step : step + self.horizon], dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)

            if self.include_state:
                current_state = np.asarray(traj[self.state_key][step], dtype=np.float32)
                goal_state = np.asarray(traj[self.state_key][goal_step], dtype=np.float32)
                current_state_n = (current_state - self.stats.state_mean) / self.stats.state_std
                goal_state_n = (goal_state - self.stats.state_mean) / self.stats.state_std
            else:
                current_state_n = np.zeros_like(self.stats.state_mean)
                goal_state_n = np.zeros_like(self.stats.state_mean)
        except Exception as exc:
            raise RuntimeError(f"Failed to load IDM sample idx={idx} path={path} step={step}") from exc

        return {
            "current_images": torch.from_numpy(current_images),
            "goal_images": torch.from_numpy(goal_images),
            "current_state": torch.from_numpy(current_state_n.astype(np.float32)),
            "goal_state": torch.from_numpy(goal_state_n.astype(np.float32)),
            "action": torch.from_numpy(action.astype(np.float32)),
        }

    def close(self) -> None:
        self.cache.close()


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoalImageIDM(nn.Module):
    def __init__(
        self,
        *,
        num_views: int,
        state_dim: int,
        action_dim: int,
        horizon: int,
        image_feature_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
        include_state: bool = False,
        action_limit: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_views = int(num_views)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.include_state = bool(include_state)
        self.action_limit = float(action_limit)
        self.encoder = ImageEncoder(feature_dim=image_feature_dim)

        fusion_dim = self.num_views * image_feature_dim * 3
        if self.include_state:
            fusion_dim += self.state_dim * 3
        layers: list[nn.Module] = []
        dim = fusion_dim
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, self.horizon * self.action_dim))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        current_images: torch.Tensor,
        goal_images: torch.Tensor,
        current_state: torch.Tensor | None = None,
        goal_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, views = current_images.shape[:2]
        cur = current_images.reshape(bsz * views, *current_images.shape[2:])
        goal = goal_images.reshape(bsz * views, *goal_images.shape[2:])
        cur_feat = self.encoder(cur).reshape(bsz, views, -1)
        goal_feat = self.encoder(goal).reshape(bsz, views, -1)
        pieces = [
            cur_feat.reshape(bsz, -1),
            goal_feat.reshape(bsz, -1),
            (goal_feat - cur_feat).reshape(bsz, -1),
        ]
        if self.include_state:
            if current_state is None or goal_state is None:
                raise ValueError("current_state and goal_state are required when include_state=True")
            pieces.extend([current_state, goal_state, goal_state - current_state])
        fused = torch.cat(pieces, dim=-1)
        out = self.head(fused).view(bsz, self.horizon, self.action_dim)
        return torch.tanh(out) * self.action_limit


def goal_image_idm_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = pred[..., : target.shape[-1]]
    if target.shape[-1] < 7:
        loss = F.smooth_l1_loss(pred, target)
        return loss, {"loss_action": loss}
    pos = F.smooth_l1_loss(pred[..., 0:3], target[..., 0:3])
    rot = F.smooth_l1_loss(pred[..., 3:6], target[..., 3:6])
    grip = F.smooth_l1_loss(pred[..., 6:], target[..., 6:])
    total = pos + rot + grip
    return total, {"loss_pos": pos, "loss_rot": rot, "loss_grip": grip}


def save_goal_image_idm_checkpoint(
    path: Path,
    *,
    model: GoalImageIDM,
    stats: GoalImageIDMStats,
    config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "stats": stats.to_dict(),
            "config": config,
        },
        path,
    )
    (path.parent / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (path.parent / "stats.json").write_text(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))


def load_goal_image_idm_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[GoalImageIDM, GoalImageIDMStats, dict[str, object]]:
    ckpt = torch.load(path, map_location=map_location)
    config = dict(ckpt["config"])
    stats = GoalImageIDMStats.from_dict(ckpt["stats"])
    model = GoalImageIDM(
        num_views=len(config["image_keys"]),
        state_dim=int(config["state_dim"]),
        action_dim=int(config["action_dim"]),
        horizon=int(config["horizon"]),
        image_feature_dim=int(config["image_feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config.get("dropout", 0.0)),
        include_state=bool(config.get("include_state", False)),
        action_limit=float(config.get("action_limit", 1.0)),
    )
    model.load_state_dict(ckpt["model_state"])
    return model, stats, config


def load_goal_image_ref(
    ref: dict[str, Any],
    *,
    image_keys: tuple[str, ...],
    image_size: int,
    stats: GoalImageIDMStats,
    npz_cache: NpzLRU,
    step_key: str = "obs_step",
    state_key: str = "state",
    use_state: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(ref["root"])) / Path(str(ref["trajectory"]))
    traj = npz_cache.get(path)
    step = int(ref.get(step_key, ref.get("obs_step", 0)))
    images = _load_image_stack(traj, image_keys=image_keys, step=step, image_size=image_size)
    if use_state and state_key in traj:
        state_idx = max(0, min(step, len(traj[state_key]) - 1))
        state = np.asarray(traj[state_key][state_idx], dtype=np.float32)
        state_n = (state - stats.state_mean) / stats.state_std
    else:
        state_n = np.zeros_like(stats.state_mean)
    return images, state_n.astype(np.float32)


@torch.no_grad()
def predict_goal_image_idm_actions(
    model: GoalImageIDM,
    stats: GoalImageIDMStats,
    current_ref: dict[str, Any],
    goal_ref: dict[str, Any],
    *,
    image_keys: tuple[str, ...],
    image_size: int,
    device: torch.device,
    npz_cache: NpzLRU | None = None,
    state_key: str = "state",
) -> np.ndarray:
    own_cache = npz_cache is None
    cache = npz_cache or NpzLRU(8)
    try:
        cur_img, cur_state = load_goal_image_ref(
            current_ref,
            image_keys=image_keys,
            image_size=image_size,
            stats=stats,
            npz_cache=cache,
            state_key=state_key,
            use_state=model.include_state,
        )
        goal_img, goal_state = load_goal_image_ref(
            goal_ref,
            image_keys=image_keys,
            image_size=image_size,
            stats=stats,
            npz_cache=cache,
            state_key=state_key,
            use_state=model.include_state,
        )
        pred = model(
            torch.from_numpy(cur_img[None]).to(device),
            torch.from_numpy(goal_img[None]).to(device),
            torch.from_numpy(cur_state[None]).to(device),
            torch.from_numpy(goal_state[None]).to(device),
        )
        return torch.clamp(pred[0], -1.0, 1.0).cpu().numpy().astype(np.float32)
    finally:
        if own_cache:
            cache.close()
