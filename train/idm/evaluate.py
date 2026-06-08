"""Evaluate a trained goal-image IDM with offline and optional LIBERO replay metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.idm import (  # noqa: E402
    NpzLRU,
    load_goal_image_idm_checkpoint,
    load_idm_config,
    predict_goal_image_idm_actions,
    resolve_idm_data_config,
)


DEFAULT_CONFIG = Path("experiment_specs/idm/libero_groot_goal_image_eval.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML/JSON IDM eval config.")
    return parser.parse_args()


def _section(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section {key!r} must be a mapping")
    return value


def _list_or_none(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError("Expected a list")
    return [str(v) for v in value]


def _device_from_config(value: object) -> str:
    if value is None or str(value) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = str(value)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is cuda but torch.cuda.is_available() is false")
    return device


def _infer_seed(root: Path, fallback: int = 7) -> int:
    match = re.search(r"seed(\d+)", root.name)
    return int(match.group(1)) if match else fallback


def _trajectory_paths(rollout_roots: Iterable[Path], trajectory_glob: str) -> list[Path]:
    paths: list[Path] = []
    for root in rollout_roots:
        paths.extend(sorted(root.glob(trajectory_glob)))
    if not paths:
        raise FileNotFoundError(f"No trajectories matching {trajectory_glob!r} under {list(rollout_roots)}")
    return paths


def _valid_samples(
    paths: list[Path],
    *,
    image_keys: tuple[str, ...],
    action_key: str,
    state_key: str,
    horizon: int,
    max_trajectories: int | None,
) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for path in paths[:max_trajectories]:
        with np.load(path, allow_pickle=False) as traj:
            required = {action_key, state_key, *image_keys}
            if not required.issubset(traj.files):
                continue
            length = min(len(traj[action_key]), len(traj[state_key]))
            for key in image_keys:
                length = min(length, len(traj[key]))
            if length <= horizon:
                continue
            samples.extend((path, int(step)) for step in range(length - horizon))
    return samples


def _ref_for(path: Path, step: int) -> dict[str, object]:
    root = path.parents[2]
    rel = path.relative_to(root)
    return {"root": str(root), "trajectory": str(rel), "obs_step": int(step), "state_step": int(step)}


def _metadata_path(traj_path: Path) -> Path:
    return traj_path.with_name("metadata.json")


def _load_metadata(traj_path: Path) -> dict[str, Any]:
    return json.loads(_metadata_path(traj_path).read_text())


def _l2(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def _mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(x, dtype=np.float64))))


def _image_mse(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32) / 255.0
    bb = np.asarray(b, dtype=np.float32) / 255.0
    return float(np.mean((aa - bb) ** 2))


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _label_image(image: np.ndarray, label: str) -> Image.Image:
    pil = Image.fromarray(_as_uint8_rgb(image)).convert("RGB")
    label_h = 26
    canvas = Image.new("RGB", (pil.width, pil.height + label_h), (0, 0, 0))
    canvas.paste(pil, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), label, fill=(255, 255, 255))
    return canvas


def _save_replay_media(
    *,
    media_dir: Path,
    sample_index: int,
    traj_path: Path,
    step: int,
    horizon: int,
    image_keys: tuple[str, ...],
    current_images: dict[str, np.ndarray],
    pred_next_images: dict[str, np.ndarray],
    rec_next_images: dict[str, np.ndarray],
    pred_step_images: list[dict[str, np.ndarray]],
    rec_step_images: list[dict[str, np.ndarray]],
    traj: Any,
    row: dict[str, object],
    fps: int,
    save_video: bool,
) -> None:
    media_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{sample_index:06d}_t{step:04d}_h{horizon}"
    meta = {
        "sample_index": int(sample_index),
        "trajectory": str(traj_path),
        "step": int(step),
        "goal_step": int(step + horizon),
        "pred_next_state_l2": row.get("pred_next_state_l2"),
        "recorded_next_state_l2": row.get("recorded_next_state_l2"),
        "replay_state_at_t_l2": row.get("replay_state_at_t_l2"),
        "pred_improves_over_current": row.get("pred_improves_over_current"),
    }
    (media_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    for image_key in image_keys:
        view = image_key.removeprefix("image_")
        if view not in current_images or view not in pred_next_images or view not in rec_next_images or image_key not in traj:
            continue
        labeled = [
            _label_image(current_images[view], f"A replay t={step}"),
            _label_image(np.asarray(traj[image_key][step + horizon], dtype=np.uint8), f"target B t+{horizon}"),
            _label_image(pred_next_images[view], "IDM pred"),
            _label_image(rec_next_images[view], "recorded action"),
        ]
        width = sum(img.width for img in labeled)
        height = max(img.height for img in labeled)
        sheet = Image.new("RGB", (width, height), (18, 18, 18))
        x = 0
        for img in labeled:
            sheet.paste(img, (x, 0))
            x += img.width
        sheet.save(media_dir / f"{stem}_{view}.png")

        seq_rows = [
            [("A replay", current_images[view])]
            + [
                (f"target +{idx}", np.asarray(traj[image_key][step + idx], dtype=np.uint8))
                for idx in range(1, horizon + 1)
            ],
            [("A replay", current_images[view])]
            + [
                (f"IDM pred +{idx}", pred_step_images[idx - 1][view])
                for idx in range(1, horizon + 1)
                if view in pred_step_images[idx - 1]
            ],
            [("A replay", current_images[view])]
            + [
                (f"recorded +{idx}", rec_step_images[idx - 1][view])
                for idx in range(1, horizon + 1)
                if view in rec_step_images[idx - 1]
            ],
        ]
        labeled_rows = [[_label_image(image, label) for label, image in seq_row] for seq_row in seq_rows]
        row_width = max(sum(img.width for img in labeled_row) for labeled_row in labeled_rows)
        row_height = max(max(img.height for img in labeled_row) for labeled_row in labeled_rows)
        rollout_sheet = Image.new("RGB", (row_width, row_height * len(labeled_rows)), (18, 18, 18))
        y = 0
        for labeled_row in labeled_rows:
            x = 0
            for img in labeled_row:
                rollout_sheet.paste(img, (x, y))
                x += img.width
            y += row_height
        rollout_sheet.save(media_dir / f"{stem}_{view}_rollout.png")

        if not save_video:
            continue
        frames: list[np.ndarray] = []
        repeat = max(1, int(fps))
        for img in labeled:
            frames.extend([np.asarray(img)] * repeat)
        try:
            import imageio.v2 as imageio

            imageio.mimsave(media_dir / f"{stem}_{view}.gif", frames, duration=1.0 / max(1, int(fps)))
            try:
                imageio.mimsave(media_dir / f"{stem}_{view}.mp4", frames, fps=max(1, int(fps)), macro_block_size=1)
            except Exception:
                pass
        except Exception:
            pass


def _first_action(pred: np.ndarray) -> np.ndarray:
    action = np.asarray(pred, dtype=np.float32)
    if action.ndim == 2:
        return action[0]
    return action.reshape(-1).astype(np.float32)


def _action_chunk(action: np.ndarray, horizon: int) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:horizon]


def _offline_eval_sample(
    *,
    model: Any,
    stats: Any,
    cache: NpzLRU,
    traj_path: Path,
    step: int,
    image_keys: tuple[str, ...],
    image_size: int,
    action_key: str,
    state_key: str,
    horizon: int,
    device: torch.device,
) -> dict[str, object]:
    current_ref = _ref_for(traj_path, step)
    goal_ref = _ref_for(traj_path, step + horizon)
    pred_chunk = predict_goal_image_idm_actions(
        model,
        stats,
        current_ref,
        goal_ref,
        image_keys=image_keys,
        image_size=image_size,
        device=device,
        npz_cache=cache,
        state_key=state_key,
    )
    pred_chunk = _action_chunk(pred_chunk, horizon)
    pred = _first_action(pred_chunk)
    traj = cache.get(traj_path)
    target_chunk = np.asarray(traj[action_key][step : step + horizon], dtype=np.float32)
    target = target_chunk[0]
    state_t = np.asarray(traj[state_key][step], dtype=np.float32)
    state_goal = np.asarray(traj[state_key][step + horizon], dtype=np.float32)
    pred_grip_sign = int(1 if pred[6] >= 0 else -1) if len(pred) >= 7 else None
    target_grip_sign = int(1 if target[6] >= 0 else -1) if len(target) >= 7 else None
    grip_sign_match_chunk = None
    grip_sign_match_rate = None
    if pred_chunk.shape[-1] >= 7 and target_chunk.shape[-1] >= 7:
        pred_signs = pred_chunk[:, 6] >= 0
        target_signs = target_chunk[:, 6] >= 0
        sign_matches = pred_signs == target_signs
        grip_sign_match_chunk = bool(np.all(sign_matches))
        grip_sign_match_rate = float(np.mean(sign_matches))

    row: dict[str, object] = {
        "trajectory": str(traj_path),
        "step": int(step),
        "goal_step": int(step + horizon),
        "pred_action": pred.astype(float).tolist(),
        "target_action": target.astype(float).tolist(),
        "pred_action_chunk": pred_chunk.astype(float).tolist(),
        "target_action_chunk": target_chunk.astype(float).tolist(),
        "action_l1": _mae(pred_chunk - target_chunk),
        "action_l2": _l2(pred_chunk - target_chunk),
        "first_action_l1": _mae(pred - target),
        "first_action_l2": _l2(pred - target),
        "pos_l2": _l2(pred_chunk[:, 0:3] - target_chunk[:, 0:3]),
        "rot_l2": _l2(pred_chunk[:, 3:6] - target_chunk[:, 3:6]),
        "first_pos_l2": _l2(pred[:3] - target[:3]),
        "first_rot_l2": _l2(pred[3:6] - target[3:6]),
        "grip_abs": float(np.mean(np.abs(pred_chunk[:, 6] - target_chunk[:, 6]))) if len(pred) >= 7 else None,
        "first_grip_abs": float(abs(pred[6] - target[6])) if len(pred) >= 7 else None,
        "grip_max_abs": float(np.max(np.abs(pred_chunk[:, 6] - target_chunk[:, 6]))) if len(pred) >= 7 else None,
        "pred_grip_sign": pred_grip_sign,
        "target_grip_sign": target_grip_sign,
        "grip_sign_match": bool(pred_grip_sign == target_grip_sign) if pred_grip_sign is not None else None,
        "grip_sign_match_chunk": grip_sign_match_chunk,
        "grip_sign_match_rate": grip_sign_match_rate,
        "current_to_goal_state_l2": _l2(state_t - state_goal),
    }
    return row


def _make_libero_benchmark(suite: str, seed: int):
    from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark

    return LIBEROBenchmark(
        suite=suite,
        seed=seed,
        env_seed=seed,
        num_steps_wait=10,
        send_wrist_image=True,
        send_state=True,
        quat_no_antipodal=True,
    )


def _state_from_obs(benchmark: Any, raw_obs: Any, task: dict[str, Any]) -> np.ndarray:
    obs = benchmark.make_obs(raw_obs, task)
    return np.asarray(obs["states"], dtype=np.float32)


def _images_from_obs(benchmark: Any, raw_obs: Any, task: dict[str, Any]) -> dict[str, np.ndarray]:
    obs = benchmark.make_obs(raw_obs, task)
    return {k: np.asarray(v, dtype=np.uint8) for k, v in obs["images"].items()}


def _replay_to_step(
    *,
    benchmark: Any,
    task: dict[str, Any],
    traj: Any,
    action_key: str,
    step: int,
) -> Any:
    raw_obs = benchmark.reset(task)
    for idx in range(step):
        raw_obs = benchmark.step({"actions": np.asarray(traj[action_key][idx], dtype=np.float32)}).obs
    return raw_obs


def _apply_action_chunk(benchmark: Any, actions: np.ndarray) -> Any:
    raw_obs = None
    for action in np.asarray(actions, dtype=np.float32):
        raw_obs = benchmark.step({"actions": action}).obs
    return raw_obs


def _apply_action_sequence(benchmark: Any, actions: np.ndarray) -> list[Any]:
    raw_obs_seq = []
    for action in np.asarray(actions, dtype=np.float32):
        raw_obs_seq.append(benchmark.step({"actions": action}).obs)
    return raw_obs_seq


def _replay_eval_sample(
    *,
    benchmarks: dict[tuple[str, int], Any],
    traj_path: Path,
    step: int,
    pred_actions: np.ndarray,
    horizon: int,
    action_key: str,
    state_key: str,
    image_keys: tuple[str, ...],
    cache: NpzLRU,
    media_dir: Path | None = None,
    media_sample_index: int = 0,
    media_fps: int = 1,
    save_video: bool = True,
) -> dict[str, object]:
    meta = _load_metadata(traj_path)
    task_meta = dict(meta["task_metadata"])
    suite = str(task_meta["suite"])
    seed = _infer_seed(traj_path.parents[2])
    key = (suite, seed)
    if key not in benchmarks:
        benchmarks[key] = _make_libero_benchmark(suite, seed)
    benchmark = benchmarks[key]
    tasks = benchmark.get_tasks()
    task = dict(tasks[int(task_meta["task_id"])])
    task["episode_idx"] = int(task_meta["episode_idx"])

    traj = cache.get(traj_path)
    raw_at_t = _replay_to_step(
        benchmark=benchmark,
        task=task,
        traj=traj,
        action_key=action_key,
        step=step,
    )
    state_at_t = _state_from_obs(benchmark, raw_at_t, task)
    current_images = _images_from_obs(benchmark, raw_at_t, task)
    target_state_t = np.asarray(traj[state_key][step], dtype=np.float32)
    target_state_next = np.asarray(traj[state_key][step + horizon], dtype=np.float32)

    pred_raw_seq = _apply_action_sequence(benchmark, _action_chunk(pred_actions, horizon))
    pred_next = pred_raw_seq[-1]
    pred_next_state = _state_from_obs(benchmark, pred_next, task)
    pred_next_images = _images_from_obs(benchmark, pred_next, task)
    pred_step_states = [_state_from_obs(benchmark, raw_obs, task) for raw_obs in pred_raw_seq]
    pred_step_images = [_images_from_obs(benchmark, raw_obs, task) for raw_obs in pred_raw_seq]

    raw_at_t_for_target = _replay_to_step(
        benchmark=benchmark,
        task=task,
        traj=traj,
        action_key=action_key,
        step=step,
    )
    target_actions = np.asarray(traj[action_key][step : step + horizon], dtype=np.float32)
    rec_raw_seq = _apply_action_sequence(benchmark, target_actions)
    rec_next = rec_raw_seq[-1]
    rec_next_state = _state_from_obs(benchmark, rec_next, task)
    rec_next_images = _images_from_obs(benchmark, rec_next, task)
    rec_step_states = [_state_from_obs(benchmark, raw_obs, task) for raw_obs in rec_raw_seq]
    rec_step_images = [_images_from_obs(benchmark, raw_obs, task) for raw_obs in rec_raw_seq]

    image_metrics: dict[str, object] = {}
    for image_key in image_keys:
        view = image_key.removeprefix("image_")
        if view not in pred_next_images or image_key not in traj:
            continue
        target_img = np.asarray(traj[image_key][step + horizon], dtype=np.uint8)
        image_metrics[f"pred_next_{view}_mse"] = _image_mse(pred_next_images[view], target_img)
        image_metrics[f"recorded_next_{view}_mse"] = _image_mse(rec_next_images[view], target_img)
        for idx in range(1, horizon + 1):
            target_step_img = np.asarray(traj[image_key][step + idx], dtype=np.uint8)
            if view in pred_step_images[idx - 1]:
                image_metrics[f"pred_step{idx}_{view}_mse"] = _image_mse(
                    pred_step_images[idx - 1][view], target_step_img
                )
                image_metrics[f"pred_to_goal_step{idx}_{view}_mse"] = _image_mse(
                    pred_step_images[idx - 1][view], target_img
                )
            if view in rec_step_images[idx - 1]:
                image_metrics[f"recorded_step{idx}_{view}_mse"] = _image_mse(
                    rec_step_images[idx - 1][view], target_step_img
                )
                image_metrics[f"recorded_to_goal_step{idx}_{view}_mse"] = _image_mse(
                    rec_step_images[idx - 1][view], target_img
                )

    step_state_metrics: dict[str, object] = {}
    pred_step_l2s = []
    recorded_step_l2s = []
    pred_to_goal_l2s = []
    recorded_to_goal_l2s = []
    for idx in range(1, horizon + 1):
        target_step_state = np.asarray(traj[state_key][step + idx], dtype=np.float32)
        pred_l2 = _l2(pred_step_states[idx - 1] - target_step_state)
        rec_l2 = _l2(rec_step_states[idx - 1] - target_step_state)
        pred_to_goal_l2 = _l2(pred_step_states[idx - 1] - target_state_next)
        rec_to_goal_l2 = _l2(rec_step_states[idx - 1] - target_state_next)
        pred_step_l2s.append(pred_l2)
        recorded_step_l2s.append(rec_l2)
        pred_to_goal_l2s.append(pred_to_goal_l2)
        recorded_to_goal_l2s.append(rec_to_goal_l2)
        step_state_metrics[f"pred_step{idx}_state_l2"] = pred_l2
        step_state_metrics[f"recorded_step{idx}_state_l2"] = rec_l2
        step_state_metrics[f"pred_to_goal_step{idx}_state_l2"] = pred_to_goal_l2
        step_state_metrics[f"recorded_to_goal_step{idx}_state_l2"] = rec_to_goal_l2
    best_idx = int(np.argmin(np.asarray(pred_step_l2s, dtype=np.float64))) + 1
    best_goal_idx = int(np.argmin(np.asarray(pred_to_goal_l2s, dtype=np.float64))) + 1

    row = {
        "replay_state_at_t_l2": _l2(state_at_t - target_state_t),
        "pred_next_state_l2": _l2(pred_next_state - target_state_next),
        "recorded_next_state_l2": _l2(rec_next_state - target_state_next),
        "pred_best_step": best_idx,
        "pred_best_step_state_l2": float(pred_step_l2s[best_idx - 1]),
        "recorded_best_step_state_l2": float(min(recorded_step_l2s)),
        "pred_to_goal_best_step": best_goal_idx,
        "pred_to_goal_best_step_state_l2": float(pred_to_goal_l2s[best_goal_idx - 1]),
        "recorded_to_goal_best_step_state_l2": float(min(recorded_to_goal_l2s)),
        "pred_improves_over_current": bool(
            _l2(pred_next_state - target_state_next) < _l2(state_at_t - target_state_next)
        ),
        "pred_vs_recorded_next_state_ratio": float(
            _l2(pred_next_state - target_state_next) / max(_l2(rec_next_state - target_state_next), 1e-8)
        ),
        **step_state_metrics,
        **image_metrics,
    }
    if media_dir is not None:
        _save_replay_media(
            media_dir=media_dir,
            sample_index=media_sample_index,
            traj_path=traj_path,
            step=step,
            horizon=horizon,
            image_keys=image_keys,
            current_images=current_images,
            pred_next_images=pred_next_images,
            rec_next_images=rec_next_images,
            pred_step_images=pred_step_images,
            rec_step_images=rec_step_images,
            traj=traj,
            row=row,
            fps=media_fps,
            save_video=save_video,
        )
    return row


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    values: dict[str, list[float]] = defaultdict(list)
    bools: dict[str, list[bool]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                bools[key].append(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                values[key].append(float(value))
            elif key.endswith("_error"):
                errors[key] += 1
    summary: dict[str, object] = {"num_samples": len(rows)}
    for key, vals in sorted(values.items()):
        arr = np.asarray(vals, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    for key, vals in sorted(bools.items()):
        summary[key] = {"mean": float(np.mean(vals)), "count": int(sum(vals))}
    for key, count in sorted(errors.items()):
        summary[key] = {"count": int(count)}
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_idm_config(Path(args.config))
    paths = _section(cfg, "paths")
    data = _section(cfg, "data")
    eval_cfg = _section(cfg, "eval")
    replay_cfg = _section(cfg, "replay")
    runtime = _section(cfg, "runtime")

    output_dir = Path(str(paths["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(_device_from_config(runtime.get("device", "auto")))

    model, stats, ckpt_config = load_goal_image_idm_checkpoint(Path(str(paths["checkpoint"])), map_location=device)
    model.to(device)
    model.eval()

    preset = str(data.get("preset", ckpt_config.get("preset", "libero:groot")))
    image_keys_override = _list_or_none(data.get("image_keys"))
    data_config = resolve_idm_data_config(
        preset,
        trajectory_glob=data.get("trajectory_glob"),
        image_keys=tuple(image_keys_override) if image_keys_override else None,
        state_key=data.get("state_key"),
        action_key=data.get("action_key"),
    )
    image_keys = tuple(image_keys_override or ckpt_config.get("image_keys") or data_config["image_keys"])
    trajectory_glob = str(data_config["trajectory_glob"])
    action_key = str(data.get("action_key") or ckpt_config.get("action_key") or data_config["action_key"])
    state_key = str(data.get("state_key") or ckpt_config.get("state_key") or data_config["state_key"])
    image_size = int(ckpt_config["image_size"])
    horizon = int(ckpt_config["horizon"])

    rollout_roots = [Path(p) for p in _list_or_none(paths.get("rollout_roots")) or []]
    if not rollout_roots:
        raise ValueError("paths.rollout_roots must contain at least one rollout root")
    traj_paths = _trajectory_paths(rollout_roots, trajectory_glob)
    rng = np.random.default_rng(int(eval_cfg.get("seed", 0)))
    if bool(eval_cfg.get("shuffle_trajectories", False)):
        traj_paths = list(traj_paths)
        rng.shuffle(traj_paths)
    samples = _valid_samples(
        traj_paths,
        image_keys=image_keys,
        action_key=action_key,
        state_key=state_key,
        horizon=horizon,
        max_trajectories=eval_cfg.get("max_trajectories"),
    )
    num_samples = min(int(eval_cfg.get("num_samples", 256)), len(samples))
    chosen = rng.choice(len(samples), size=num_samples, replace=False) if num_samples else []
    chosen_samples = [samples[int(i)] for i in np.sort(chosen)]

    print(
        f"Evaluating IDM checkpoint={paths['checkpoint']} samples={len(chosen_samples)} "
        f"horizon={horizon} replay={bool(replay_cfg.get('enabled', False))}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    cache = NpzLRU(int(runtime.get("cache_size", 32)))
    benchmarks: dict[tuple[str, int], Any] = {}
    try:
        iterator = chosen_samples
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(chosen_samples, desc="evaluate IDM", dynamic_ncols=True)
        except ImportError:
            pass
        replay_limit = int(replay_cfg.get("num_samples", 0))
        replay_media_enabled = bool(replay_cfg.get("save_media", False))
        replay_media_limit = int(replay_cfg.get("media_samples", replay_limit))
        replay_media_dir = output_dir / str(replay_cfg.get("media_dir", "replay_media"))
        replay_media_fps = int(replay_cfg.get("media_fps", 1))
        replay_save_video = bool(replay_cfg.get("save_video", True))
        for replay_count, (traj_path, step) in enumerate(iterator):
            row = _offline_eval_sample(
                model=model,
                stats=stats,
                cache=cache,
                traj_path=traj_path,
                step=step,
                image_keys=image_keys,
                image_size=image_size,
                action_key=action_key,
                state_key=state_key,
                horizon=horizon,
                device=device,
            )
            if replay_cfg.get("enabled", False) and replay_count < replay_limit:
                try:
                    row.update(
                        _replay_eval_sample(
                            benchmarks=benchmarks,
                            traj_path=traj_path,
                            step=step,
                            pred_actions=np.asarray(row["pred_action_chunk"], dtype=np.float32),
                            horizon=horizon,
                            action_key=action_key,
                            state_key=state_key,
                            image_keys=image_keys,
                            cache=cache,
                            media_dir=(
                                replay_media_dir
                                if replay_media_enabled and replay_count < replay_media_limit
                                else None
                            ),
                            media_sample_index=replay_count,
                            media_fps=replay_media_fps,
                            save_video=replay_save_video,
                        )
                    )
                except Exception as exc:  # Keep offline eval useful when LIBERO replay is unavailable.
                    row["replay_error"] = repr(exc)
            rows.append(row)
    finally:
        cache.close()
        for benchmark in benchmarks.values():
            benchmark.cleanup()

    summary = _summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (output_dir / "samples.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote IDM eval results to {output_dir}")


if __name__ == "__main__":
    main()
