"""Train a goal-conditioned image IDM on rollout trajectories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.idm import (  # noqa: E402
    GoalImageIDM,
    TrajectoryGoalImageIDMDataset,
    goal_image_idm_loss,
    load_idm_config,
    resolve_idm_data_config,
    save_goal_image_idm_checkpoint,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_CONFIG = Path("experiment_specs/idm/libero_groot_goal_image_train.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML/JSON IDM training config.")
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
        raise RuntimeError(
            "runtime.device is set to cuda, but torch.cuda.is_available() is false. "
            "Check nvidia-smi, NVIDIA driver, CUDA_VISIBLE_DEVICES, and the active conda env."
        )
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _epoch(
    *,
    model: GoalImageIDM,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total = 0
    sums = {
        "loss": 0.0,
        "loss_pos": 0.0,
        "loss_rot": 0.0,
        "loss_grip": 0.0,
        "pred_abs_max": 0.0,
        "target_abs_max": 0.0,
    }
    iterator = loader
    progress = None
    if tqdm is not None:
        progress = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False)
        iterator = progress
    for batch in iterator:
        current_images = batch["current_images"].to(device, non_blocking=True)
        goal_images = batch["goal_images"].to(device, non_blocking=True)
        current_state = batch["current_state"].to(device, non_blocking=True)
        goal_state = batch["goal_state"].to(device, non_blocking=True)
        target = batch["action"].to(device, non_blocking=True)

        pred = model(current_images, goal_images, current_state, goal_state)
        loss, parts = goal_image_idm_loss(pred, target)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        n = int(target.shape[0])
        total += n
        sums["loss"] += float(loss.detach().cpu()) * n
        for key, value in parts.items():
            if key not in sums:
                sums[key] = 0.0
            sums[key] += float(value.detach().cpu()) * n
        sums["pred_abs_max"] = max(sums["pred_abs_max"], float(pred.detach().abs().max().cpu()))
        sums["target_abs_max"] = max(sums["target_abs_max"], float(target.detach().abs().max().cpu()))
        if progress is not None:
            progress.set_postfix(loss=f"{sums['loss'] / max(1, total):.4f}")
    out = {key: value / max(1, total) for key, value in sums.items() if key.startswith("loss")}
    out["pred_abs_max"] = sums["pred_abs_max"]
    out["target_abs_max"] = sums["target_abs_max"]
    return out


def _init_wandb(*, output_dir: Path, wandb_config: dict[str, object], run_config: dict[str, object]):
    mode = str(wandb_config.get("mode", "online"))
    if mode == "disabled":
        print("wandb disabled by config", flush=True)
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Install wandb or use --wandb-mode disabled.") from exc
    run = wandb.init(
        project=str(wandb_config.get("project", "libero-goal-image-idm")),
        entity=wandb_config.get("entity"),
        name=wandb_config.get("run_name") or output_dir.name,
        mode=mode,
        config=run_config,
    )
    print(
        "wandb run: "
        f"entity={run.entity}, project={run.project}, name={run.name}, id={run.id}, url={run.url}",
        flush=True,
    )
    return run


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_idm_config(config_path)
    paths = _section(cfg, "paths")
    data = _section(cfg, "data")
    model_cfg = _section(cfg, "model")
    training = _section(cfg, "training")
    runtime = _section(cfg, "runtime")
    wandb_cfg = _section(cfg, "wandb")

    _load_dotenv(Path(str(runtime.get("dotenv", ".env"))))
    seed = int(training.get("seed", 0))
    _seed_everything(seed)

    output_dir = Path(str(paths["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(_device_from_config(runtime.get("device", "auto")))
    if device.type == "cuda":
        print(
            f"Using CUDA device {torch.cuda.current_device()}: "
            f"{torch.cuda.get_device_name(torch.cuda.current_device())}",
            flush=True,
        )
    else:
        print(f"Using device: {device}", flush=True)
    rollout_roots = [Path(p) for p in _list_or_none(paths.get("rollout_roots")) or []]
    if not rollout_roots:
        raise ValueError("paths.rollout_roots must contain at least one rollout root")

    preset = str(data.get("preset", "libero:groot"))
    data_config = resolve_idm_data_config(
        preset,
        trajectory_glob=data.get("trajectory_glob"),
        image_keys=tuple(_list_or_none(data.get("image_keys")) or []) or None,
        state_key=data.get("state_key"),
        action_key=data.get("action_key"),
    )
    image_keys = tuple(data_config["image_keys"])  # type: ignore[arg-type]
    trajectory_glob = str(data_config["trajectory_glob"])
    action_key = str(data_config["action_key"])
    state_key = str(data_config["state_key"])
    horizon = int(model_cfg.get("horizon", 3))
    image_size = int(model_cfg.get("image_size", 128))
    include_state = bool(model_cfg.get("include_state", False))
    cache_size = int(runtime.get("cache_size", 32))

    print(
        f"Building IDM dataset from {len(rollout_roots)} rollout roots "
        f"(preset={preset}, horizon={horizon}, image_keys={image_keys})",
        flush=True,
    )
    dataset = TrajectoryGoalImageIDMDataset(
        rollout_roots=rollout_roots,
        trajectory_glob=trajectory_glob,
        horizon=horizon,
        action_key=action_key,
        state_key=state_key,
        image_keys=image_keys,
        image_size=image_size,
        include_state=include_state,
        chunk_start_only=bool(data.get("chunk_start_only", False)),
        max_samples=data.get("max_samples"),
        max_trajectories=data.get("max_trajectories"),
        shuffle_trajectories=bool(data.get("shuffle_trajectories", True)),
        index_cache_path=Path(str(data["index_cache_path"])) if data.get("index_cache_path") else None,
        rebuild_index_cache=bool(data.get("rebuild_index_cache", False)),
        seed=seed,
        cache_size=cache_size,
    )
    val_ratio = float(training.get("val_ratio", 0.05))
    val_len = max(1, int(round(len(dataset) * val_ratio)))
    train_len = len(dataset) - val_len
    if train_len <= 0:
        raise ValueError("Dataset is too small for requested val split")
    train_set = Subset(dataset, range(train_len))
    val_set = Subset(dataset, range(train_len, train_len + val_len))

    batch_size = int(training.get("batch_size", 128))
    num_workers = int(runtime.get("num_workers", 4))
    shuffle_batches = bool(training.get("shuffle_batches", False))
    persistent_workers = num_workers > 0 and bool(runtime.get("persistent_workers", True))
    state_dim, action_dim = dataset.infer_dims()
    dataset.close()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_batches,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    checkpoint_config: dict[str, object] = {
        "config_path": str(config_path),
        "rollout_roots": [str(p.resolve()) for p in rollout_roots],
        "preset": preset,
        "trajectory_glob": trajectory_glob,
        "action_key": action_key,
        "state_key": state_key,
        "image_keys": list(image_keys),
        "image_size": image_size,
        "horizon": horizon,
        "include_state": include_state,
        "chunk_start_only": bool(data.get("chunk_start_only", False)),
        "max_samples": data.get("max_samples"),
        "max_trajectories": data.get("max_trajectories"),
        "shuffle_trajectories": bool(data.get("shuffle_trajectories", True)),
        "index_cache_path": data.get("index_cache_path"),
        "shuffle_batches": shuffle_batches,
        "batch_size": batch_size,
        "epochs": int(training.get("epochs", 20)),
        "lr": float(training.get("lr", 3e-4)),
        "weight_decay": float(training.get("weight_decay", 1e-4)),
        "image_feature_dim": int(model_cfg.get("image_feature_dim", 128)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 512)),
        "num_layers": int(model_cfg.get("num_layers", 3)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "action_limit": float(model_cfg.get("action_limit", 1.0)),
        "val_ratio": val_ratio,
        "seed": seed,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "train_samples": train_len,
        "val_samples": val_len,
        "wandb_project": wandb_cfg.get("project", "libero-goal-image-idm"),
        "wandb_mode": wandb_cfg.get("mode", "online"),
    }

    model = GoalImageIDM(
        num_views=len(image_keys),
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=horizon,
        image_feature_dim=int(checkpoint_config["image_feature_dim"]),
        hidden_dim=int(checkpoint_config["hidden_dim"]),
        num_layers=int(checkpoint_config["num_layers"]),
        dropout=float(checkpoint_config["dropout"]),
        include_state=include_state,
        action_limit=float(checkpoint_config["action_limit"]),
    ).to(device)
    print(f"Model parameter device: {next(model.parameters()).device}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(checkpoint_config["lr"]),
        weight_decay=float(checkpoint_config["weight_decay"]),
    )

    metrics_path = output_dir / "metrics.jsonl"
    best_val = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    start = time.time()

    print(f"Loaded {len(dataset)} goal-image IDM samples ({train_len} train / {val_len} val)")
    print(
        f"config={config_path}, preset={preset}, trajectory_glob={trajectory_glob}, action_key={action_key}, "
        f"state_key={state_key}, image_keys={image_keys}, image_size={image_size}, state_dim={state_dim}, "
        f"action_dim={action_dim}, horizon={horizon}, include_state={include_state}, device={device}"
    )
    print("Initializing wandb...", flush=True)
    wandb_run = _init_wandb(output_dir=output_dir, wandb_config=wandb_cfg, run_config=checkpoint_config)

    epochs = int(checkpoint_config["epochs"])
    with metrics_path.open("w") as f:
        for epoch in range(1, epochs + 1):
            train_metrics = _epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                desc=f"epoch {epoch:03d}/{epochs:03d} train",
            )
            with torch.no_grad():
                val_metrics = _epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    desc=f"epoch {epoch:03d}/{epochs:03d} val",
                )
            row = {
                "epoch": epoch,
                "elapsed_sec": round(time.time() - start, 3),
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            print(
                f"epoch {epoch:03d} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_pos={val_metrics.get('loss_pos', 0.0):.6f} "
                f"val_rot={val_metrics.get('loss_rot', 0.0):.6f} "
                f"val_grip={val_metrics.get('loss_grip', 0.0):.6f} "
                f"pred_abs_max={val_metrics['pred_abs_max']:.3f}"
            )
            if wandb_run is not None:
                import wandb

                wandb.log(row, step=epoch)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_goal_image_idm_checkpoint(best_path, model=model, stats=dataset.stats, config=checkpoint_config)
                if wandb_run is not None:
                    wandb_run.summary["best_val_loss"] = best_val
                    wandb_run.summary["best_epoch"] = epoch

    save_goal_image_idm_checkpoint(last_path, model=model, stats=dataset.stats, config=checkpoint_config)
    if wandb_run is not None:
        wandb_run.summary["last_checkpoint"] = str(last_path)
        wandb_run.summary["best_checkpoint"] = str(best_path)
        wandb_run.finish()
    dataset.close()
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
