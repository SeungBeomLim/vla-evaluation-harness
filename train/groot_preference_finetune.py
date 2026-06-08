"""Custom GR00T-N1.6 preference fine-tuning entrypoint.

This script uses the official Isaac-GR00T model/processor stack, but replaces
the standard supervised trainer loss with the current LIBERO preference loss:

    BC(success-only) + distance NCE(onset) + distance NCE(IDM)

The model predicts a normalized action chunk with a differentiable copy of the
GR00T flow-matching inference loop. Targets are converted from the rollout
server action space back into GR00T's native action space, then normalized with
the loaded GR00T processor statistics before loss computation.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LOGGER = logging.getLogger("groot_preference_finetune")


GR00T_STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
GR00T_ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
IMAGE_KEY_MAP = {
    "image": "image_agentview",
    "wrist_image": "image_wrist",
}


@dataclass
class LoadedRef:
    images: dict[str, list[np.ndarray]]
    state: dict[str, np.ndarray]
    flat_state: np.ndarray


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log", mode="a"),
        ],
    )


def _load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class _NpzLRU:
    def __init__(self, max_open: int = 32) -> None:
        self.max_open = max(1, int(max_open))
        self._cache: OrderedDict[Path, Any] = OrderedDict()

    def get(self, path: Path) -> Any:
        path = path.resolve()
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        if len(self._cache) >= self.max_open:
            _, old = self._cache.popitem(last=False)
            old.close()
        arr = np.load(path, allow_pickle=False)
        self._cache[path] = arr
        return arr

    def close(self) -> None:
        for arr in self._cache.values():
            arr.close()
        self._cache.clear()


def _trajectory_path(ref: dict[str, Any]) -> Path:
    return Path(str(ref["root"])) / Path(str(ref["trajectory"]))


def _state_dict_from_flat(flat: np.ndarray) -> dict[str, np.ndarray]:
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    if flat.shape[0] < 8:
        raise ValueError(f"Expected LIBERO state dim >= 8, got {flat.shape}")
    return {
        "x": flat[0:1][None, :],
        "y": flat[1:2][None, :],
        "z": flat[2:3][None, :],
        "roll": flat[3:4][None, :],
        "pitch": flat[4:5][None, :],
        "yaw": flat[5:6][None, :],
        "gripper": flat[6:8][None, :],
    }


def _server_action_to_native(raw: np.ndarray, *, invert_gripper: bool) -> np.ndarray:
    """Convert rollout server/env action to GR00T native physical action.

    The GR00T LIBERO model emits gripper in [0, 1] before the harness applies
    ``server_g = 1 - 2 * native_g`` for LIBERO's close-positive [-1, 1] action.
    """
    action = np.asarray(raw, dtype=np.float32).copy()
    if action.shape[-1] < 7:
        raise ValueError(f"Expected 7D action, got {action.shape}")
    if invert_gripper:
        action[..., 6] = (1.0 - action[..., 6]) * 0.5
    return action


def _action_dict_from_flat(raw: np.ndarray, *, invert_gripper: bool) -> dict[str, np.ndarray]:
    action = _server_action_to_native(raw, invert_gripper=invert_gripper)
    return {
        "x": action[:, 0:1],
        "y": action[:, 1:2],
        "z": action[:, 2:3],
        "roll": action[:, 3:4],
        "pitch": action[:, 4:5],
        "yaw": action[:, 5:6],
        "gripper": action[:, 6:7],
    }


def _concat_action_dict(action: dict[str, np.ndarray], keys: tuple[str, ...] = GR00T_ACTION_KEYS) -> np.ndarray:
    return np.concatenate([np.asarray(action[k], dtype=np.float32) for k in keys], axis=-1)


def _load_window(
    traj: Any,
    key: str,
    start: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(traj[key], dtype=np.float32)
    dim = arr.shape[-1]
    out = np.zeros((length, dim), dtype=np.float32)
    mask = np.zeros((length,), dtype=np.float32)
    if start < len(arr):
        end = min(len(arr), start + length)
        n = max(0, end - start)
        if n:
            out[:n] = arr[start:end]
            mask[:n] = 1.0
    return out, mask


def _load_ref(ref: dict[str, Any], npz_cache: _NpzLRU) -> LoadedRef:
    traj = npz_cache.get(_trajectory_path(ref))
    obs_step = int(ref["obs_step"])
    state_step = int(ref.get("state_step", obs_step))

    images: dict[str, list[np.ndarray]] = {}
    for groot_key, traj_key in IMAGE_KEY_MAP.items():
        if traj_key in traj.files:
            images[groot_key] = [np.asarray(traj[traj_key][obs_step], dtype=np.uint8)]
    if "image" not in images:
        raise KeyError(f"image_agentview not found in {_trajectory_path(ref)}")

    flat_state = np.asarray(traj["state"][state_step], dtype=np.float32)
    return LoadedRef(images=images, state=_state_dict_from_flat(flat_state), flat_state=flat_state)


class SuccessOnlyDataset(Dataset):
    def __init__(self, path: Path, *, action_window: int, action_key: str = "env_action", cache_size: int = 32) -> None:
        self.rows = _read_jsonl(path)
        self.action_window = int(action_window)
        self.action_key = action_key
        self.cache = _NpzLRU(cache_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        ref = row["vla_input"]
        target_ref = row["target"]
        loaded = _load_ref(ref, self.cache)
        traj = self.cache.get(_trajectory_path(target_ref))
        action_step = int(target_ref.get("action_step", row.get("chunk_start_step", 0)))
        action, mask = _load_window(traj, row.get("action_key", self.action_key), action_step, self.action_window)
        return {
            "sample_type": "success",
            "row": row,
            "images": loaded.images,
            "state": loaded.state,
            "task": row["task"],
            "target_action": action,
            "target_mask": mask,
        }

    def close(self) -> None:
        self.cache.close()


class PreferencePairDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        idm_actions_path: Path,
        action_key: str = "env_action",
        cache_size: int = 32,
    ) -> None:
        self.rows = _read_jsonl(path)
        with np.load(idm_actions_path, allow_pickle=False) as idm_npz:
            idm_key = "idm_action" if "idm_action" in idm_npz.files else "actions"
            self.idm_actions = np.asarray(idm_npz[idm_key], dtype=np.float32)
        self.action_key = action_key
        self.cache = _NpzLRU(cache_size)
        if len(self.idm_actions) < len(self.rows):
            raise ValueError(
                f"IDM action count ({len(self.idm_actions)}) is smaller than rows ({len(self.rows)})"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        loaded = _load_ref(row["vla_input"], self.cache)
        horizon = int(row.get("target_window_size", row["idm"].get("horizon", 3)))
        action_key = row.get("action_key", self.action_key)

        pos_ref = row["positive"]
        neg_ref = row["negative"]
        pos_traj = self.cache.get(_trajectory_path(pos_ref))
        neg_traj = self.cache.get(_trajectory_path(neg_ref))
        pos_action, pos_mask = _load_window(pos_traj, action_key, int(pos_ref["action_step"]), horizon)
        neg_action, neg_mask = _load_window(neg_traj, action_key, int(neg_ref["action_step"]), horizon)

        idm_idx = int(row["idm"]["index"])
        idm_action = np.asarray(self.idm_actions[idm_idx], dtype=np.float32)[:horizon]
        idm_mask = np.ones((len(idm_action),), dtype=np.float32)
        if len(idm_action) < horizon:
            pad = np.zeros((horizon - len(idm_action), idm_action.shape[-1]), dtype=np.float32)
            idm_action = np.concatenate([idm_action, pad], axis=0)
            idm_mask = np.concatenate([idm_mask, np.zeros((horizon - len(idm_mask),), dtype=np.float32)], axis=0)

        mask = np.minimum(np.minimum(pos_mask, neg_mask), idm_mask)
        return {
            "sample_type": "preference",
            "row": row,
            "images": loaded.images,
            "state": loaded.state,
            "task": row["task"],
            "action_offset": int(row["action_offset"]),
            "positive_action": pos_action,
            "negative_action": neg_action,
            "idm_action": idm_action,
            "target_mask": mask,
        }

    def close(self) -> None:
        self.cache.close()


class GrootPreferenceCollator:
    def __init__(
        self,
        processor: Any,
        *,
        embodiment: Any,
        invert_gripper: bool = True,
        dummy_action_horizon: int = 16,
    ) -> None:
        from gr00t.data.types import MessageType, VLAStepData

        self.processor = processor
        self.embodiment = embodiment
        self.invert_gripper = invert_gripper
        self.dummy_action_horizon = int(dummy_action_horizon)
        self.MessageType = MessageType
        self.VLAStepData = VLAStepData

    def _dummy_action(self) -> dict[str, np.ndarray]:
        zeros = np.zeros((self.dummy_action_horizon, 7), dtype=np.float32)
        return _action_dict_from_flat(zeros, invert_gripper=self.invert_gripper)

    def _normalize_actions(
        self,
        actions: list[np.ndarray],
        states: list[dict[str, np.ndarray]],
    ) -> torch.Tensor:
        normalized = []
        for action, state in zip(actions, states):
            _, norm_action = self.processor.state_action_processor.apply(
                state=state,
                action=_action_dict_from_flat(action, invert_gripper=self.invert_gripper),
                embodiment_tag=self.embodiment.value,
            )
            normalized.append(_concat_action_dict(norm_action))
        return torch.from_numpy(np.stack(normalized).astype(np.float32))

    def _process_inputs(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed = []
        for sample in batch:
            step = self.VLAStepData(
                images=sample["images"],
                states=sample["state"],
                actions=self._dummy_action(),
                text=sample["task"],
                embodiment=self.embodiment,
            )
            processed.append(
                self.processor(
                    [{"type": self.MessageType.EPISODE_STEP.value, "content": step}]
                )
            )
        return self.processor.collator(processed)["inputs"]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = self._process_inputs(batch)
        states = [sample["state"] for sample in batch]
        masks = torch.from_numpy(np.stack([sample["target_mask"] for sample in batch]).astype(np.float32))

        out: dict[str, Any] = {
            "inputs": inputs,
            "target_mask": masks,
            "sample_type": batch[0]["sample_type"],
        }
        if batch[0]["sample_type"] == "success":
            out["target_action"] = self._normalize_actions(
                [sample["target_action"] for sample in batch],
                states,
            )
        else:
            out["action_offset"] = torch.tensor([sample["action_offset"] for sample in batch], dtype=torch.long)
            out["positive_action"] = self._normalize_actions(
                [sample["positive_action"] for sample in batch],
                states,
            )
            out["negative_action"] = self._normalize_actions(
                [sample["negative_action"] for sample in batch],
                states,
            )
            out["idm_action"] = self._normalize_actions(
                [sample["idm_action"] for sample in batch],
                states,
            )
        return out


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_to_device(v, device) for v in batch]
    return batch


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    if isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


def sample_actions_with_grad(
    model: Any,
    inputs: dict[str, torch.Tensor],
    *,
    num_inference_timesteps: int | None = None,
) -> torch.Tensor:
    """Differentiable copy of GR00T-N1.6 action sampling.

    Official ``model.get_action`` is inference-only and wraps the flow-matching
    sampler in ``torch.no_grad``. Preference NCE needs gradients through the
    sampled normalized action chunk, so we mirror the sampler here.
    """
    backbone_inputs, action_inputs = model.prepare_input(inputs)
    backbone_outputs = model.backbone(backbone_inputs)
    head = model.action_head
    features = head._encode_features(backbone_outputs, action_inputs)

    vl_embeds = features.backbone_features
    state_features = features.state_features
    embodiment_id = action_inputs.embodiment_id
    device = vl_embeds.device
    dtype = vl_embeds.dtype
    batch_size = vl_embeds.shape[0]
    steps = int(num_inference_timesteps or head.num_inference_timesteps)
    steps = max(1, steps)

    actions = torch.randn(
        size=(batch_size, head.config.action_horizon, head.action_dim),
        dtype=dtype,
        device=device,
    )
    dt = 1.0 / float(steps)
    for t in range(steps):
        t_cont = t / float(steps)
        t_discretized = int(t_cont * head.num_timestep_buckets)
        timesteps = torch.full((batch_size,), t_discretized, dtype=torch.long, device=device)
        action_features = head.action_encoder(actions, timesteps, embodiment_id)
        if head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)
        sa_embs = torch.cat((state_features, action_features), dim=1)
        if head.config.use_alternate_vl_dit:
            model_output = head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps,
                image_mask=backbone_outputs.image_mask,
                backbone_attention_mask=backbone_outputs.backbone_attention_mask,
            )
        else:
            model_output = head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps,
            )
        pred = head.action_decoder(model_output, embodiment_id)
        actions = actions + dt * pred[:, -head.action_horizon :]
    return actions


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    lambda_gripper: float,
) -> torch.Tensor:
    dim = target.shape[-1]
    pred = pred[..., :dim]
    weights = torch.ones((dim,), dtype=pred.dtype, device=pred.device)
    if dim >= 7:
        weights[6] = float(lambda_gripper)
    sq = (pred - target.to(dtype=pred.dtype)).pow(2) * weights
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    denom = (mask.sum() * weights.sum()).clamp_min(1.0)
    return (sq * mask.unsqueeze(-1)).sum() / denom


def _window_distances(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    lambda_gripper: float,
) -> torch.Tensor:
    dim = target.shape[-1]
    pred = pred[..., :dim]
    weights = torch.ones((dim,), dtype=pred.dtype, device=pred.device)
    if dim >= 7:
        weights[6] = float(lambda_gripper)
    sq = (pred - target.to(dtype=pred.dtype)).pow(2) * weights
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return (sq.sum(dim=-1) * mask).sum(dim=-1) / denom


def bc_loss(
    pred_actions: torch.Tensor,
    batch: dict[str, Any],
    *,
    lambda_gripper: float,
) -> torch.Tensor:
    target = batch["target_action"].to(pred_actions.device)
    mask = batch["target_mask"].to(pred_actions.device)
    pred = pred_actions[:, : target.shape[1], : target.shape[2]]
    return _masked_mse(pred, target, mask, lambda_gripper=lambda_gripper)


def onset_nce_loss(
    pred_actions: torch.Tensor,
    batch: dict[str, Any],
    *,
    tau: float,
    lambda_gripper: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    offsets = batch["action_offset"].to(pred_actions.device)
    pos = batch["positive_action"].to(pred_actions.device)
    neg = batch["negative_action"].to(pred_actions.device)
    mask = batch["target_mask"].to(pred_actions.device)
    h = pos.shape[1]
    windows = torch.stack([pred_actions[i, int(offsets[i]) : int(offsets[i]) + h] for i in range(len(offsets))])
    d_pos = _window_distances(windows, pos, mask, lambda_gripper=lambda_gripper)
    d_neg = _window_distances(windows, neg, mask, lambda_gripper=lambda_gripper)
    loss = F.softplus((d_pos - d_neg) / max(float(tau), 1e-8)).mean()
    return loss, {
        "onset_d_pos": d_pos.detach().mean(),
        "onset_d_neg": d_neg.detach().mean(),
        "onset_prob_pos": torch.sigmoid((d_neg - d_pos) / max(float(tau), 1e-8)).detach().mean(),
    }


def idm_nce_loss(
    pred_actions: torch.Tensor,
    batch: dict[str, Any],
    *,
    tau: float,
    lambda_gripper: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    offsets = batch["action_offset"].to(pred_actions.device)
    idm = batch["idm_action"].to(pred_actions.device)
    neg = batch["negative_action"].to(pred_actions.device)
    mask = batch["target_mask"].to(pred_actions.device)
    h = idm.shape[1]
    windows = torch.stack([pred_actions[i, int(offsets[i]) : int(offsets[i]) + h] for i in range(len(offsets))])
    d_pos = _window_distances(windows, idm, mask, lambda_gripper=lambda_gripper)
    d_neg = _window_distances(windows, neg, mask, lambda_gripper=lambda_gripper)
    loss = F.softplus((d_pos - d_neg) / max(float(tau), 1e-8)).mean()
    return loss, {
        "idm_d_pos": d_pos.detach().mean(),
        "idm_d_neg": d_neg.detach().mean(),
        "idm_prob_pos": torch.sigmoid((d_neg - d_pos) / max(float(tau), 1e-8)).detach().mean(),
    }


def _cycle(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def _split_dataset(dataset: Dataset, *, eval_ratio: float, seed: int) -> tuple[Dataset, Dataset | None]:
    if eval_ratio <= 0.0 or len(dataset) < 2:
        return dataset, None
    eval_len = max(1, int(round(len(dataset) * eval_ratio)))
    eval_len = min(eval_len, len(dataset) - 1)
    train_len = len(dataset) - eval_len
    generator = torch.Generator().manual_seed(seed)
    train_set, eval_set = random_split(dataset, [train_len, eval_len], generator=generator)
    return train_set, eval_set


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    success_loader: DataLoader | None,
    preference_loader: DataLoader | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    step: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    torch.manual_seed(args.eval_seed + step)

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    def add(name: str, value: torch.Tensor | float) -> None:
        totals[name] = totals.get(name, 0.0) + float(value.item() if isinstance(value, torch.Tensor) else value)
        counts[name] = counts.get(name, 0) + 1

    if success_loader is not None:
        for batch_idx, batch in enumerate(success_loader):
            if batch_idx >= args.eval_batches:
                break
            batch = _to_device(batch, device)
            pred = sample_actions_with_grad(
                model,
                _rec_to_dtype(batch["inputs"], dtype),
                num_inference_timesteps=args.num_inference_timesteps,
            )
            add(
                "loss_bc",
                bc_loss(pred, batch, lambda_gripper=args.lambda_gripper_bc),
            )

    if preference_loader is not None:
        for batch_idx, batch in enumerate(preference_loader):
            if batch_idx >= args.eval_batches:
                break
            batch = _to_device(batch, device)
            pred = sample_actions_with_grad(
                model,
                _rec_to_dtype(batch["inputs"], dtype),
                num_inference_timesteps=args.num_inference_timesteps,
            )
            loss_onset, onset_metrics = onset_nce_loss(
                pred,
                batch,
                tau=args.tau_onset,
                lambda_gripper=args.lambda_gripper_nce,
            )
            loss_idm, idm_metrics = idm_nce_loss(
                pred,
                batch,
                tau=args.tau_idm,
                lambda_gripper=args.lambda_gripper_nce,
            )
            add("loss_onset", loss_onset)
            add("loss_idm", loss_idm)
            for key, value in {**onset_metrics, **idm_metrics}.items():
                add(key, value)

    metrics = {key: totals[key] / max(1, counts[key]) for key in totals}
    metrics["loss_total"] = (
        args.lambda_bc * metrics.get("loss_bc", 0.0)
        + args.lambda_onset * metrics.get("loss_onset", 0.0)
        + args.lambda_idm * metrics.get("loss_idm", 0.0)
    )
    if was_training:
        model.train()
    return metrics


def _maybe_apply_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if not args.use_lora:
        return model
    from peft import LoraConfig, get_peft_model

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _save_checkpoint(
    model: torch.nn.Module,
    processor: Any,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
) -> None:
    ckpt_dir = output_dir / f"ckpt-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir / "processor")
    (ckpt_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    LOGGER.info("Saved checkpoint: %s", ckpt_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", default="0xAnkitSingh/GR00T-N1.6-LIBERO")
    parser.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    parser.add_argument("--preference-pairs", required=True)
    parser.add_argument("--idm-actions", required=True)
    parser.add_argument("--success-only", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--success-batch-size", type=int, default=1)
    parser.add_argument("--preference-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lr-scheduler", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--num-inference-timesteps", type=int, default=4)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--lambda-onset", type=float, default=0.1)
    parser.add_argument("--lambda-idm", type=float, default=0.05)
    parser.add_argument("--lambda-gripper-bc", type=float, default=1.0)
    parser.add_argument("--lambda-gripper-nce", type=float, default=1.0)
    parser.add_argument("--tau-onset", type=float, default=1.0)
    parser.add_argument("--tau-idm", type=float, default=1.0)
    parser.add_argument("--nce-warmup-steps", type=int, default=1000)
    parser.add_argument("--invert-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,out_proj,fc1,fc2,to_q,to_k,to_v,to_out.0,proj_out_1,proj_out_2",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    _setup_logging(output_dir)
    _load_dotenv()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Register GR00T model/processor classes with transformers.
    import gr00t.model  # noqa: F401
    from gr00t.data.embodiment_tags import EmbodimentTag

    tag = getattr(EmbodimentTag, args.embodiment_tag, args.embodiment_tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32

    LOGGER.info("Loading GR00T model from %s", args.base_model_path)
    model = AutoModel.from_pretrained(args.base_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    processor.train()
    model = _maybe_apply_lora(model, args)
    model.to(device=device, dtype=dtype)
    model.train()

    collator = GrootPreferenceCollator(
        processor,
        embodiment=tag,
        invert_gripper=args.invert_gripper,
        dummy_action_horizon=args.chunk_size,
    )
    success_dataset = SuccessOnlyDataset(
        Path(args.success_only),
        action_window=args.chunk_size,
        cache_size=args.cache_size,
    )
    preference_dataset = PreferencePairDataset(
        Path(args.preference_pairs),
        idm_actions_path=Path(args.idm_actions),
        cache_size=args.cache_size,
    )
    success_train, success_eval = _split_dataset(
        success_dataset,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    preference_train, preference_eval = _split_dataset(
        preference_dataset,
        eval_ratio=args.eval_ratio,
        seed=args.seed + 1,
    )
    success_loader = DataLoader(
        success_train,
        batch_size=args.success_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
    )
    preference_loader = DataLoader(
        preference_train,
        batch_size=args.preference_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
    )
    success_eval_loader = (
        DataLoader(
            success_eval,
            batch_size=args.success_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            drop_last=False,
        )
        if success_eval is not None
        else None
    )
    preference_eval_loader = (
        DataLoader(
            preference_eval,
            batch_size=args.preference_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            drop_last=False,
        )
        if preference_eval is not None
        else None
    )
    LOGGER.info(
        "Loaded datasets: success=%d (%d train / %d eval) preference=%d (%d train / %d eval)",
        len(success_dataset),
        len(success_train),
        0 if success_eval is None else len(success_eval),
        len(preference_dataset),
        len(preference_train),
        0 if preference_eval is None else len(preference_eval),
    )

    if args.dry_run:
        success_batch = next(iter(success_loader))
        preference_batch = next(iter(preference_loader))
        LOGGER.info("Dry-run success input keys: %s", sorted(success_batch["inputs"].keys()))
        LOGGER.info("Dry-run preference input keys: %s", sorted(preference_batch["inputs"].keys()))
        return

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.lr_scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
        )

    wandb_run = None
    if args.wandb_mode != "disabled" and args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or output_dir.name,
            mode=args.wandb_mode,
            config=vars(args),
        )

    success_iter = _cycle(success_loader)
    preference_iter = _cycle(preference_loader)
    optimizer.zero_grad(set_to_none=True)
    running: dict[str, float] = {}

    pbar = tqdm(range(1, args.max_steps + 1), desc="GR00T preference fine-tune")
    for step in pbar:
        success_batch = _to_device(next(success_iter), device)
        preference_batch = _to_device(next(preference_iter), device)

        success_inputs = _rec_to_dtype(success_batch["inputs"], dtype)
        preference_inputs = _rec_to_dtype(preference_batch["inputs"], dtype)

        pred_success = sample_actions_with_grad(
            model,
            success_inputs,
            num_inference_timesteps=args.num_inference_timesteps,
        )
        pred_preference = sample_actions_with_grad(
            model,
            preference_inputs,
            num_inference_timesteps=args.num_inference_timesteps,
        )

        loss_bc = bc_loss(
            pred_success,
            success_batch,
            lambda_gripper=args.lambda_gripper_bc,
        )
        loss_onset, onset_metrics = onset_nce_loss(
            pred_preference,
            preference_batch,
            tau=args.tau_onset,
            lambda_gripper=args.lambda_gripper_nce,
        )
        loss_idm, idm_metrics = idm_nce_loss(
            pred_preference,
            preference_batch,
            tau=args.tau_idm,
            lambda_gripper=args.lambda_gripper_nce,
        )

        nce_scale = min(1.0, step / max(1, args.nce_warmup_steps))
        lambda_onset = args.lambda_onset * nce_scale
        lambda_idm = args.lambda_idm * nce_scale
        loss = args.lambda_bc * loss_bc + lambda_onset * loss_onset + lambda_idm * loss_idm
        (loss / args.gradient_accumulation_steps).backward()

        grad_norm = None
        if step % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        metrics = {
            "loss_total": float(loss.detach().item()),
            "loss_bc": float(loss_bc.detach().item()),
            "loss_onset": float(loss_onset.detach().item()),
            "loss_idm": float(loss_idm.detach().item()),
            "lambda_onset": float(lambda_onset),
            "lambda_idm": float(lambda_idm),
            "lr": float(scheduler.get_last_lr()[0]),
            **{k: float(v.item()) for k, v in onset_metrics.items()},
            **{k: float(v.item()) for k, v in idm_metrics.items()},
        }
        if grad_norm is not None:
            metrics["grad_norm"] = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        running = metrics
        if step % args.logging_steps == 0:
            LOGGER.info(
                "[%d/%d] loss=%.4f bc=%.4f onset=%.4f idm=%.4f lr=%.2e",
                step,
                args.max_steps,
                metrics["loss_total"],
                metrics["loss_bc"],
                metrics["loss_onset"],
                metrics["loss_idm"],
                metrics["lr"],
            )
            pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items() if k in {"loss_total", "loss_bc", "loss_onset", "loss_idm"}})
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

        if args.eval_steps > 0 and step % args.eval_steps == 0:
            eval_metrics = evaluate(
                model,
                success_eval_loader,
                preference_eval_loader,
                device=device,
                dtype=dtype,
                args=args,
                step=step,
            )
            LOGGER.info(
                "[%d/%d] eval_loss=%.4f eval_bc=%.4f eval_onset=%.4f eval_idm=%.4f",
                step,
                args.max_steps,
                eval_metrics.get("loss_total", 0.0),
                eval_metrics.get("loss_bc", 0.0),
                eval_metrics.get("loss_onset", 0.0),
                eval_metrics.get("loss_idm", 0.0),
            )
            running.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            if wandb_run is not None:
                wandb_run.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)

        if args.save_steps > 0 and step % args.save_steps == 0:
            _save_checkpoint(model, processor, output_dir, step, args)

    _save_checkpoint(model, processor, output_dir, args.max_steps, args)
    (output_dir / "summary.json").write_text(
        json.dumps({"last_step": args.max_steps, "last_metrics": running}, indent=2, sort_keys=True)
    )
    if wandb_run is not None:
        wandb_run.finish()
    success_dataset.close()
    preference_dataset.close()


if __name__ == "__main__":
    main()
