"""LoRA fine-tuning entrypoint for X-VLA with rollout manifests."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.backends.cudnn as cudnn
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XVLA_ROOT = REPO_ROOT.parent / "vla-bench" / "third_party" / "models" / "x-vla"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DEFAULT_XVLA_ROOT) not in sys.path:
    sys.path.append(str(DEFAULT_XVLA_ROOT))

from models.modeling_xvla import XVLA  # type: ignore  # noqa: E402
from models.processing_xvla import XVLAProcessor  # type: ignore  # noqa: E402

from train.xvla_loss import (
    TorchActionStats,
    compute_distance_probabilistic_nce_xvla_loss,
    compute_normalized_squared_hinge_triplet_xvla_loss,
    compute_preference_kl_loss,
)
from train.xvla_manifest_dataset import XVLAManifestCollator, XVLAManifestDataset


def load_env_file(path: str | Path) -> bool:
    """Load KEY=VALUE pairs from a .env file into the process environment."""
    env_path = Path(path)
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return True


def auto_load_dotenv() -> Path | None:
    """Try common project-local .env locations without requiring extra deps."""
    candidates = [Path.cwd() / ".env", REPO_ROOT / ".env"]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if load_env_file(resolved):
            return resolved
    return None


def get_logger(name: str = "xvla_peft", output_dir: str | Path | None = None, accelerator: Accelerator | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    is_main = accelerator is None or accelerator.is_main_process
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    if is_main:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    if output_dir and is_main:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(output_dir) / "train.log", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("X-VLA rollout PEFT fine-tuning", add_help=False)
    parser.add_argument("--models", type=str, required=True, help="Path or HF repo for pretrained X-VLA")
    parser.add_argument("--output_dir", type=str, default=None, help="Checkpoint/output directory")
    parser.add_argument("--manifest_dir", type=str, required=True, help="Directory with success_only.jsonl and triplets.jsonl")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval in optimizer steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain_id", type=int, default=2, help="CALVIN domain id")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lambda_pos", type=float, default=1.0)
    parser.add_argument("--lambda_ctr", type=float, default=1.0)
    parser.add_argument(
        "--loss_type",
        choices=(
            "triplet_squared_hinge",
            "distance_probabilistic_nce",
            "triplet_hinge",
            "probabilistic_nce",
        ),
        default="triplet_squared_hinge",
        help=(
            "Contrastive loss objective. triplet_squared_hinge keeps the existing "
            "squared-hinge triplet loss; distance_probabilistic_nce uses action-distance NCE. "
            "triplet_hinge/probabilistic_nce are legacy aliases."
        ),
    )
    parser.add_argument("--lambda_nce", type=float, default=1.0, help="Weight for probabilistic NCE loss.")
    parser.add_argument("--nce_tau", type=float, default=1.0, help="Temperature tau for probabilistic NCE/KL logits.")
    parser.add_argument("--lambda_gripper_nce", type=float, default=1.0, help="Gripper distance weight in NCE/KL distances.")
    parser.add_argument("--use_preference_kl", action="store_true", default=False, help="Add frozen base-model preference KL.")
    parser.add_argument("--lambda_kl", type=float, default=0.0, help="Weight for optional teacher preference KL.")
    # TODO: Evaluate hinge (non-squared) contrastive loss plus lambda_ctr warmup
    # after checking the effect of frequency/confidence triplet weighting.
    parser.add_argument("--use_triplet_balance_weights", action="store_true", default=False)
    parser.add_argument("--triplet_weight_min", type=float, default=0.5)
    parser.add_argument("--triplet_weight_max", type=float, default=2.0)
    parser.add_argument("--triplet_confidence_events", type=int, default=10)
    parser.add_argument("--stats_path", type=str, default=None, help="Optional cached action_stats.json path")
    parser.add_argument("--eval_split_ratio", type=float, default=0.1, help="Fraction of samples used for eval")
    parser.add_argument("--wandb_project", type=str, default="XVLA-LoRA-Training")
    parser.add_argument("--run_name", type=str, default="xvla-calvin-lora-new-260424")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument(
        "--lora_only",
        action="store_true",
        default=False,
        help="Train only LoRA adapter weights; keep soft prompts/action encoder/action decoder frozen.",
    )
    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def build_optimizer(model: XVLA, lr: float, weight_decay: float, betas: tuple[float, float], lr_coef_soft: float = 1.0) -> AdamW:
    vlm_params = list(model.vlm.parameters())
    soft_prompt_params = list(model.transformer.soft_prompt_hub.parameters())
    action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    exclude = set(map(id, vlm_params + soft_prompt_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]

    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "soft_prompts", "params": soft_prompt_params, "lr": lr * lr_coef_soft, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, betas=betas)


def set_group_lr(optimizer: AdamW, name: str, lr: float) -> None:
    for group in optimizer.param_groups:
        if group["name"] == name:
            group["lr"] = lr


def linear_warmup_cosine(step: int, start: int, warmup: int, total: int, base_lr: float, min_ratio: float) -> float:
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optimizer: AdamW, step: int, args: argparse.Namespace) -> None:
    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
    }
    if step < args.freeze_steps:
        set_group_lr(optimizer, "vlm", 0.0)
        set_group_lr(optimizer, "transformer_core", 0.0)
        set_group_lr(optimizer, "soft_prompts", base["soft_prompts"])
        set_group_lr(optimizer, "action_heads", base["action_heads"])
        return
    for name, base_lr in base.items():
        lr = linear_warmup_cosine(step, args.freeze_steps, args.warmup_steps, args.iters, base_lr, args.min_lr_ratio) if args.use_cosine_decay else base_lr
        set_group_lr(optimizer, name, lr)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for key, value in batch.items():
        outputs[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return outputs


def make_action_noise_context(model: XVLA, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Create the diffusion timestep/noisy action shared by student and teacher."""
    positive_action_seq = batch["positive_action_seq"]
    batch_size = batch["input_ids"].shape[0]
    device = batch["input_ids"].device
    t = (torch.rand(1, device=device) + torch.arange(batch_size, device=device) / batch_size) % (1 - 1e-5)
    action_noisy = torch.randn_like(positive_action_seq) * t.view(-1, 1, 1) + positive_action_seq * (1 - t).view(-1, 1, 1)
    proprio_m, action_noisy_m = model.action_space.preprocess(batch["proprio"], action_noisy)
    return {
        "t": t,
        "proprio_m": proprio_m,
        "action_noisy_m": action_noisy_m,
    }


def predict_action_sequence(model: XVLA, batch: dict[str, Any], noise_context: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    """Mirror XVLA.forward but return predicted action sequence for custom loss."""
    enc = model.forward_vlm(batch["input_ids"], batch["image_input"], batch["image_mask"])
    if noise_context is None:
        noise_context = make_action_noise_context(model, batch)
    pred_action = model.transformer(
        domain_id=batch["domain_id"],
        action_with_noise=noise_context["action_noisy_m"],
        t=noise_context["t"],
        proprio=noise_context["proprio_m"],
        **enc,
    )
    return pred_action


def compute_loss_for_batch(
    model: XVLA,
    batch: dict[str, Any],
    stats_torch: TorchActionStats,
    args: argparse.Namespace,
    *,
    teacher_model: XVLA | None = None,
) -> dict[str, torch.Tensor]:
    noise_context = make_action_noise_context(model, batch)
    pred_action_seq = predict_action_sequence(model, batch, noise_context=noise_context)
    if args.loss_type in {"triplet_squared_hinge", "triplet_hinge"}:
        loss_dict = compute_normalized_squared_hinge_triplet_xvla_loss(
            pred_action_seq=pred_action_seq,
            positive_action_first=batch["positive_action_first"],
            negative_action_first=batch["negative_action_first"],
            has_negative=batch["has_negative"],
            sample_weight=batch.get("sample_weight"),
            stats=stats_torch,
            margin=args.margin,
            lambda_pos=args.lambda_pos,
            lambda_ctr=args.lambda_ctr,
        )
    elif args.loss_type in {"distance_probabilistic_nce", "probabilistic_nce"}:
        loss_dict = compute_distance_probabilistic_nce_xvla_loss(
            pred_action_seq=pred_action_seq,
            positive_action_first=batch["positive_action_first"],
            negative_action_first=batch["negative_action_first"],
            has_negative=batch["has_negative"],
            sample_weight=batch.get("sample_weight"),
            stats=stats_torch,
            tau=args.nce_tau,
            lambda_pos=args.lambda_pos,
            lambda_nce=args.lambda_nce,
            lambda_gripper_nce=args.lambda_gripper_nce,
        )
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")

    if args.use_preference_kl:
        if args.loss_type not in {"distance_probabilistic_nce", "probabilistic_nce"}:
            raise ValueError("--use_preference_kl requires --loss_type distance_probabilistic_nce")
        if teacher_model is None:
            raise ValueError("teacher_model is required when --use_preference_kl is enabled")
        with torch.no_grad():
            teacher_pred_action_seq = predict_action_sequence(teacher_model, batch, noise_context=noise_context)
        kl_dict = compute_preference_kl_loss(
            student_pred_action_seq=pred_action_seq,
            teacher_pred_action_seq=teacher_pred_action_seq,
            positive_action_first=batch["positive_action_first"],
            negative_action_first=batch["negative_action_first"],
            has_negative=batch["has_negative"],
            sample_weight=batch.get("sample_weight"),
            stats=stats_torch,
            tau=args.nce_tau,
            lambda_gripper_kl=args.lambda_gripper_nce,
        )
        loss_dict = dict(loss_dict)
        loss_dict.update(kl_dict)
        loss_dict["loss_total"] = loss_dict["loss_total"] + args.lambda_kl * kl_dict["loss_kl"]

    return loss_dict


def save_checkpoint(accelerator: Accelerator, model: XVLA, output_dir: Path, global_step: int, stats_path: Path) -> None:
    save_dir = output_dir / f"ckpt-{global_step}"
    accelerator.print(f"Saving checkpoint to {save_dir}")
    accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
    (save_dir / "state.json").write_text(json.dumps({"global_step": global_step}, indent=2))
    if stats_path.exists():
        (save_dir / "action_stats.json").write_text(stats_path.read_text())


def split_indices(num_items: int, eval_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(num_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    eval_size = max(1, int(round(num_items * eval_ratio)))
    eval_size = min(eval_size, num_items - 1)
    eval_indices = sorted(indices[:eval_size])
    train_indices = sorted(indices[eval_size:])
    return train_indices, eval_indices


def evaluate(
    model: XVLA,
    eval_loader: DataLoader,
    accelerator: Accelerator,
    stats_torch: TorchActionStats,
    args: argparse.Namespace,
    teacher_model: XVLA | None = None,
) -> dict[str, float]:
    model.eval()
    if teacher_model is not None:
        teacher_model.eval()
    totals: dict[str, float] = {}
    metric_keys: list[str] | None = None
    total_samples = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            batch = move_batch_to_device(batch, accelerator.device)
            loss_dict = compute_loss_for_batch(model, batch, stats_torch, args, teacher_model=teacher_model)
            batch_size = float(batch["positive_action_first"].shape[0])
            if metric_keys is None:
                metric_keys = list(loss_dict.keys())
                totals = {key: 0.0 for key in metric_keys}
            stats_tensor = torch.tensor(
                [loss_dict[key].detach().float().item() * batch_size for key in metric_keys] + [batch_size],
                device=accelerator.device,
            )
            stats_tensor = accelerator.reduce(stats_tensor, reduction="sum")
            for idx, key in enumerate(metric_keys):
                totals[key] += float(stats_tensor[idx].item())
            total_samples += float(stats_tensor[-1].item())
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    denom = max(total_samples, 1.0)
    return {key: value / denom for key, value in totals.items()}


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / "train" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    accelerator = Accelerator(log_with="wandb", project_dir=output_dir)
    accelerator.init_trackers(
        args.wandb_project,
        config=vars(args),
        init_kwargs={"wandb": {"name": args.run_name, "dir": str(output_dir)}},
    )
    accelerator.wait_for_everyone()
    logger = get_logger(output_dir=output_dir, accelerator=accelerator)

    dotenv_path = auto_load_dotenv()
    set_seed(args.seed + accelerator.process_index)
    logger.info("Args: %s", args)
    if dotenv_path is not None:
        logger.info("Loaded environment variables from %s", dotenv_path)
    logger.info("PYTORCH_CUDA_ALLOC_CONF=%s", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    if accelerator.is_main_process:
        (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    model = XVLA.from_pretrained(args.models)
    processor = XVLAProcessor.from_pretrained(args.models)
    modules_to_save = None if args.lora_only else [
        "transformer.soft_prompt_hub",
        "transformer.action_encoder",
        "transformer.action_decoder",
    ]
    logger.info("LoRA modules_to_save=%s", modules_to_save)
    lora_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        r=args.lora_r,
        bias="none",
        target_modules="all-linear",
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    teacher_model: XVLA | None = None
    if args.use_preference_kl:
        logger.info("Loading frozen teacher model for preference KL from %s", args.models)
        teacher_model = XVLA.from_pretrained(args.models)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)

    manifest_dir = Path(args.manifest_dir)
    manifest_paths = [
        manifest_dir / "success_only.jsonl",
        manifest_dir / "triplets.jsonl",
    ]
    dataset = XVLAManifestDataset(
        [str(path) for path in manifest_paths],
        num_actions=model.num_actions,
        domain_id=args.domain_id,
        repo_root=REPO_ROOT,
        use_triplet_balance_weights=args.use_triplet_balance_weights,
        triplet_weight_min=args.triplet_weight_min,
        triplet_weight_max=args.triplet_weight_max,
        triplet_confidence_events=args.triplet_confidence_events,
    )
    logger.info("Triplet balance weights: %s", json.dumps(dataset.triplet_weight_summary, sort_keys=True))
    train_indices, eval_indices = split_indices(len(dataset), args.eval_split_ratio, args.seed)
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)

    stats_path = Path(args.stats_path) if args.stats_path else output_dir / "action_stats.json"
    if stats_path.exists():
        action_stats = dataset.load_action_stats(stats_path)
        logger.info("Loaded action stats from %s", stats_path)
    else:
        action_stats = dataset.save_action_stats(stats_path, indices=train_indices)
        logger.info("Computed action stats and saved to %s", stats_path)

    collator = XVLAManifestCollator(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    optimizer = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_soft=args.learning_coef,
    )
    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)
    if teacher_model is not None:
        teacher_model.to(accelerator.device)
    stats_torch = TorchActionStats.from_python(action_stats, accelerator.device)

    model.train()
    global_step = 0
    log_start = time.time()
    summary: dict[str, Any] = {
        "run_name": args.run_name,
        "wandb_project": args.wandb_project,
        "output_dir": str(output_dir),
        "train_size": len(train_indices),
        "eval_size": len(eval_indices),
        "stats_path": str(stats_path),
        "triplet_weight_summary": dataset.triplet_weight_summary,
    }
    logger.info(
        "Start training for %d iterations | train=%d eval=%d | world_size=%d",
        args.iters,
        len(train_indices),
        len(eval_indices),
        accelerator.num_processes,
    )

    progress = tqdm(total=args.iters, disable=not accelerator.is_main_process, desc=args.run_name, dynamic_ncols=True)
    train_iter = iter(train_loader)

    while global_step < args.iters:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = move_batch_to_device(batch, accelerator.device)
        update_group_lrs(optimizer, global_step, args)

        loss_dict = compute_loss_for_batch(model, batch, stats_torch, args, teacher_model=teacher_model)
        loss = loss_dict["loss_total"]

        accelerator.backward(loss)
        grad_norm = None
        if args.max_grad_norm:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if global_step % args.log_interval == 0:
            logs = {
                f"train/{key}": float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
                for key, value in loss_dict.items()
            }
            if grad_norm is not None:
                logs["train/grad_norm"] = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            logs.update({f"lr/{group['name']}": float(group["lr"]) for group in optimizer.param_groups})
            accelerator.log(logs, step=global_step)
            summary["last_train_step"] = global_step
            for key, value in logs.items():
                if key.startswith("train/"):
                    summary[f"last_{key.replace('/', '_')}"] = value
            if accelerator.is_main_process:
                dt = (time.time() - log_start) / max(1, args.log_interval)
                log_start = time.time()
                logger.info(
                    "[%d/%d] train_loss=%.4f train_pos=%.4f train_ctr=%.4f triplet_frac=%.3f triplet_w=%.3f lr=%.2e (%.2fs/it)",
                    global_step,
                    args.iters,
                    logs["train/loss_total"],
                    logs["train/loss_pos"],
                    logs["train/loss_ctr"],
                    logs["train/triplet_fraction"],
                    logs["train/triplet_weight_mean"],
                    logs["lr/transformer_core"],
                    dt,
                )
                progress.set_postfix(
                    train_loss=f"{logs['train/loss_total']:.4f}",
                    eval_loss=f"{summary.get('last_eval_loss_total', float('nan')):.4f}" if "last_eval_loss_total" in summary else "n/a",
                )

        if global_step % args.eval_interval == 0:
            eval_metrics = evaluate(model, eval_loader, accelerator, stats_torch, args, teacher_model=teacher_model)
            accelerator.log({f"eval/{key}": value for key, value in eval_metrics.items()}, step=global_step)
            summary["last_eval_step"] = global_step
            for key, value in eval_metrics.items():
                summary[f"last_eval_{key}"] = value
            if accelerator.is_main_process:
                logger.info(
                    "[%d/%d] eval_loss=%.4f eval_pos=%.4f eval_ctr=%.4f",
                    global_step,
                    args.iters,
                    eval_metrics["loss_total"],
                    eval_metrics["loss_pos"],
                    eval_metrics["loss_ctr"],
                )

        global_step += 1
        progress.update(1)
        if accelerator.is_main_process and (global_step == args.iters or global_step % args.save_interval == 0):
            save_checkpoint(accelerator, model, output_dir, global_step, stats_path)

    progress.close()
    summary["final_global_step"] = global_step
    if accelerator.is_main_process:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("X-VLA rollout PEFT", parents=[get_args_parser()])
    main(parser.parse_args())
