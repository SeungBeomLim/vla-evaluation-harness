"""Custom normalized loss for X-VLA rollout fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TorchActionStats:
    pos_mean: torch.Tensor
    pos_std: torch.Tensor
    rot_mean: torch.Tensor
    rot_std: torch.Tensor

    @classmethod
    def from_python(cls, stats: Any, device: torch.device) -> "TorchActionStats":
        return cls(
            pos_mean=torch.tensor(stats.pos_mean, dtype=torch.float32, device=device),
            pos_std=torch.tensor(stats.pos_std, dtype=torch.float32, device=device),
            rot_mean=torch.tensor(stats.rot_mean, dtype=torch.float32, device=device),
            rot_std=torch.tensor(stats.rot_std, dtype=torch.float32, device=device),
        )


class ActionEmbedder(nn.Module):
    """Projection head for embedding-based action preference learning."""

    def __init__(
        self,
        action_dim: int = 10,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        act: type[nn.Module]
        if activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, action: torch.Tensor, *, return_pre_norm: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = self.net(action)
        z_norm = F.normalize(z, dim=-1, eps=1e-8)
        if return_pre_norm:
            return z_norm, z
        return z_norm


def _normalize_components(action_10d: torch.Tensor, stats: TorchActionStats) -> tuple[torch.Tensor, torch.Tensor]:
    pos = (action_10d[..., 0:3] - stats.pos_mean) / stats.pos_std
    rot = (action_10d[..., 3:9] - stats.rot_mean) / stats.rot_std
    return pos, rot


def _bc_components(
    pred_action_first: torch.Tensor,
    positive_action_first: torch.Tensor,
    stats: TorchActionStats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_pos_n, pred_rot_n = _normalize_components(pred_action_first, stats)
    pos_pos_n, pos_rot_n = _normalize_components(positive_action_first, stats)

    pos_loss = F.mse_loss(pred_pos_n, pos_pos_n, reduction="none").mean(dim=-1)
    rot_loss = F.mse_loss(pred_rot_n, pos_rot_n, reduction="none").mean(dim=-1)
    grip_loss = F.binary_cross_entropy_with_logits(
        pred_action_first[..., 9],
        positive_action_first[..., 9],
        reduction="none",
    )
    return pos_loss + rot_loss + grip_loss, pos_loss, rot_loss, grip_loss


def _embedding_action_input(action_10d: torch.Tensor, stats: TorchActionStats, *, gripper_is_logit: bool) -> torch.Tensor:
    pos_n, rot_n = _normalize_components(action_10d, stats)
    gripper = torch.sigmoid(action_10d[..., 9]) if gripper_is_logit else action_10d[..., 9]
    return torch.cat([pos_n, rot_n, gripper.unsqueeze(-1)], dim=-1)


def _preference_squared_distances(
    pred_action_first: torch.Tensor,
    positive_action_first: torch.Tensor,
    negative_action_first: torch.Tensor,
    stats: TorchActionStats,
    *,
    lambda_gripper: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_pos_n, pred_rot_n = _normalize_components(pred_action_first, stats)
    pos_pos_n, pos_rot_n = _normalize_components(positive_action_first, stats)
    neg_pos_n, neg_rot_n = _normalize_components(negative_action_first, stats)

    pred_grip_prob = torch.sigmoid(pred_action_first[..., 9])
    positive_grip = positive_action_first[..., 9]
    negative_grip = negative_action_first[..., 9]

    d_pos = (
        (pred_pos_n - pos_pos_n).pow(2).sum(dim=-1)
        + (pred_rot_n - pos_rot_n).pow(2).sum(dim=-1)
        + float(lambda_gripper) * (pred_grip_prob - positive_grip).pow(2)
    )
    d_neg = (
        (pred_pos_n - neg_pos_n).pow(2).sum(dim=-1)
        + (pred_rot_n - neg_rot_n).pow(2).sum(dim=-1)
        + float(lambda_gripper) * (pred_grip_prob - negative_grip).pow(2)
    )
    return d_pos, d_neg


def _masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None,
    *,
    average_over_batch: bool = True,
) -> torch.Tensor:
    masked = torch.where(mask, values, torch.zeros_like(values))
    if sample_weight is not None:
        masked = masked * sample_weight.to(device=values.device, dtype=values.dtype)
    if average_over_batch:
        return masked.mean()
    denom = mask.float().sum().clamp_min(1.0).to(dtype=values.dtype)
    return masked.sum() / denom


def compute_normalized_squared_hinge_triplet_xvla_loss(
    pred_action_seq: torch.Tensor,
    positive_action_first: torch.Tensor,
    negative_action_first: torch.Tensor,
    has_negative: torch.Tensor,
    stats: TorchActionStats,
    *,
    sample_weight: torch.Tensor | None = None,
    margin: float = 1.0,
    lambda_pos: float = 1.0,
    lambda_ctr: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute normalized BC + squared-hinge triplet contrastive loss."""
    pred = pred_action_seq[:, 0, :10]

    pred_pos_n, pred_rot_n = _normalize_components(pred, stats)
    pos_pos_n, pos_rot_n = _normalize_components(positive_action_first, stats)

    pred_grip_logit = pred[..., 9]
    positive_grip = positive_action_first[..., 9]

    pos_loss = F.mse_loss(pred_pos_n, pos_pos_n, reduction="none").mean(dim=-1)
    rot_loss = F.mse_loss(pred_rot_n, pos_rot_n, reduction="none").mean(dim=-1)
    grip_loss = F.binary_cross_entropy_with_logits(
        pred_grip_logit,
        positive_grip,
        reduction="none",
    )
    l_pos = pos_loss + rot_loss + grip_loss

    l_ctr = torch.zeros_like(l_pos)
    ctr_weight = (
        torch.ones_like(l_pos)
        if sample_weight is None
        else sample_weight.to(device=l_pos.device, dtype=l_pos.dtype)
    )
    if has_negative.any():
        neg_pos_n, neg_rot_n = _normalize_components(negative_action_first, stats)
        pred_ctr = torch.cat(
            [pred_pos_n, pred_rot_n, torch.sigmoid(pred_grip_logit).unsqueeze(-1)],
            dim=-1,
        )
        pos_ctr = torch.cat([pos_pos_n, pos_rot_n, positive_grip.unsqueeze(-1)], dim=-1)
        neg_ctr = torch.cat([neg_pos_n, neg_rot_n, negative_action_first[..., 9].unsqueeze(-1)], dim=-1)

        d_pos = torch.linalg.norm(pred_ctr - pos_ctr, dim=-1)
        d_neg = torch.linalg.norm(pred_ctr - neg_ctr, dim=-1)
        violation = d_pos - d_neg + margin
        l_ctr = torch.where(has_negative, F.relu(violation).pow(2), torch.zeros_like(violation))

    loss_pos = l_pos.mean()
    loss_ctr = (l_ctr * ctr_weight).mean()
    loss_total = lambda_pos * loss_pos + lambda_ctr * loss_ctr

    return {
        "loss_total": loss_total,
        "loss_pos": loss_pos,
        "loss_ctr": loss_ctr,
        "loss_pos_position": pos_loss.mean(),
        "loss_pos_rotation": rot_loss.mean(),
        "loss_pos_gripper": grip_loss.mean(),
        "triplet_fraction": has_negative.float().mean(),
        "triplet_weight_mean": ctr_weight[has_negative].mean()
        if has_negative.any()
        else ctr_weight.new_tensor(0.0),
    }


def compute_distance_probabilistic_nce_xvla_loss(
    pred_action_seq: torch.Tensor,
    positive_action_first: torch.Tensor,
    negative_action_first: torch.Tensor,
    has_negative: torch.Tensor,
    stats: TorchActionStats,
    *,
    sample_weight: torch.Tensor | None = None,
    tau: float = 1.0,
    lambda_pos: float = 1.0,
    lambda_nce: float = 1.0,
    lambda_gripper_nce: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute BC + distance-based probabilistic two-way NCE preference loss.

    The NCE term treats the model action as the mean of a fixed-variance
    Gaussian and optimizes ``-log P(a+ > a- | o)``:

        softplus((d_pos - d_neg) / tau)

    where ``d_pos`` and ``d_neg`` are normalized squared distances from the
    prediction to positive/negative actions.  Gripper distance is measured in
    probability space, while BC keeps BCEWithLogits for the binary target.
    """
    pred = pred_action_seq[:, 0, :10]

    pred_pos_n, pred_rot_n = _normalize_components(pred, stats)
    pos_pos_n, pos_rot_n = _normalize_components(positive_action_first, stats)

    pred_grip_logit = pred[..., 9]
    positive_grip = positive_action_first[..., 9]

    pos_loss = F.mse_loss(pred_pos_n, pos_pos_n, reduction="none").mean(dim=-1)
    rot_loss = F.mse_loss(pred_rot_n, pos_rot_n, reduction="none").mean(dim=-1)
    grip_loss = F.binary_cross_entropy_with_logits(
        pred_grip_logit,
        positive_grip,
        reduction="none",
    )
    l_bc = pos_loss + rot_loss + grip_loss

    tau_tensor = pred.new_tensor(max(float(tau), 1e-8))
    d_pos, d_neg = _preference_squared_distances(
        pred,
        positive_action_first,
        negative_action_first,
        stats,
        lambda_gripper=lambda_gripper_nce,
    )
    nce_logits = (d_neg - d_pos) / tau_tensor
    l_nce_all = F.softplus(-nce_logits)
    loss_nce = _masked_weighted_mean(l_nce_all, has_negative, sample_weight)

    loss_bc = l_bc.mean()
    loss_total = lambda_pos * loss_bc + lambda_nce * loss_nce

    active = has_negative
    active_count = active.float().sum().clamp_min(1.0).to(dtype=pred.dtype)
    d_pos_mean = torch.where(active, d_pos, torch.zeros_like(d_pos)).sum() / active_count
    d_neg_mean = torch.where(active, d_neg, torch.zeros_like(d_neg)).sum() / active_count
    logit_mean = torch.where(active, nce_logits, torch.zeros_like(nce_logits)).sum() / active_count
    prob_mean = torch.where(active, torch.sigmoid(nce_logits), torch.zeros_like(nce_logits)).sum() / active_count
    ctr_weight = (
        torch.ones_like(l_bc)
        if sample_weight is None
        else sample_weight.to(device=l_bc.device, dtype=l_bc.dtype)
    )

    return {
        "loss_total": loss_total,
        "loss_pos": loss_bc,
        "loss_ctr": loss_nce,
        "loss_bc": loss_bc,
        "loss_nce": loss_nce,
        "loss_pos_position": pos_loss.mean(),
        "loss_pos_rotation": rot_loss.mean(),
        "loss_pos_gripper": grip_loss.mean(),
        "nce_d_pos": d_pos_mean,
        "nce_d_neg": d_neg_mean,
        "nce_logit": logit_mean,
        "nce_prob_pos": prob_mean,
        "triplet_fraction": has_negative.float().mean(),
        "triplet_weight_mean": ctr_weight[has_negative].mean()
        if has_negative.any()
        else ctr_weight.new_tensor(0.0),
    }


def compute_embedding_probabilistic_nce_xvla_loss(
    pred_action_seq: torch.Tensor,
    positive_action_first: torch.Tensor,
    negative_action_first: torch.Tensor,
    has_negative: torch.Tensor,
    stats: TorchActionStats,
    action_embedder: ActionEmbedder,
    *,
    sample_weight: torch.Tensor | None = None,
    tau: float = 0.1,
    lambda_pos: float = 1.0,
    lambda_nce: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute BC + embedding-based probabilistic two-way NCE preference loss."""
    pred = pred_action_seq[:, 0, :10]
    l_bc, pos_loss, rot_loss, grip_loss = _bc_components(pred, positive_action_first, stats)
    loss_bc = l_bc.mean()

    pred_embed_input = _embedding_action_input(pred, stats, gripper_is_logit=True)
    pos_embed_input = _embedding_action_input(positive_action_first, stats, gripper_is_logit=False)
    neg_embed_input = _embedding_action_input(negative_action_first, stats, gripper_is_logit=False)

    z_pred, z_pred_raw = action_embedder(pred_embed_input, return_pre_norm=True)
    z_pos, z_pos_raw = action_embedder(pos_embed_input.detach(), return_pre_norm=True)
    z_neg, z_neg_raw = action_embedder(neg_embed_input.detach(), return_pre_norm=True)

    tau_tensor = pred.new_tensor(max(float(tau), 1e-8))
    sim_pos = (z_pred * z_pos).sum(dim=-1)
    sim_neg = (z_pred * z_neg).sum(dim=-1)
    logits_gap = (sim_pos - sim_neg) / tau_tensor
    l_nce_all = F.softplus(-logits_gap)
    loss_nce = _masked_weighted_mean(l_nce_all, has_negative, sample_weight)
    loss_total = lambda_pos * loss_bc + lambda_nce * loss_nce

    active = has_negative
    active_count = active.float().sum().clamp_min(1.0).to(dtype=pred.dtype)

    def active_mean(values: torch.Tensor) -> torch.Tensor:
        return torch.where(active, values, torch.zeros_like(values)).sum() / active_count

    ctr_weight = (
        torch.ones_like(l_bc)
        if sample_weight is None
        else sample_weight.to(device=l_bc.device, dtype=l_bc.dtype)
    )
    pred_raw_norm = torch.linalg.norm(z_pred_raw, dim=-1)
    pos_raw_norm = torch.linalg.norm(z_pos_raw, dim=-1)
    neg_raw_norm = torch.linalg.norm(z_neg_raw, dim=-1)

    return {
        "loss_total": loss_total,
        "loss_pos": loss_bc,
        "loss_ctr": loss_nce,
        "loss_bc": loss_bc,
        "loss_nce": loss_nce,
        "loss_pos_position": pos_loss.mean(),
        "loss_pos_rotation": rot_loss.mean(),
        "loss_pos_gripper": grip_loss.mean(),
        "emb_sim_pos": active_mean(sim_pos),
        "emb_sim_neg": active_mean(sim_neg),
        "emb_sim_gap": active_mean(sim_pos - sim_neg),
        "emb_logit_gap": active_mean(logits_gap),
        "emb_prob_pos": active_mean(torch.sigmoid(logits_gap)),
        "emb_raw_norm_pred": pred_raw_norm.mean(),
        "emb_raw_norm_pos": pos_raw_norm.mean(),
        "emb_raw_norm_neg": neg_raw_norm.mean(),
        "emb_raw_norm_std": torch.cat([pred_raw_norm, pos_raw_norm, neg_raw_norm], dim=0).std(unbiased=False),
        "triplet_fraction": has_negative.float().mean(),
        "triplet_weight_mean": ctr_weight[has_negative].mean()
        if has_negative.any()
        else ctr_weight.new_tensor(0.0),
    }


def compute_preference_kl_loss(
    student_pred_action_seq: torch.Tensor,
    teacher_pred_action_seq: torch.Tensor,
    positive_action_first: torch.Tensor,
    negative_action_first: torch.Tensor,
    has_negative: torch.Tensor,
    stats: TorchActionStats,
    *,
    sample_weight: torch.Tensor | None = None,
    tau: float = 1.0,
    lambda_gripper_kl: float = 1.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute Bernoulli KL between teacher and student pair preferences."""
    student_pred = student_pred_action_seq[:, 0, :10]
    teacher_pred = teacher_pred_action_seq[:, 0, :10].detach()
    tau_tensor = student_pred.new_tensor(max(float(tau), 1e-8))

    d_pos, d_neg = _preference_squared_distances(
        student_pred,
        positive_action_first,
        negative_action_first,
        stats,
        lambda_gripper=lambda_gripper_kl,
    )
    with torch.no_grad():
        d_pos_ref, d_neg_ref = _preference_squared_distances(
            teacher_pred,
            positive_action_first,
            negative_action_first,
            stats,
            lambda_gripper=lambda_gripper_kl,
        )
        q_pos = torch.sigmoid((d_neg_ref - d_pos_ref) / tau_tensor)

    p_pos = torch.sigmoid((d_neg - d_pos) / tau_tensor)
    p = p_pos.clamp(eps, 1.0 - eps)
    q = q_pos.clamp(eps, 1.0 - eps)
    kl_all = q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))
    loss_kl = _masked_weighted_mean(kl_all, has_negative, sample_weight)

    active = has_negative
    active_count = active.float().sum().clamp_min(1.0).to(dtype=student_pred.dtype)
    p_mean = torch.where(active, p_pos, torch.zeros_like(p_pos)).sum() / active_count
    q_mean = torch.where(active, q_pos, torch.zeros_like(q_pos)).sum() / active_count

    return {
        "loss_kl": loss_kl,
        "kl_student_prob_pos": p_mean,
        "kl_teacher_prob_pos": q_mean,
    }
