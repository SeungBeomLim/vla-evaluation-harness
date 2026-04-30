"""Custom normalized loss for X-VLA rollout fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
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


def _normalize_components(action_10d: torch.Tensor, stats: TorchActionStats) -> tuple[torch.Tensor, torch.Tensor]:
    pos = (action_10d[..., 0:3] - stats.pos_mean) / stats.pos_std
    rot = (action_10d[..., 3:9] - stats.rot_mean) / stats.rot_std
    return pos, rot


def compute_normalized_xvla_loss(
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
    """Compute normalized positive + squared-hinge contrastive loss."""
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
