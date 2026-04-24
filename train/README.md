# Train Data Builder

This directory contains standalone utilities for converting rollout artifacts
under `results/.../rollouts/*/trajectory.npz` into lightweight training
manifests.

The builder does not duplicate rollout images or actions. Instead it writes
JSONL manifests that reference:

- rollout directory
- step index
- sample type (`success_only` or `triplet`)

Current policy:

- `success_only`
  - use only successful episodes
  - cap each subtask to `min(count, 30)` segments
  - sample progress steps at `0.3`, `0.6`, `0.85`
- `triplet`
  - use only the last failed subtask of failed episodes
  - match against successful segments of the same subtask
  - detect a divergence point after the trajectories become spatially close
  - sample a small window around that divergence point

Example:

```bash
python3 train/build_dataset.py \
  --results-dir results/xvla_calvin_base_260422 \
  --output-dir train/outputs/xvla_calvin_base_260422
```

LoRA fine-tuning from the generated manifests:

```bash
python3 train/xvla_peft_finetune.py \
  --models /path/to/X-VLA-Calvin-ABC_D \
  --manifest_dir train/outputs/xvla_calvin_base_260422 \
  --output_dir train/runs/xvla_calvin_lora
```

This training path:

- reads `success_only.jsonl` and `triplets.jsonl`
- reconstructs X-VLA-style 20-D proprio from CALVIN 8-D state
- reconstructs positive action chunks for X-VLA's sequence-conditioned transformer
- applies custom loss on `pred[:, 0, :10]`
- uses normalized position/rotation losses, gripper BCE, and squared-hinge triplet loss
