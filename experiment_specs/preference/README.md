# GR00T / LIBERO Preference Fine-tuning Specs

Run fine-tuning from YAML instead of long CLI argument lists.

```bash
python3 train/groot_preference_finetune.py \
  --config experiment_specs/preference/groot_libero_bc_only.yaml

python3 train/groot_preference_finetune.py \
  --config experiment_specs/preference/groot_libero_onset_only.yaml
```

Recommended diagnostic order:

1. `groot_libero_bc_only.yaml`
2. benchmark eval of the BC-only checkpoint
3. `groot_libero_onset_only.yaml`
4. benchmark eval of onset-only checkpoints
5. add IDM loss only if the first two stages are healthy

`idm_actions` can be `null` when `lambda_idm: 0.0`.
