# IDM Experiment Specs

Use these experiment specs instead of passing long option lists on the command line.

```bash
python3 train/idm/train.py --config experiment_specs/idm/libero_groot_goal_image_train.yaml
python3 train/idm/evaluate.py --config experiment_specs/idm/libero_groot_goal_image_eval.yaml
```

## Train Config Sections

- `paths`: rollout roots and output directory.
- `data`: benchmark/model preset and trajectory key names.
- `model`: goal-image IDM architecture and horizon.
- `training`: optimizer, epochs, split ratio, and seed.
- `runtime`: device, dataloader workers, npz cache, dotenv path.
- `wandb`: logging project/run/mode.

For a new benchmark/model, add an entry to `IDM_DATA_PRESETS` in
`train/idm/core.py`, then create a new train/eval config pair here.
