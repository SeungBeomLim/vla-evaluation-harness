# GR00T / LIBERO Training Data

This directory contains the current GR00T/LIBERO data construction, IDM,
and preference fine-tuning utilities.

## Current Data Layout

Collected GR00T LIBERO rollout roots:

```text
/data/vla-evaluation/benchmark/groot_libero_base_seed7_260506
/data/vla-evaluation/benchmark/groot_libero_base_seed13_260506
/data/vla-evaluation/benchmark/groot_libero_base_seed17_260506
/data/vla-evaluation/benchmark/groot_libero_base_seed23_260506
/data/vla-evaluation/benchmark/groot_libero_base_seed29_260506
/data/vla-evaluation/benchmark/groot_libero_base_seed31_260506
```

Current GR00T/LIBERO training manifests:

```text
train/outputs/groot_libero_chunk_preference_pairs_260508/
  preference_pairs.jsonl
  success_only.jsonl
  summary.json
```

## Build Chunk Preference Pairs

Use this builder for the current training plan. It separates the VLA input
chunk-start observation from the chunk-internal target step:

```text
policy_obs_step = chunk start c
target_step = c + action_offset
```

The VLA input should use image/state/language at `policy_obs_step`, while NCE
targets use success/failure actions at `target_step`. IDM caching uses the
negative image observation and positive goal image observation at `target_step`.

```bash
python3 train/build_libero_preference_pairs.py \
  --rollout-root /data/vla-evaluation/benchmark/groot_libero_base_seed7_260506 \
  --rollout-root /data/vla-evaluation/benchmark/groot_libero_base_seed13_260506 \
  --rollout-root /data/vla-evaluation/benchmark/groot_libero_base_seed17_260506 \
  --rollout-root /data/vla-evaluation/benchmark/groot_libero_base_seed23_260506 \
  --rollout-root /data/vla-evaluation/benchmark/groot_libero_base_seed29_260506 \
  --rollout-root /data/vla-evaluation/benchmark/groot_libero_base_seed31_260506 \
  --output-dir train/outputs/groot_libero_chunk_preference_pairs_260508
```

Default selection:

```text
policy score =
  0.5 * current_state_similarity
+ 0.4 * short_future_divergence
+ 0.1 * weak_action_saliency

target score =
  0.35 * state_growth_from_policy_step
+ 0.25 * local_step_growth
+ 0.20 * short_future_growth
+ 0.15 * action_saliency
+ 0.05 * earliness
```

The default `--target-window-size 3` ensures the selected offset leaves a
3-step target window inside the same action chunk, which matches the current
goal-image IDM horizon.

## Train Goal-Image IDM

The IDM code is organized as a benchmark/model-generic entrypoint with a
LIBERO/GR00T preset. The current model maps two image observations to the
action chunk that moves from the first observation toward the goal observation:

```text
(obs_t images, obs_{t+K} images) -> action[t:t+K]
```

```bash
python3 train/idm/train.py \
  --config experiment_specs/idm/libero_groot_goal_image_train.yaml
```

For another benchmark/model, add a preset in `train/idm/core.py` and create another
YAML/JSON file under `experiment_specs/idm/`. The spec owns these selectable options:

```text
paths.rollout_roots
paths.output_dir
data.preset
data.trajectory_glob
data.image_keys
data.state_key
data.action_key
model.*
training.*
runtime.*
wandb.*
```

## Evaluate IDM

After training, run offline action prediction metrics and optional LIBERO
one-step replay metrics:

```bash
python3 train/idm/evaluate.py \
  --config experiment_specs/idm/libero_groot_goal_image_eval.yaml
```

The replay path restores step `t` by replaying recorded actions from the
episode start, then executes the predicted IDM action once and compares the
next state/image to the recorded `t+1` target.

## Inspect Preference Pairs

Before training, create contact sheets to verify that `policy_obs_step` is a
reasonable chunk-start input and `target_step` captures a meaningful
success/failure difference.

For a numeric manifest summary, run:

```bash
python3 train/analyze_libero_preference_pairs.py \
  --preference-pairs train/outputs/groot_libero_chunk_preference_pairs_260508/preference_pairs.jsonl \
  --output train/outputs/groot_libero_chunk_preference_pairs_260508/analysis_summary.json \
  --compute-action-gaps
```

```bash
python3 train/inspect_libero_preference_pairs.py \
  --pairs train/idm/outputs/groot_libero_chunk_preference_pairs_goal_image_idm_h3_260526/preference_pairs_with_idm.jsonl \
  --idm-actions train/idm/outputs/groot_libero_chunk_preference_pairs_goal_image_idm_h3_260526/idm_actions.npz \
  --output-dir train/idm/outputs/groot_libero_chunk_preference_pairs_goal_image_idm_h3_260526/inspection_random_50 \
  --num-samples 50 \
  --selection random \
  --seed 0 \
  --view agentview \
  --view wrist \
  --context 2
```

Useful selection modes:

```text
random
offset13
high-score
```

## Train GR00T Preference LoRA

Use YAML specs for fine-tuning runs:

```bash
python3 train/groot_preference_finetune.py \
  --config experiment_specs/preference/groot_libero_bc_only.yaml

python3 train/groot_preference_finetune.py \
  --config experiment_specs/preference/groot_libero_onset_only.yaml
```

Run BC-only first. If BC-only damages benchmark performance, debug the
fine-tuning pipeline before adding onset NCE or IDM NCE.

## Removed Legacy Code

Older CALVIN/X-VLA LoRA scripts, state-only IDM scripts, and rank-triplet
builders were removed from the active tree. The current path is the
GR00T/LIBERO chunk preference manifest plus goal-image IDM.
