# GR00T / LIBERO IDM + Preference Fine-tuning Log

이 문서는 현재 사용하는 GR00T-N1.6 / LIBERO 실험 흐름만 기록한다.
이전 CALVIN / X-VLA LoRA 실험 기록과 코드는 현재 경로에서 제거했다.

## 목표

- GR00T-N1.6 base policy가 실패하는 LIBERO 분기점에서 실패 action보다 성공 action 또는 recovery action 쪽으로 출력을 유도한다.
- IDM은 state 기반이 아니라 두 image observation을 입력으로 받아 A에서 B로 이동하는 action chunk를 예측한다.
- 학습 코드는 나중에 다른 benchmark/model을 붙일 수 있도록 `train/idm/`와 config 기반 entrypoint로 유지한다.

## 현재 코드 축

```text
train/idm/
  core.py
  train.py
  evaluate.py
  analyze_gripper_outliers.py

train/build_libero_preference_pairs.py
train/libero_preference_pair_builder.py
train/inspect_libero_preference_pairs.py
train/groot_preference_finetune.py

experiment_specs/idm/
  libero_groot_goal_image_train*.yaml
  libero_groot_goal_image_eval*.yaml
```

## IDM 실험

- 입력: current image observation, goal image observation
- 출력: horizon 길이의 GR00T/LIBERO env action chunk
- 현재 주요 horizon:
  - h1: 1-step debug
  - h5: 두 observation 사이 recovery prefix 검증용

검증은 두 축으로 본다.

- Offline action prediction: validation loss, action MSE/L1, position/rotation/gripper breakdown
- Replay evaluation: 같은 episode initial state에서 recorded action prefix를 replay해 A 시점까지 복원한 뒤, IDM predicted action을 실행하고 B에 가까워지는지 확인

full MuJoCo state가 trajectory에 저장되어 있지 않으므로 replay는 완벽한 state set 검증은 아니다. 하지만 같은 episode의 initial state와 recorded prefix를 사용하므로 현재 artifact에서 가능한 가장 현실적인 동역학 검증이다.

## Preference / LoRA 실험

현재 preference 데이터는 GR00T LIBERO rollout에서 성공/실패 분기점을 찾아 만든다.

```text
train/outputs/groot_libero_chunk_preference_pairs_260508/
  preference_pairs.jsonl
  success_only.jsonl
  summary.json
```

현재 의심 지점:

- BC-only에서도 base 성능이 깨지는지 확인해야 한다.
- onset NCE가 `d_pos < d_neg` 방향으로 실제로 작동하는지 확인해야 한다.
- IDM target은 replay 검증을 통과한 horizon/action convention만 사용해야 한다.
- preference pair는 duplicated `vla_input`, offset 편향, suite imbalance를 줄여야 한다.

## 제거한 코드

- state-only IDM 학습/캐시/검증 코드
- CALVIN / X-VLA legacy LoRA 학습 코드
- 오래된 rank-triplet dataset builder
- 아직 쓰지 않는 IDM action cache entrypoint/config
