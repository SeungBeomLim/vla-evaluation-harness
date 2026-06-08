# LIBERO / GR00T Preference Fine-tuning TODO

## 현재 상태

- 목표: GR00T-N1.6 LIBERO base policy가 실패하는 분기점에서 실패 action보다 성공 action 쪽으로 출력을 유도한다.
- 데이터:
  - base rollout seeds: 7, 13, 17, 23, 29, 31
  - common episode key: 2000
  - unstable success/failure key: 409
  - preference pairs: 2372
  - success-only BC samples: 3558
- 현재 manifest:
  - `train/outputs/groot_libero_chunk_preference_pairs_260508/preference_pairs.jsonl`
  - `train/outputs/groot_libero_chunk_preference_pairs_260508/success_only.jsonl`
- 이전 state-only IDM manifest는 제거 대상이다.
- IDM은 현재 action cache를 만들지 않고 train/evaluate/replay debug로 품질을 먼저 확인한다.
- 현재 학습 run:
  - `train/runs/groot_libero_pref_lora_260513`
  - LoRA-only
  - `lambda_bc=1.0`
  - `lambda_onset=0.1`
  - `lambda_idm=0.05`
  - `num_inference_timesteps=4`
- 결과:

```text
base all seeds: 11450 / 12000 = 95.4%
LoRA ckpt-10000: 753 / 2000 = 37.7%

suite별:
libero_10      89.9% ->  6.0%
libero_goal    96.2% -> 59.2%
libero_object  99.2% -> 28.0%
libero_spatial 96.4% -> 57.4%
```

현재 LoRA는 base를 보존하지 못했고, 특히 `libero_10`이 크게 무너졌다.

## 현재 의심 지점

### 1. 기존 state-only IDM action target이 action bound 밖으로 나감

- `idm_actions.npz`의 raw env action range가 `[-1, 1]`을 넘는다.
- 확인된 예:

```text
idm pos/rot/gripper min: [-1.199, -1.402, -1.363, ..., -1.333]
idm pos/rot/gripper max: [ 1.238,  1.397,  1.331, ...,  1.466]
```

- GR00T/LIBERO server action은 gripper 변환을 거치므로, out-of-bound IDM action이 native action normalization으로 들어가면 policy를 불가능한 action 방향으로 당길 수 있다.
- 기존 state-only IDM은 삭제하고 더 이상 사용하지 않는다.
- 새 IDM은 `train/idm/` 패키지로 일반화한다.
- 새 IDM은 두 image observation을 입력으로 받아 A에서 B로 가는 action chunk를 예측한다.

우선 조치:

- goal-image IDM output을 `tanh * action_limit`로 bound한다.
- gripper는 native 변환 후 `[0, 1]` 범위도 확인한다.
- clamp 전/후 action range를 train/eval log에 남긴다.

### 2. NCE가 원하는 방향으로 학습되지 않음

최종 eval metric:

```text
onset_d_pos = 0.786
onset_d_neg = 0.735
onset_prob_pos = 0.492

idm_d_pos = 1.084
idm_d_neg = 0.735
idm_prob_pos = 0.423
```

- positive/IDM target이 negative failure action보다 더 가까워져야 하는데, 실제로는 반대다.
- 특히 IDM NCE는 failure negative보다 IDM positive가 훨씬 멀다.
- 현재 loss는 benchmark 성능과도 잘 연결되지 않는다.

우선 조치:

- `lambda_idm=0` ablation을 먼저 돌린다.
- onset-only에서도 `d_pos < d_neg`가 안 되면 preference pair 또는 sampled-action NCE 자체를 의심한다.
- train/eval에서 `d_pos - d_neg` 평균과 suite/task별 breakdown을 기록한다.

### 3. Preference pair 데이터가 뒤쪽 offset과 일부 failure input에 편향됨

현재 분포:

```text
action_offset=13: 987 / 2372
policy_obs_step=0: 1133 / 2372
same vla_input duplicated groups: 591
duplicated rows in those groups: 1653
positive/negative action gap <= 0.1: 910 / 2372 = 38.4%
```

- 같은 failure observation에 여러 success target이 붙는 경우가 많다.
- `target_step`이 chunk 뒤쪽으로 몰려서 실제 correction onset보다 늦은 action을 학습할 수 있다.
- `libero_10` pair가 1211/2372로 많고, long-horizon failure pattern이 gradient를 지배할 수 있다.

우선 조치:

- 같은 `vla_input` 중복을 dedup하거나 cap을 둔다.
- `action_offset <= 8` 또는 `<= 10` ablation을 만든다.
- `target_action_gap <= 0.1` 같은 약한 pair는 제거하거나 downweight한다.
- suite/task 균형 sampling을 넣는다.
- `train/analyze_libero_preference_pairs.py`로 새 manifest마다 offset/dup/action gap summary를 저장한다.

### 4. Success-only BC 보존이 너무 약함

- base가 이미 95% 성공하는 policy인데, success-only BC는 3558 samples뿐이다.
- 매 training step에서 preference loss가 같이 들어가므로, base behavior 보존력이 부족할 수 있다.
- 현재 `BC + NCE`가 base policy를 크게 흔든 것으로 보인다.

우선 조치:

- BC-only fine-tuning부터 반드시 확인한다.
- success-only sample 수를 늘린다.
  - 성공 rollout당 3 chunk 제한을 완화한다.
  - chunk start 전체 또는 균등 샘플링을 사용한다.
- BC batch 또는 `lambda_bc`를 키우는 ablation을 둔다.

### 5. Sampled-action NCE와 GR00T 원래 objective mismatch

- 현재 fine-tuning은 differentiable sampling loop로 sampled action을 만든 뒤 distance loss/NCE를 건다.
- GR00T의 원래 학습 objective는 flow-matching velocity loss다.
- `num_inference_timesteps=4` sampled action은 실제 benchmark inference distribution과 다를 수 있다.
- loss가 내려가도 rollout success가 나빠질 수 있다.

우선 조치:

- BC-only에서 이미 망가지면 sampled-action BC 자체를 의심한다.
- sampled-action 방식이 불안정하면 flow-matching objective로 전환한다.

## 최소 ablation 순서

아래 순서로 해야 원인을 분리할 수 있다.

```text
A. base GR00T benchmark 재확인
B. BC-only LoRA
C. BC + onset NCE
D. BC + onset NCE + IDM NCE(clamped)
E. BC + onset NCE + IDM NCE(off 또는 very small lambda)
```

해석:

- B에서 떨어지면:
  - action normalization, gripper convention, LoRA target, sampled-action BC, training pipeline 문제.
- B는 괜찮고 C에서 떨어지면:
  - preference pair 품질 또는 onset NCE 문제.
- C는 괜찮고 D에서 떨어지면:
  - IDM target 품질 또는 IDM action convention 문제.
- train/eval loss는 좋아지는데 benchmark가 떨어지면:
  - sampled-action NCE와 actual rollout distribution mismatch 문제.

## 바로 해야 할 작업

- [ ] `groot_preference_finetune.py`에 target action range logging 추가.
  - raw env action range
  - native action range
  - normalized action range
  - pred action range
- [x] 기존 state-only IDM 학습/캐시/검증 코드를 제거.
- [x] generic goal-image IDM 구조 생성.
  - `train/idm/core.py`
  - `train/idm/train.py`
  - 기본 preset: `libero:groot`
- [x] IDM 학습/평가 설정을 config 파일로 분리.
  - `experiment_specs/idm/libero_groot_goal_image_train.yaml`
  - `experiment_specs/idm/libero_groot_goal_image_eval.yaml`
- [x] goal-image IDM 학습/검증 실행.
  - 입력: current image observation, goal image observation
  - 출력: A에서 B로 가는 K-step action chunk
  - h1/h5 debug 결과를 replay와 offline metric으로 확인
- [x] GR00T preference fine-tuning을 YAML config 기반으로 변경.
  - `experiment_specs/preference/groot_libero_bc_only.yaml`
  - `experiment_specs/preference/groot_libero_onset_only.yaml`
- [x] `lambda_idm=0`일 때 IDM action 없이 학습 가능하도록 optional화.
- [x] preference pair manifest diagnostic 추가.
  - `train/analyze_libero_preference_pairs.py`
- [ ] BC-only run 실행.
  - `lambda_onset=0`
  - `lambda_idm=0`
  - checkpoint별 benchmark 확인
- [ ] BC-only checkpoint benchmark eval.
- [ ] onset-only run 실행.
  - `lambda_idm=0`
  - `onset_d_pos < onset_d_neg`가 되는지 확인
- [ ] onset-only checkpoint benchmark eval.
- [ ] IDM을 LoRA loss에 다시 붙이는 실험은 replay 검증 이후에만 실행.
  - 시작값: `lambda_idm=0.01` 또는 `0.02`
  - `lambda_idm=0.05`는 현재 기준으로 강할 수 있음
- [ ] preference pair filtered dataset 생성.
  - `action_offset <= 8`
  - same `vla_input` dedup 또는 max N cap
  - weak action gap pair 제거
  - suite/task balance 적용
- [ ] success-only dataset 증강.
  - 성공 rollout당 더 많은 chunk start를 포함
  - suite/task balance 유지
- [ ] suite/task별 eval metric 저장.
  - BC loss
  - onset NCE
  - IDM NCE
  - `d_pos - d_neg`
  - action offset distribution
- [ ] LoRA eval 결과를 base와 task별로 비교하는 분석 스크립트 작성.

## 다음 구현 후보

### 1. Flow-matching BC/NCE

현재 sampled-action loss가 계속 불안정하면 가장 우선순위가 높다.

아이디어:

```text
L_bc =
  FM_loss(obs_success, A_success)

L_onset =
  softplus((FM_loss(obs_failure, A_success)
          - FM_loss(obs_failure, A_failure)) / tau)

L_idm =
  softplus((FM_loss(obs_failure, A_idm)
          - FM_loss(obs_failure, A_failure)) / tau_idm)
```

주의:

- positive/negative 비교 시 같은 noise epsilon과 같은 flow time t를 사용한다.
- 전체 chunk가 아니라 `target_step : target_step + L` window mask를 적용한다.
- IDM target은 clamp 후 사용한다.

### 2. Target selection 개선

- onset 방식:
  - gap이 큰 지점이 아니라 gap이 막 증가하기 시작하는 지점을 고른다.
- discount 방식:
  - 뒤쪽 offset에 penalty를 준다.
- peak-change 방식:
  - `state_gap(t) - state_gap(t-1)`가 큰 지점을 고른다.
- multi-window 방식:
  - 한 failure pair에서 하나의 top-1 target만 뽑지 않고, 여러 후보를 만들고 score로 sample weight를 준다.

### 3. Goal-image IDM

- state-only IDM은 object-relative visual cue를 못 봤고 action range도 불안정했다.
- 다음 IDM은 target step의 두 image observation을 입력으로 받는다.
- 현재는 LIBERO만 쓰지만 entrypoint와 preset은 다른 benchmark/model을 붙일 수 있게 유지한다.

```text
input:
  failure image at target_step
  success image at target_step
  task instruction optional

output:
  K-step recovery action prefix
```

현재 구현:

```text
train/idm/core.py
  IDM_DATA_PRESETS["libero:groot"]
  TrajectoryGoalImageIDMDataset
  GoalImageIDM

train/idm/train.py
  --config experiment_specs/idm/libero_groot_goal_image_train.yaml

train/idm/evaluate.py
  --config experiment_specs/idm/libero_groot_goal_image_eval.yaml
  offline action prediction + optional LIBERO replay validation
```

## 보류 / 제거한 이전 기록

- 이전 rank-triplet manifest와 builder는 제거했다.
- CALVIN / X-VLA LoRA 실험 로그는 이 TODO에서 다루지 않는다.
- smoke run 이전 메모리는 현재 진단 이후 우선순위가 낮아 제거했다.
- 현재 문서는 GR00T/LIBERO preference fine-tuning의 성능 하락 원인 분리와 다음 실험만 추적한다.
