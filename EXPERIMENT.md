# X-VLA CALVIN LoRA Experiment Log

이 문서는 CALVIN base rollout으로 만든 성공/실패 데이터셋을 이용해 X-VLA LoRA를 학습한 실험 흐름을 기록한다.
목표는 base가 실패하는 subtask를 개선하되, base가 이미 잘하던 task의 성능 하락을 줄이는 것이다.

## 공통 평가 기준

- Benchmark: CALVIN ABC->D, 1000 sequences
- 주요 비교 지표:
  - full success: 5개 subtask를 모두 성공한 episode 비율
  - first-subtask success: 첫 번째 subtask 성공률
  - avg completed subtasks: episode당 평균 완료 subtask 수
- task별 분석은 주로 first-subtask 기준 성공률 차이(pp)를 사용한다.

## Base: `xvla_calvin_base_260422`

Base model 평가 결과:

| run | full success | first-subtask success | avg completed |
|---|---:|---:|---:|
| base | 73.6% | 95.2% | 4.295 |

실패 episode 264개에서 실패한 subtask를 보면 `push_*_block_right` 계열이 가장 많이 등장했다.

| failed subtask | count | ratio |
|---|---:|---:|
| push_red_block_right | 36 | 13.6% |
| push_blue_block_right | 29 | 11.0% |
| push_pink_block_right | 27 | 10.2% |

세 task 합계는 92/264 = 34.8%였다. 따라서 초기 LoRA 실험은 base가 약한 `push_*_right` 실패 패턴을 성공 rollout과 비교해 교정하는 방향으로 진행했다.

## v1: `xvla_calvin_lora_260424`

### 세팅

- Manifest: `train/outputs/xvla_calvin_base_260422`
- Triplet divergence: EE distance 기반
- Training:
  - batch size 4
  - `lambda_pos=1.0`
  - `lambda_ctr=1.0`
  - squared hinge triplet loss

### 결과

| run | full success | first-subtask success | avg completed |
|---|---:|---:|---:|
| base | 73.6% | 95.2% | 4.295 |
| v1 | 57.1% | 93.0% | 3.814 |

학습 summary:

| metric | value |
|---|---:|
| final eval loss | 0.2045 |
| final eval pos | 0.1886 |
| final eval ctr | 0.0159 |
| final grad norm | 18.64 |

v1은 일부 실패 task를 교정했지만, 전체 full success는 base보다 크게 낮았다. first-subtask 기준으로도 base보다 낮아져서, 단순 triplet 학습이 base의 기존 능력을 보존하지 못하는 문제가 있었다.

## v2: `xvla_calvin_lora_v2_260427`

### 변경한 부분

v1에서 EE distance divergence만 사용하던 것을 개선하기 위해 gripper-aware divergence를 추가했다.

- Manifest: `train/outputs/xvla_calvin_base_v2_260427`
- 변경:
  - `gripper_persistence_steps=3`
  - divergence reason 기록: `distance`, `gripper`, `both`
  - gripper mismatch가 일정 step 이상 지속되면 divergence 후보로 사용

Manifest 규모:

| item | v1 | v2 |
|---|---:|---:|
| success_only samples | 2941 | 2941 |
| triplet samples | 1194 | 1221 |
| matched failed segments | 240 | 246 |
| unmatched failed segments | 24 | 18 |

v2 triplet divergence reason:

| reason | count |
|---|---:|
| distance | 149 |
| gripper | 49 |
| both | 48 |

### 결과

| run | full success | first-subtask success | avg completed |
|---|---:|---:|---:|
| base | 73.6% | 95.2% | 4.295 |
| v1 | 57.1% | 93.0% | 3.814 |
| v2 | 56.4% | 94.1% | 3.819 |

학습 summary:

| metric | value |
|---|---:|
| best eval loss | 0.1374 @ step 8100 |
| final eval loss | 0.1752 |
| final eval pos | 0.1574 |
| final eval ctr | 0.0177 |
| final grad norm | 226.91 |

### 해석

v2는 first-subtask 기준으로 v1보다 좋아졌고, 특히 base가 약했던 `push_*_right` 계열을 크게 개선했다.

| first subtask | base | v1 | v2 | v2-base | v2-v1 |
|---|---:|---:|---:|---:|---:|
| push_red_block_right | 41.4% | 55.2% | 86.2% | +44.8 pp | +31.0 pp |
| push_pink_block_right | 65.5% | 75.9% | 96.6% | +31.0 pp | +20.7 pp |
| push_blue_block_right | 52.2% | 56.5% | 65.2% | +13.0 pp | +8.7 pp |

이는 gripper-aware event가 추가되면서 `push_*_right`의 실패 원인을 더 잘 잡은 효과로 보인다.

반면 v2에서 크게 하락한 task도 있었다.

| first subtask | base | v2 | diff |
|---|---:|---:|---:|
| lift_blue_block_slider | 100.0% | 56.7% | -43.3 pp |
| push_into_drawer | 100.0% | 66.7% | -33.3 pp |
| rotate_blue_block_right | 100.0% | 67.9% | -32.1 pp |
| push_blue_block_left | 97.1% | 77.1% | -20.0 pp |

원인 가설:

- `push_*_right` triplet이 전체 triplet의 큰 비중을 차지했다.
- gripper/both divergence가 일부 task에서는 유효했지만, rotate/lift/drawer류에서는 noisy hard negative처럼 작동했을 수 있다.
- final checkpoint는 grad norm이 매우 커서 학습 후반 안정성이 낮았다.

## v3: `xvla_calvin_lora_v3_260429`

### 변경한 부분

v2의 문제를 줄이기 위해 task family별 수동 weighting 대신, 데이터 기반 triplet weighting을 도입했다.

목표:

- sample이 많은 subtask가 전체 contrastive gradient를 지배하지 않게 한다.
- matched event가 적은 subtask는 과하게 upweight하지 않는다.
- task 이름에 의존하지 않는 일반화 가능한 조절 방식을 사용한다.

추가한 방식:

```text
w_freq = sqrt(mean_triplet_count / subtask_triplet_count)
w_freq = clamp(w_freq, min=0.5, max=2.0)

w_conf = min(1.0, sqrt(num_matched_events / 10))

final_triplet_weight = w_freq * w_conf
```

Training:

- Manifest: `train/outputs/xvla_calvin_base_v2_260427`
- batch size 4
- `lambda_pos=1.0`
- `lambda_ctr=1.0`
- squared hinge triplet loss 유지
- `--use_triplet_balance_weights`
- `triplet_weight_min=0.5`
- `triplet_weight_max=2.0`
- `triplet_confidence_events=10`

주요 weight:

| subtask | samples | events | weight |
|---|---:|---:|---:|
| push_red_block_right | 175 | 35 | 0.518 |
| push_blue_block_right | 135 | 28 | 0.590 |
| push_pink_block_right | 135 | 27 | 0.590 |
| lift_blue_block_slider | 73 | 15 | 0.802 |
| push_into_drawer | 45 | 9 | 0.969 |
| rotate_blue_block_right | 15 | 3 | 0.969 |

### 결과

| run | full success | first-subtask success | avg completed |
|---|---:|---:|---:|
| base | 73.6% | 95.2% | 4.295 |
| v2 | 56.4% | 94.1% | 3.819 |
| v3 | 59.1% | 92.8% | 3.853 |

학습 summary:

| metric | value |
|---|---:|
| best eval loss | 0.1199 @ step 2000 |
| final eval loss | 0.1884 |
| final eval pos | 0.1771 |
| final eval ctr | 0.0114 |
| final grad norm | 10.16 |

v3는 v2보다 학습 안정성이 크게 개선되었다.

| run | final grad norm |
|---|---:|
| v2 | 226.91 |
| v3 | 10.16 |

v2에서 하락했던 일부 task는 회복했다.

| first subtask | v2 | v3 | diff |
|---|---:|---:|---:|
| rotate_blue_block_right | 67.9% | 92.9% | +25.0 pp |
| lift_blue_block_slider | 56.7% | 76.7% | +20.0 pp |
| push_into_drawer | 66.7% | 80.0% | +13.3 pp |
| push_blue_block_left | 77.1% | 85.7% | +8.6 pp |

하지만 v2에서 크게 좋아졌던 `push_*_right` 계열은 다시 하락했다.

| first subtask | v2 | v3 | diff |
|---|---:|---:|---:|
| push_red_block_right | 86.2% | 55.2% | -31.0 pp |
| push_pink_block_right | 96.6% | 65.5% | -31.0 pp |
| push_blue_block_right | 65.2% | 34.8% | -30.4 pp |

### 해석

frequency/confidence weighting은 의도대로 많이 나온 triplet task의 영향력을 낮췄다. 그 결과 v2에서 손상됐던 일부 task는 회복했지만, base 실패의 핵심이었던 `push_*_right` correction이 너무 약해졌다.

즉 v3는 다음을 보여준다.

- 장점: 학습 안정성 개선, v2 regression 일부 회복
- 단점: frequent failure task를 너무 많이 downweight해서 핵심 개선 task가 약화됨

## v4: `xvla_calvin_lora_v4_260430`

### 변경한 부분

v3의 방향은 유지하되, frequent task downweight를 완화하기 위해 `triplet_weight_min`을 올린다.

변경:

| setting | v3 | v4 |
|---|---:|---:|
| triplet_weight_min | 0.5 | 0.75 |
| triplet_weight_max | 2.0 | 2.0 |
| triplet_confidence_events | 10 | 10 |
| batch size | 4 | 4 |
| lambda_ctr | 1.0 | 1.0 |
| triplet loss | squared hinge | squared hinge |

의도:

- `push_*_right`처럼 많이 나온 task를 완전히 누르지 않는다.
- v2에서 얻은 push-right correction을 일부 회복한다.
- v3에서 얻은 안정성과 regression 회복 효과는 최대한 유지한다.

예상 weight 변화:

| subtask | v3 weight | v4 expected lower bound |
|---|---:|---:|
| push_red_block_right | 0.518 | 0.75 |
| push_blue_block_right | 0.590 | 0.75 |
| push_pink_block_right | 0.590 | 0.75 |

v4 학습 명령의 핵심 차이:

```bash
--use_triplet_balance_weights \
--triplet_weight_min 0.75 \
--triplet_weight_max 2.0 \
--triplet_confidence_events 10
```

## 다음 TODO

- v4 학습 후 checkpoint별 eval loss와 grad norm을 확인한다.
- v4 benchmark에서 다음 두 가지가 동시에 개선되는지 확인한다:
  - `push_*_right`가 v3보다 회복되는지
  - v2에서 하락했던 rotate/lift/drawer류가 다시 크게 무너지지 않는지
- v4도 불안정하거나 성능 trade-off가 남으면 다음 실험을 진행한다:
  - v5: 원본 X-VLA CALVIN base에서 시작하고 `--lora_only`를 사용해 LoRA adapter만 학습
  - `soft_prompt_hub`, `action_encoder`, `action_decoder` full fine-tuning 제외
  - 목적: base의 action/gripper calibration을 보존하면서 contrastive signal만 약하게 반영
  - squared hinge loss -> hinge loss
  - `lambda_ctr=0.5`
  - `lambda_ctr` warmup: step 1000~3000 사이에 0.0에서 target으로 증가
  - hard/noisy negative clipping
