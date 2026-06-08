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

### 결과

| run | full success | first-subtask success | avg completed |
|---|---:|---:|---:|
| base | 73.6% | 95.2% | 4.295 |
| v3 | 59.1% | 92.8% | 3.853 |
| v4 | 56.8% | 92.9% | 3.800 |

학습 summary:

| metric | value |
|---|---:|
| best eval loss | 0.1221 @ step 2000 |
| final eval loss | 0.1907 |
| final eval pos | 0.1805 |
| final eval ctr | 0.0102 |
| final grad norm | 9.45 |

v4는 `triplet_weight_min=0.75`로 frequent task downweight를 완화했지만, benchmark는 v3보다 개선되지 않았다. 특히 full success와 avg completed가 모두 낮아져서, reweight 하한을 올리는 것만으로는 base 보존과 push-right correction을 동시에 얻기 어렵다는 결과가 나왔다.

## v5: `xvla_calvin_lora_v5_260430`

### 변경한 부분

v4까지는 LoRA adapter 외에 `soft_prompt_hub`, `action_encoder`, `action_decoder`도 `modules_to_save`에 포함되어 full fine-tuning되었다. v5는 이 세 모듈을 freeze하고 LoRA adapter만 학습하도록 `--lora_only`를 추가했다.

변경:

| setting | v4 | v5 |
|---|---:|---:|
| LoRA-only | false | true |
| modules_to_save | soft/action modules | none |
| triplet reweight | enabled | enabled |
| triplet_weight_min | 0.75 | 0.75 |
| batch size | 4 | 4 |
| freeze_steps | 1000 | 1000 |
| lambda_ctr | 1.0 | 1.0 |

주의할 점:

- v5는 v2와 동일 조건이 아니다.
- v5에는 v4와 동일한 triplet reweighting이 들어가 있다.
- `freeze_steps=1000`도 유지되었다. LoRA-only에서는 LoRA adapter가 `transformer_core` group에 들어가므로 초반 1000 step 동안 실질적으로 학습이 거의 멈췄을 가능성이 있다.

### 결과

| run | full success | first-subtask success | avg completed |
|---|---:|---:|---:|
| base | 73.6% | 95.2% | 4.295 |
| v4 | 56.8% | 92.9% | 3.800 |
| v5 | 64.3% | 94.7% | 4.057 |

step별 성공률:

| step | base | v5 | diff |
|---|---:|---:|---:|
| 1/5 | 95.2% | 94.7% | -0.5 pp |
| 2/5 | 91.0% | 89.1% | -1.9 pp |
| 3/5 | 87.6% | 82.6% | -5.0 pp |
| 4/5 | 82.1% | 75.0% | -7.1 pp |
| 5/5 | 73.6% | 64.3% | -9.3 pp |

학습 summary:

| metric | value |
|---|---:|
| best eval loss | 0.1232 @ step 2400 |
| final eval loss | 0.1857 |
| final eval pos | 0.1748 |
| final eval ctr | 0.0109 |
| final grad norm | 0.084 |

v5는 v4보다 base 보존이 훨씬 좋아졌다. `rotate_blue_block_right`, `lift_blue_block_slider`, `push_into_drawer`처럼 v2/v4에서 무너졌던 task들이 base 수준으로 회복되었다. 이는 `action_encoder/action_decoder`를 직접 업데이트하지 않은 효과로 보인다.

반면 v2에서 크게 좋아졌던 `push_*_right` correction은 대부분 사라졌다.

| first subtask | base | v2 | v5 |
|---|---:|---:|---:|
| push_red_block_right | 41.4% | 86.2% | 44.8% |
| push_pink_block_right | 65.5% | 96.6% | 65.5% |
| push_blue_block_right | 52.2% | 65.2% | 47.8% |

### gripper loss 해석

v5 final eval 기준 loss 구성:

| component | loss | total 내 비중 |
|---|---:|---:|
| position | 0.0055 | 3.0% |
| rotation | 0.0144 | 7.7% |
| gripper | 0.1548 | 83.4% |
| contrastive | 0.0109 | 5.9% |

eval loss는 대부분 positive gripper BCE가 지배한다. train 마지막 step에서는 gripper loss가 매우 작지만, train 그래프 중간중간 gripper spike가 total loss spike를 만들고 eval에서는 gripper generalization error가 가장 큰 비중을 차지한다. 따라서 gripper calibration을 보존하면서 contrastive correction을 주는 방향이 중요하다.

## v5b: `xvla_calvin_lora_v5b_260501`

목적:

- v5와 달리 triplet reweighting을 제거한다.
- v2의 데이터/contrastive 조건에 가깝게 맞춘 상태에서 LoRA-only만 적용한다.
- v2에서 강했던 `push_*_right` correction이 LoRA-only에서도 살아나는지 확인한다.

설정:

| setting | value |
|---|---:|
| output_dir | `train/runs/xvla_calvin_lora_v5b_260501` |
| run_name | `xvla-calvin-lora-v5b-260501` |
| manifest | `train/outputs/xvla_calvin_base_v2_260427` |
| batch size | 4 |
| iters | 10000 |
| freeze_steps | 0 |
| triplet reweight | disabled |
| lambda_pos | 1.0 |
| lambda_ctr | 1.0 |
| LoRA-only | true |

v2와의 차이:

| setting | v2 | v5b |
|---|---:|---:|
| batch size | 1 | 4 |
| freeze_steps | 1000 | 0 |
| triplet reweight | disabled | disabled |
| LoRA-only | false | true |
| soft/action full-tune | true | false |

### 결과

| run | full success | first-subtask success | avg completed |
|---|---:|---:|---:|
| base | 73.6% | 95.2% | 4.295 |
| v2 | 56.4% | 94.1% | 3.819 |
| v5 | 64.3% | 94.7% | 4.057 |
| v5b | 65.2% | 94.1% | 4.030 |
| v6b | 66.2% | 95.3% | 4.094 |

step별 성공률:

| step | base | v5 | v5b | v6b |
|---|---:|---:|---:|---:|
| 1/5 | 95.2% | 94.7% | 94.1% | 95.3% |
| 2/5 | 91.0% | 89.1% | 88.2% | 89.7% |
| 3/5 | 87.6% | 82.6% | 81.6% | 82.6% |
| 4/5 | 82.1% | 75.0% | 73.9% | 75.6% |
| 5/5 | 73.6% | 64.3% | 65.2% | 66.2% |

학습 summary:

| metric | value |
|---|---:|
| best eval loss | 0.1296 @ step 2500 |
| final eval loss | 0.1727 |
| final eval pos | 0.1594 |
| final eval ctr | 0.0133 |
| final eval pos/gripper | 0.1392 |
| final grad norm | 0.056 |

v5b는 v5보다 full success가 0.9 pp 높지만, avg completed는 0.027 낮고 first-subtask도 0.6 pp 낮다. 즉 전체 성공은 조금 올랐지만, base 보존 관점에서는 v6b보다 약하다.

first-subtask 기준 주요 task:

| first subtask | n | base | v2 | v5 | v5b | v6b |
|---|---:|---:|---:|---:|---:|---:|
| push_red_block_right | 29 | 41.4% | 86.2% | 44.8% | 55.2% | 58.6% |
| push_pink_block_right | 29 | 65.5% | 96.6% | 65.5% | 62.1% | 69.0% |
| push_blue_block_right | 23 | 52.2% | 65.2% | 47.8% | 52.2% | 47.8% |
| push_blue_block_left | 35 | 97.1% | 77.1% | 74.3% | 80.0% | 88.6% |
| lift_blue_block_table | 24 | 100.0% | 100.0% | 95.8% | 79.2% | 95.8% |
| lift_pink_block_table | 25 | 92.0% | 88.0% | 88.0% | 84.0% | 76.0% |
| push_into_drawer | 15 | 100.0% | 66.7% | 100.0% | 93.3% | 100.0% |

v5b에서 v5 대비 좋아진 task:

| first subtask | n | v5 | v5b | diff |
|---|---:|---:|---:|---:|
| push_red_block_right | 29 | 44.8% | 55.2% | +10.3 pp |
| push_blue_block_left | 35 | 74.3% | 80.0% | +5.7 pp |
| push_blue_block_right | 23 | 47.8% | 52.2% | +4.3 pp |

v5b에서 v5 대비 나빠진 task:

| first subtask | n | v5 | v5b | diff |
|---|---:|---:|---:|---:|
| lift_blue_block_table | 24 | 95.8% | 79.2% | -16.7 pp |
| turn_off_led | 54 | 100.0% | 90.7% | -9.3 pp |
| push_into_drawer | 15 | 100.0% | 93.3% | -6.7 pp |
| lift_pink_block_table | 25 | 88.0% | 84.0% | -4.0 pp |
| push_pink_block_right | 29 | 65.5% | 62.1% | -3.4 pp |

해석:

- reweight를 제거하면 `push_red_block_right`, `push_blue_block_right` correction은 일부 살아난다.
- 하지만 v2처럼 강한 `push_*_right` 개선은 재현되지 않았다.
- `lift_blue_block_table`, `turn_off_led`, `push_into_drawer`가 하락해서 no-reweight contrastive signal이 일부 base task를 다시 흔드는 것으로 보인다.
- v5b는 v5보다 full success가 조금 높지만 v6b보다 낮다. 현재는 `LoRA-only + reweight min 0.75 + batch 8`인 v6b가 더 좋은 trade-off다.

## v6: `xvla_calvin_lora_v6_260501`

목적:

- v5의 숨은 변수였던 `freeze_steps=1000`을 제거한다.
- v5와 나머지 조건은 동일하게 유지해 LoRA-only에서 freeze 제거 효과만 본다.

설정:

| setting | value |
|---|---:|
| batch size | 4 |
| iters | 10000 |
| freeze_steps | 0 |
| triplet reweight | enabled |
| triplet_weight_min | 0.75 |
| triplet_weight_max | 2.0 |
| triplet_confidence_events | 10 |
| LoRA-only | true |

비교 대상:

- v5: 같은 reweight, 같은 batch size, `freeze_steps=1000`
- v6: 같은 reweight, 같은 batch size, `freeze_steps=0`

### 결과

| run | full success | first-subtask success | avg completed |
|---|---:|---:|---:|
| base | 73.6% | 95.2% | 4.295 |
| v5 | 64.3% | 94.7% | 4.057 |
| v6 | 61.6% | 93.7% | 3.937 |

step별 성공률:

| step | base | v5 | v6 |
|---|---:|---:|---:|
| 1/5 | 95.2% | 94.7% | 93.7% |
| 2/5 | 91.0% | 89.1% | 86.9% |
| 3/5 | 87.6% | 82.6% | 79.6% |
| 4/5 | 82.1% | 75.0% | 71.9% |
| 5/5 | 73.6% | 64.3% | 61.6% |

v6는 v5보다 full success가 2.7 pp 낮고, avg completed도 0.120 낮았다. 따라서 LoRA-only 조건에서 `freeze_steps=0`만 적용하는 것은 성능 개선으로 이어지지 않았다.

학습 summary:

| metric | value |
|---|---:|
| best eval loss | 0.1266 @ step 2200 |
| final eval loss | 0.1848 |
| final eval pos | 0.1743 |
| final eval ctr | 0.0105 |
| final eval pos/gripper | 0.1553 |
| final grad norm | 0.093 |

first-subtask 기준 v5 대비 변화:

| first subtask | n | v5 | v6 | diff |
|---|---:|---:|---:|---:|
| push_blue_block_left | 35 | 74.3% | 85.7% | +11.4 pp |
| lift_red_block_table | 24 | 95.8% | 100.0% | +4.2 pp |
| push_red_block_right | 29 | 44.8% | 41.4% | -3.4 pp |
| push_into_drawer | 15 | 100.0% | 93.3% | -6.7 pp |
| lift_pink_block_table | 25 | 88.0% | 80.0% | -8.0 pp |
| push_pink_block_left | 31 | 100.0% | 90.3% | -9.7 pp |
| lift_blue_block_table | 24 | 95.8% | 66.7% | -29.2 pp |

해석:

- freeze 제거만으로는 v5의 약점인 `push_*_right` correction이 살아나지 않았다.
- 오히려 `lift_blue_block_table` 같은 base가 잘하던 task가 크게 흔들렸다.
- v5의 `freeze_steps=1000`이 LoRA-only 초반 학습을 지연시키는 문제는 있었지만, 그것만 제거하면 contrastive/positive signal이 더 거칠게 들어가 base 보존이 나빠지는 것으로 보인다.

## v6b: `xvla_calvin_lora_v6b_260501`

목적:

- LoRA-only + freeze0 조건에서 batch size를 키워 gripper spike와 triplet sampling noise가 줄어드는지 확인한다.
- 총 sample 노출량을 v5/v6와 비슷하게 맞추기 위해 `batch_size=8`, `iters=5000`을 사용한다.

설정:

| setting | value |
|---|---:|
| batch size | 8 |
| iters | 5000 |
| freeze_steps | 0 |
| triplet reweight | enabled |
| triplet_weight_min | 0.75 |
| triplet_weight_max | 2.0 |
| triplet_confidence_events | 10 |
| LoRA-only | true |

비교 대상:

- v6: batch 4, 10000 steps
- v6b: batch 8, 5000 steps

두 실험은 총 sample 노출량을 대략 맞춘다.

```text
v6:  batch_size 4 * 10000 steps = 40000 samples
v6b: batch_size 8 * 5000 steps  = 40000 samples
```

단, 둘은 완전히 같은 학습은 아니다. v6b는 더 큰 batch로 gradient를 평균내고, optimizer update 횟수는 절반이다. 즉 같은 양의 sample을 보되 더 낮은 gradient noise로 덜 자주 업데이트한 실험이다.

### 결과

| run | full success | first-subtask success | avg completed |
|---|---:|---:|---:|
| base | 73.6% | 95.2% | 4.295 |
| v5 | 64.3% | 94.7% | 4.057 |
| v6 | 61.6% | 93.7% | 3.937 |
| v6b | 66.2% | 95.3% | 4.094 |

step별 성공률:

| step | base | v5 | v6 | v6b |
|---|---:|---:|---:|---:|
| 1/5 | 95.2% | 94.7% | 93.7% | 95.3% |
| 2/5 | 91.0% | 89.1% | 86.9% | 89.7% |
| 3/5 | 87.6% | 82.6% | 79.6% | 82.6% |
| 4/5 | 82.1% | 75.0% | 71.9% | 75.6% |
| 5/5 | 73.6% | 64.3% | 61.6% | 66.2% |

v6b는 v5보다 full success가 1.9 pp 높고, v6보다 4.6 pp 높았다. LoRA-only 계열에서는 현재 가장 좋은 결과다.

학습 summary:

| metric | value |
|---|---:|
| best eval loss | 0.1263 @ step 600 |
| final eval loss | 0.1743 |
| final eval pos | 0.1632 |
| final eval ctr | 0.0111 |
| final eval pos/gripper | 0.1434 |
| final grad norm | 0.136 |

| 비교 | full success | avg completed | first-subtask |
|---|---:|---:|---:|
| v6b vs base | -7.4 pp | -0.201 | +0.1 pp |
| v6b vs v5 | +1.9 pp | +0.037 | +0.6 pp |
| v6b vs v6 | +4.6 pp | +0.157 | +1.6 pp |

first-subtask 기준 주요 task:

| first subtask | n | base | v2 | v5 | v6 | v6b |
|---|---:|---:|---:|---:|---:|---:|
| push_red_block_right | 29 | 41.4% | 86.2% | 44.8% | 41.4% | 58.6% |
| push_pink_block_right | 29 | 65.5% | 96.6% | 65.5% | 65.5% | 69.0% |
| push_blue_block_right | 23 | 52.2% | 65.2% | 47.8% | 47.8% | 47.8% |
| rotate_blue_block_right | 28 | 100.0% | 67.9% | 100.0% | 100.0% | 100.0% |
| lift_blue_block_slider | 30 | 100.0% | 56.7% | 100.0% | 100.0% | 100.0% |
| push_into_drawer | 15 | 100.0% | 66.7% | 100.0% | 93.3% | 100.0% |
| push_blue_block_left | 35 | 97.1% | 77.1% | 74.3% | 85.7% | 88.6% |

v6b에서 v5 대비 좋아진 task:

| first subtask | n | v5 | v6b | diff |
|---|---:|---:|---:|---:|
| push_blue_block_left | 35 | 74.3% | 88.6% | +14.3 pp |
| push_red_block_right | 29 | 44.8% | 58.6% | +13.8 pp |
| push_pink_block_right | 29 | 65.5% | 69.0% | +3.4 pp |

v6b에서 base 대비 남은 하락:

| first subtask | n | base | v6b | diff |
|---|---:|---:|---:|---:|
| lift_pink_block_table | 25 | 92.0% | 76.0% | -16.0 pp |
| push_blue_block_left | 35 | 97.1% | 88.6% | -8.6 pp |
| push_blue_block_right | 23 | 52.2% | 47.8% | -4.3 pp |
| lift_blue_block_table | 24 | 100.0% | 95.8% | -4.2 pp |
| lift_red_block_table | 24 | 100.0% | 95.8% | -4.2 pp |

해석:

- batch size를 8로 키우고 update 수를 5000으로 줄인 것이 base 보존에 유리했다.
- 같은 총 sample 수라도 v6b는 gradient noise가 낮아져 gripper/triplet sampling noise를 덜 따라간 것으로 보인다.
- final eval gripper loss도 v6의 0.1553보다 v6b의 0.1434가 낮다. v5b의 0.1392가 더 낮기는 하지만, benchmark에서는 v6b가 더 좋으므로 gripper loss 하나만으로 최종 성능을 설명하기는 어렵다.
- `push_red_block_right`는 v5보다 회복했지만, v2 수준의 강한 correction은 아직 아니다.
- `push_blue_block_right`는 v5/v6/v6b 모두 47.8%로 고정되어, 현재 LoRA-only + reweight 조건에서는 잘 교정되지 않는다.
- v6b가 현재 가장 좋은 LoRA-only 후보지만, long-horizon step 5 success는 base보다 7.4 pp 낮아 compounding error는 남아 있다.

## 새 loss 계열 실험 규칙

v1~v6b까지는 기존 squared-hinge triplet loss 계열 실험으로 관리했다.
다음 실험부터는 probabilistic NCE 기반의 새 loss를 사용하므로, 기존 `v` 계열과 분리해서 기록한다.

버전 규칙:

| prefix | 의미 | loss 설정 |
|---|---|---|
| `v` | 기존 triplet 계열 | `--loss_type triplet_squared_hinge` |
| `d` | 거리 기반 probabilistic NCE | `--loss_type distance_probabilistic_nce` |
| `d_kl` | 거리 기반 probabilistic NCE + teacher preference KL | `--loss_type distance_probabilistic_nce --use_preference_kl` |
| `e` | 임베딩 기반 probabilistic NCE | `--loss_type embedding_probabilistic_nce` |
| `e_kl` | 임베딩 기반 probabilistic NCE + teacher preference KL | `--loss_type embedding_probabilistic_nce --use_preference_kl` |

새 loss 실험의 목적:

- 기존 hinge triplet loss의 hard margin / squared penalty가 만드는 과한 correction을 줄인다.
- positive와 negative 사이의 거리 차이를 확률적 preference로 해석한다.
- base 성능을 보존하면서 실패 action보다 성공 action에 더 가까워지도록 약하게 유도한다.
- KL 실험에서는 frozen base teacher의 preference 분포를 reference로 사용해 base behavior 파괴를 더 줄일 수 있는지 확인한다.
- 이후 `e` 계열에서는 raw action distance 대신 학습 가능한 action embedding의 cosine similarity로 positive/negative preference를 계산한다.

### 완료된 새 loss 실험

네 실험 모두 loss 변경 효과를 보기 위해 단순한 no-reweight 조건으로 진행했다.

- LoRA-only 유지
- `batch_size=4`
- `iters=10000`
- `freeze_steps=0`
- triplet reweight는 사용하지 않는다.
- checkpoint: `ckpt-10000`
- CALVIN benchmark: 1000 sequences, 4 shards, rollout video 저장

여기서 reweight는 `--use_triplet_balance_weights`로 켜는 frequency/confidence 기반 sample weighting 전체를 의미한다.
이전 v4/v5/v6/v6b에서 사용한 `triplet_weight_min=0.75`는 이 reweight 안에서 frequent task의 downweight를 완화하기 위한 하한 cap이었다.
이번 `d` / `d_kl` / `e` / `e_kl` 실험에서는 `--use_triplet_balance_weights` 자체를 넣지 않았다.

공통 세팅:

| setting | value |
|---|---:|
| manifest | `train/outputs/xvla_calvin_base_v2_260427` |
| LoRA-only | true |
| batch size | 4 |
| iters | 10000 |
| freeze steps | 0 |
| triplet reweight | disabled |
| lambda_pos | 1.0 |
| lambda_nce | 1.0 |

run별 loss 세팅:

| run | output_dir | loss_type | tau | KL | lambda_kl | extra |
|---|---|---|---:|---:|---:|---|
| `d` | `train/runs/xvla_calvin_lora_d_260502` | `distance_probabilistic_nce` | 1.0 | false | 0.0 | `lambda_gripper_nce=1.0` |
| `d_kl` | `train/runs/xvla_calvin_lora_d_kl_260502` | `distance_probabilistic_nce` | 1.0 | true | 1.0 | `lambda_gripper_nce=1.0` |
| `e` | `train/runs/xvla_calvin_lora_e_260503` | `embedding_probabilistic_nce` | 0.07 | false | 0.0 | embed dim 64, hidden 128, LR 1e-4 |
| `e_kl` | `train/runs/xvla_calvin_lora_e_kl_260503` | `embedding_probabilistic_nce` | 0.07 | true | 1.0 | embed dim 64, hidden 128, LR 1e-4 |

Benchmark 결과:

| run | full success | first-subtask success | avg completed |
|---|---:|---:|---:|
| base | 73.6% | 95.2% | 4.295 |
| v5b | 65.2% | 94.1% | 4.030 |
| v6b | 66.2% | 95.3% | 4.094 |
| d | 63.9% | 94.0% | 3.994 |
| d_kl | 61.5% | 93.1% | 3.925 |
| e | 57.3% | 93.3% | 3.798 |
| e_kl | 57.8% | 93.0% | 3.784 |

step별 success:

| run | step 1/5 | step 2/5 | step 3/5 | step 4/5 | step 5/5 |
|---|---:|---:|---:|---:|---:|
| d | 94.0% | 87.8% | 81.0% | 72.7% | 63.9% |
| d_kl | 93.1% | 86.6% | 79.9% | 71.4% | 61.5% |
| e | 93.3% | 85.2% | 76.4% | 67.6% | 57.3% |
| e_kl | 93.0% | 84.3% | 76.0% | 67.3% | 57.8% |

학습 summary:

| run | final eval total | eval BC/pos | eval NCE | eval KL | final grad norm |
|---|---:|---:|---:|---:|---:|
| d | 0.2052 | 0.1766 | 0.0285 | - | 0.148 |
| d_kl | 0.2077 | 0.1748 | 0.0297 | 0.0032 | 0.069 |
| e | 0.1692 | 0.1681 | 0.0011 | - | 0.047 |
| e_kl | 0.1845 | 0.1705 | 0.0010 | 0.0131 | 0.038 |

`e` 계열 sanity check:

- `emb_sim_gap = sim(pred, positive) - sim(pred, negative)`가 학습 중 커지는지 확인한다.
- `emb_prob_pos`가 초반부터 1.0에 붙으면 temperature가 너무 작거나 task가 너무 쉬운 것이다.
- `emb_prob_pos`가 계속 0.5 근처면 embedding contrastive signal이 충분히 작동하지 않는 것이다.
- `emb_raw_norm_*`와 `emb_raw_norm_std`로 embedding collapse 여부를 확인한다.

`e` 계열 최종 로그:

| run | eval sim_pos | eval sim_neg | eval sim_gap | eval logit_gap | train prob_pos | raw norm std |
|---|---:|---:|---:|---:|---:|---:|
| e | 0.270 | 0.009 | 0.260 | 3.719 | 0.99998 | 0.438 |
| e_kl | 0.266 | 0.008 | 0.258 | 3.685 | 0.99999 | 0.454 |

해석:

- `d`는 새 loss 계열 중 가장 좋은 결과였지만, v5b/v6b보다 낮았다.
- `d_kl`은 `d`보다 full success가 2.4 pp 낮았다. 이번 세팅에서는 `lambda_kl=1.0`이 correction을 보존하기보다는 성능을 더 누른 것으로 보인다.
- `e` 계열은 train/eval loss는 낮았지만 benchmark는 크게 낮았다. action embedding NCE가 loss를 쉽게 줄이는 방향으로 학습되었지만, 실제 rollout action 품질 개선으로는 잘 이어지지 않은 것으로 보인다.
- `e_kl`은 `e`보다 full success가 0.5 pp 높았지만 avg completed는 0.014 낮았다. KL 추가 효과는 제한적이었다.
- 현재 기준으로는 새 loss 계열보다 기존 LoRA-only hinge 계열의 v6b가 더 좋은 trade-off다.

## 다음 확인할 것

- v5 `ckpt-3000` benchmark를 확인한다. v5 best eval loss가 step 2400 근처였기 때문에 final checkpoint보다 나을 가능성이 있다.
- `push_blue_block_right`가 계속 회복되지 않으므로, 해당 task의 triplet event 품질과 negative type을 따로 확인한다.
- `d_kl`은 KL을 켤 경우 `lambda_kl=1.0`이 강했을 수 있으므로, 재시도한다면 `lambda_kl=0.05~0.1`부터 확인한다.
- `e` 계열은 `tau=0.07`에서 train probability가 거의 1.0에 붙었으므로, 재시도한다면 `tau=0.1~0.5` sweep 또는 embedder LR 축소를 먼저 검토한다.
- v5b/v6b 모두 eval loss에서 gripper 항이 여전히 크므로, 다음 실험에서 gripper loss clipping이나 NCE gripper weight 조정을 검토한다.

## 서버 모델 + 로컬 CALVIN 시뮬 연결 방법

서버에서 X-VLA model server를 띄우고 로컬에서 CALVIN simulation benchmark를 돌릴 때는 SSH port forwarding을 사용한다.

중요한 점:

- 모델 서버가 실제로 보이는 서버 노드를 tunnel 목적지로 지정한다.
- 이번 실험에서는 서버 터미널에서 `hostname`이 `pleiades3`였고, `curl http://127.0.0.1:8000/config`가 성공했다.
- 로컬에서 `-L 8100:127.0.0.1:8000`로 연결하면 destination이 애매하게 꼬일 수 있으므로, `pleiades3:8000`처럼 노드명을 명시한다.
- `ssh -N -L ...` 명령은 비밀번호 입력 후 아무 출력 없이 멈춰 있는 것이 정상이다. 해당 터미널을 닫으면 tunnel도 닫힌다.

v6 모델 서버가 서버 `pleiades3:8000`에 떠 있을 때 로컬에서 여는 tunnel:

```bash
ssh -N \
  -L 8100:pleiades3:8000 \
  -p 3022 s20265327@10.20.23.30
```

로컬에서 tunnel 확인:

```bash
curl http://127.0.0.1:8100/config
```

정상 응답 예시:

```json
{"applied": {}, "config": {"max_batch_size": 1, "max_wait_time": 0.01}}
```

v6 benchmark:

```bash
vla-eval run-sharded \
  -c configs/calvin_eval.yaml \
  -n 2 \
  -o results/xvla_calvin_lora_v6_260501_eval/merged.json \
  --output-dir results/xvla_calvin_lora_v6_260501_eval \
  --server-url ws://localhost:8100 \
  --dev \
  -y
```

v6b 모델 서버가 서버 `pleiades3:8001`에 떠 있을 때 로컬에서 여는 tunnel:

```bash
ssh -N \
  -L 8111:pleiades3:8001 \
  -p 3022 s20265327@10.20.23.30
```

`8101`이 이미 사용 중이면 위처럼 `8111` 같은 다른 로컬 포트를 사용한다.

로컬에서 tunnel 확인:

```bash
curl http://127.0.0.1:8111/config
```

v6b benchmark:

```bash
vla-eval run-sharded \
  -c configs/calvin_eval.yaml \
  -n 2 \
  -o results/xvla_calvin_lora_v6b_260501_eval/merged.json \
  --output-dir results/xvla_calvin_lora_v6b_260501_eval \
  --server-url ws://localhost:8111 \
  --dev \
  -y
```

문제 확인 순서:

1. 서버에서 모델 서버가 보이는지 확인한다.

```bash
curl http://127.0.0.1:8000/config
curl http://127.0.0.1:8001/config
```

2. 로컬에서 tunnel이 보이는지 확인한다.

```bash
curl http://127.0.0.1:8100/config
curl http://127.0.0.1:8111/config
```

3. 위 curl이 성공한 후에만 benchmark를 실행한다.
