# X-VLA CALVIN LoRA 실험 TODO

## v3 frequency/confidence triplet weighting 이후 확인할 것

- [ ] `xvla_calvin_lora_v3_260429` checkpoint들을 평가하고 base, v1, v2와 비교한다.
- [ ] v3에서도 gradient가 불안정하거나 특정 task 성능 하락이 남으면 squared hinge loss 대신 hinge loss를 실험한다.
- [ ] hinge loss 비교 이후 `lambda_ctr`를 낮추는 실험을 한다. 시작값은 `lambda_ctr=0.5`로 둔다.
- [ ] contrastive loss 영향이 여전히 강하면 `lambda_ctr` warmup을 추가한다:
  - 초반에는 `lambda_ctr=0.0`으로 시작한다.
  - step 1000부터 3000 사이에 target value까지 선형 증가시킨다.
- [ ] task family별 수동 weighting은 마지막 수단으로 둔다. 우선은 데이터 기반 제어를 사용한다:
  - subtask inverse-frequency weighting
  - matched-event confidence weighting
  - hard/noisy negative clipping
- [ ] v5에서는 원본 X-VLA CALVIN base에서 시작하고 `--lora_only`를 켜서 LoRA adapter만 학습한다.
  - `soft_prompt_hub`, `action_encoder`, `action_decoder`는 freeze한다.
  - base의 gripper/action calibration이 보존되는지 v4와 비교한다.
