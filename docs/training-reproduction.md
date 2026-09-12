# Stride 탐색과 학습 update 재현성

2026-09-12, 재현성 감사의 2번과 3번에 대한 수정이다. 비교 대상은 WongKinYiu/yolov9의
`5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff`에 있는 `train_dual.py`의 누적 규칙이다.
원본 소스를 복사하지 않고 해당 동작을 독립적으로 구현했다.

## 2. Stride 자동 탐색

`Vec2Box`와 `Anc2Box`는 공통 probe를 사용한다. 실제 모델 parameter/buffer의 device와 dtype으로
입력을 만들고 `eval()` 및 `torch.inference_mode()`에서 실행한다. 탐색 전에 모든 하위 모듈의
`training` 값을 저장하고 `finally`에서 각각 복원한다. 따라서 일부 BatchNorm만 eval인 혼합 상태와
forward 예외도 처리한다. 직사각형 입력에서는 높이와 너비에서 구한 stride가 일치하는지 확인한다.

실제 v9-t/s/m의 probe 전후 전체 `state_dict`가 정확히 같고 strides가 `[8,16,32]`인지 검사한다.
이 검증은 probe가 로드된 BN 통계를 오염시키는 문제를 차단하며, pretrained AP 개선 폭을 뜻하지 않는다.

## 3. 누적 gradient와 optimizer step

`TrainModel`은 Lightning manual optimization을 사용한다. `B`를 설정된 GPU당 batch 크기,
`W`를 world size, `N`을 nominal batch 크기, `nb`를 실제 rank당 epoch batch 수라고 하면:

- backward 입력은 `loss × 실제 local batch 크기 × W`이다. 누적 횟수로 나누지 않는다.
- `ni = epoch × nb + batch_idx`로 microbatch 시간을 계산한다.
- `nw = max(round(warmup_epochs × nb), min_iterations)`이며 기본 `min_iterations=100`이다.
- warmup 중 누적 횟수는 `max(1, round(1 + (N/(B×W)-1) × ni/nw))`이다. 비율을 먼저 반올림하지 않는다.
- 이후에는 `max(1, round(N/(B×W)))`를 사용한다.
- `ni - last_opt_step >= accumulation`일 때만 optimizer step을 호출한다.
- 원본처럼 epoch 마지막에 임의의 추가 step을 실행하지 않는다. 남은 gradient는 다음 epoch 시작에
  폐기하지만 `last_opt_step`은 이어간다.
- 모든 microbatch에서 DDP gradient 동기화를 수행한다. DDP의 rank 평균은 위의 `W` 곱으로 보정한다.
- AMP unscale 이후 `on_before_optimizer_step`에서 norm clipping을 적용한다.
- scheduler는 epoch 끝에 명시적으로 한 번 진행한다.

`GradientAccumulation` callback은 기존 설정 코드와의 연결을 유지하는 호환용 marker다.
누적 계산은 callback 유무에 관계없이 `TrainModel`이 담당한다. `equivalent_batch_size`가 없으면
설정된 global batch를 nominal로 사용한다(누적 1).

### EMA와 AMP

EMA는 실제 torch optimizer의 post-step hook에서 갱신한다. 일반 GradScaler가 optimizer step을
생략하거나 fused optimizer의 `found_inf`가 표시되면 갱신하지 않는다. sanity validation을 꺼도
train 시작 시 초기화한다. tau를 world size로 나누지 않으며 정수 buffer는 지수 평균하지 않는다.
EMA 값과 update counter는 callback checkpoint state에 저장한다.

**의도적인 차이:** 원본은 AMP overflow로 가중치 update가 생략돼도 EMA counter를 증가시킨다.
여기서는 성공한 update만 센다. overflow가 없는 조건에서 update/EMA 수치 일치를 검사하고,
overflow 조건은 별도의 성공 횟수/상태 보존 검증으로 구분한다. Lightning `global_step`은
AMP update 성공 수가 아니라 optimizer 호출 시도 수일 수 있다.

### 호출 설정

기본 설정은 `task.gradient_clip_val=10.0`, `task.gradient_clip_algorithm=norm`이다.
외부에서 `Trainer`를 만들 때는 `gradient_clip_val`을 지정하지 않고
`accumulate_grad_batches=1`(기본값)을 유지해야 한다. Lightning의 자동 clipping/누적과
수동 제어를 중복 적용할 수 없기 때문이다. 프로젝트 CLI와 검증/benchmark 호출부를 함께 수정했다.
테스트에서 warmup 없는 고정 누적을 사용하려면 `task.scheduler.warmup.epochs=0`,
`task.scheduler.warmup.min_iterations=0`을 지정한다.

## 검증과 범위

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  tests/test_utils/test_auto_stride_probe.py \
  tests/test_utils/test_bounding_box_utils.py \
  tests/test_tools/test_training_step_parity.py \
  tests/test_tools/test_ema_optimizer_steps.py \
  tests/test_tools/test_amp_training_clipping.py \
  tests/test_tools/test_coco_validation_integration.py
```

독립적인 float64 PyTorch 기준 루프와 실제 `TrainModel.training_step`/Lightning Trainer를 비교한다.
고정/가변 누적, 두 epoch, 불완전 epoch tail, 작은 마지막 batch, clipping 유무, SGD momentum,
scheduler 및 EMA를 포함한다. CPU 2-process DDP는 동일한 global batch의 합산 기준과 비교한다.
허용 오차는 절대 `1e-12`이다. CUDA 테스트에서는 GradScaler overflow와 clipping 순서를 별도로 검사한다.
위 테스트와 기존 COCO/EMA/loss/NMS 회귀 테스트를 합친 최종 실행은 **94 passed**였다.
CUDA 조건부 테스트도 실행했으며 skip은 없었다.

| 비교 조건 | optimizer 호출 iteration (0부터) | 가중치 최대 절대 오차 | momentum 최대 절대 오차 |
| --- | --- | ---: | ---: |
| 단일 프로세스, 가변 누적, clipping 적용/미적용 | 0, 2, 6 | 0 | 0 |
| 단일 프로세스, 고정 누적 4, epoch tail 폐기 | 3, 7 | 0 | 0 |
| CPU DDP 2-process, clipping 적용 | 1, 3 | 0 | 0 |
| CPU DDP 2-process, clipping 미적용 | 1, 3 | 2.78e-16 | 1.07e-14 |

가변 누적의 단일 프로세스 EMA 최대 절대 오차도 두 clipping 조건 모두 0이었다.
CUDA 해석적 검사에서는 서로 다른 scaler scale에서 clipping 직전 gradient가 정확히 4였고,
성공→overflow→성공에서 EMA counter가 `[1,1,2]`였다.

실제 RTX 2070 SUPER, torch 2.6.0+cu124에서 v9-t scratch / 128px / batch 4 / AMP로
COCO 이미지 80장(20 batch)의 진단 실행을 완료했다. optimizer 호출 10회 중 AMP 성공은 3회,
skip은 7회였으며 checkpoint의 EMA counter도 3이었다. loss와 최종 가중치가 유한하고 가중치 변경을
확인했다. 실행 기록은 `runs/train/reproduction-steps-smoke/verification.json`과 `checkpoints/last.ckpt`에 있다.
val2017 일부를 이용한 실행 경로 점검이며 성능 평가나 하이퍼파라미터 선택 실험이 아니다.

이 변경으로 논문 AP 재현을 완료했다고 주장하지 않는다. 데이터 증강, matcher, 전처리, 초기화 및
LR/momentum recipe 등 감사에 기록된 다른 차이는 별도 검증 대상이다. 동일 입력과 loss 및 optimizer
조건에서 누적 update가 맞는지를 검증한 것이다. checkpoint counter 저장은 epoch 경계 재개를 위한 것이며,
미완성 gradient 자체를 저장하지 않으므로 epoch 중간 accumulation의 정확한 재개를 보장하지 않는다.
standalone 평가의 EMA 가중치 선택 문제(감사 5번 전체)도 별도 작업이다.

후속 작업: [학습 증강 recipe(감사 4번)](augmentation-reproduction.md)도 원본에 맞췄다. 마지막 mosaic 종료는 요청에 따라 제외했다.
