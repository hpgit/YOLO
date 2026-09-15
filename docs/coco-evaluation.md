# 공식 COCO JSON 평가

현재 검증은 COCO annotation JSON이 있으면 원본 JSON을 정답으로 사용한다. 학습 중 validation과 독립 validation에 같은 경로가 적용된다. 데이터 로더에서 재구성한 박스를 정답으로 사용하는 이전 방식과 평가 수치가 달라질 수 있다.

## 사용

저장소 루트에서 실행한다. 기본값 `evaluator=auto`는 `dataset.path/annotations/instances_<validation split>.json`이 있으면 COCOeval을 선택하고, JSON이 없는 TXT 데이터셋에서는 기존 TorchMetrics 평가를 사용한다. 다만 `dataset.path/<validation split>.txt` 목록이 있으면 로더의 TXT 선택을 따라 기존 평가를 사용한다. JSON이 이때도 정답 기준이어야 한다면 `evaluator=coco` 또는 `annotation_path`를 명시한다. 로그에 선택한 평가 방식과 JSON 경로를 표시한다.

```bash
# 공식 JSON 사용을 강제한다. JSON이 없으면 오류를 내며 대체하지 않는다.
.venv/bin/python yolo/lazy.py task=validation task.evaluator=coco \
  accelerator=gpu device=1 use_wandb=false weight=weights/v9-c.pt

# 학습 중 validation도 같은 평가 경로를 사용한다.
.venv/bin/python yolo/lazy.py task=train task.validation.evaluator=coco \
  accelerator=gpu device=1 use_wandb=false

# 과거 평가 수치와 비교할 때만 이전 tensor GT 평가를 명시한다.
.venv/bin/python yolo/lazy.py task=validation task.evaluator=torchmetrics \
  accelerator=gpu device=1 use_wandb=false
```

다른 annotation 파일은 `task.annotation_path=/absolute/path/instances.json`으로 지정한다. 상대 경로는 `dataset.path` 기준이다. 학습에서는 `task.validation.annotation_path`를 사용한다. 명시한 파일이 없거나 category 수가 `dataset.class_num`과 다르면 실패한다.

## 평가 계약

- 원본 JSON의 bbox, area, crowd 정보를 그대로 COCOeval에 전달한다. bbox 전용 JSON에서 `area` 또는 `iscrowd`를 생략한 경우에만 원본 bbox 면적과 `0`으로 보완한다. 리사이즈된 박스 면적으로 small/medium/large를 다시 분류하지 않는다. 입력 형식과 로딩 검증 규칙은 [bbox annotation 문서](bbox-annotations.md)를 참고한다.
- 예측의 원본 좌표 복원은 현재 `PadAndResize`의 정수 크기 반올림 규칙과 padding을 역산한다. x/y에 실제 리사이즈된 크기 비율을 각각 사용하고 원본 경계로 clip한다. clip 후 면적이 0인 예측도 임의로 제거하지 않는다.
- 모델의 연속 class index를 JSON category ID 정렬 순서로 매핑한다. 기존 JSON 학습 로더와 같은 매핑이며 COCO 80개 ID를 하드코딩하지 않는다.
- image ID는 JSON의 file_name으로 찾는다. 비숫자 이름과 중첩 상대 경로를 지원한다. basename 대체는 유일할 때만 허용한다.
- 예측이 없는 이미지도 평가에 포함한다. 배치별 AP를 평균내지 않고 관측된 이미지 전체를 epoch 말에 평가한다. `maxDets=[1,10,100]`을 사용한다.
- 분산 실행에서는 각 rank의 이미지별 예측을 rank 0에만 모아 전체 예측이 모든 프로세스에 복제되지 않게 한다. DistributedSampler가 추가한 중복 이미지는 낮은 rank의 결과 하나만 사용하고 전체 AP를 계산해 모든 rank에 전달한다. rank별 AP를 평균내지 않는다.
- 12개 AP/AR는 COCOeval의 float64 값을 유지한다. 기존 `map`, `map_50`, `PyCOCO/AP @ .5:.95` 등의 로깅 이름과 checkpoint monitor를 유지한다.

이 역변환은 증강 없는 `PadAndResize` 검증 경로를 대상으로 한다. JSON 평가에서 `data_augment`가 비어 있지 않으면 오류를 낸다. annotation의 width/height 및 file_name은 실제 이미지와 일치해야 한다. 원본 모델의 rectangular batching·전처리·NMS·EMA 등의 다른 차이를 함께 수정한 것은 아니다.

## 검증 방법과 위임

Sol 에이전트 하나가 평가 모듈을 구현하고, 별도의 Sol 에이전트가 독립 COCOeval 대조 테스트를 작성했다. 주 작업에서 구현을 검토하고 solver/config 연결, 학습·검증 통합 테스트 및 실제 GPU 검증을 수행했다.

1. 합성 annotation과 이미 알고 있는 원본 좌표의 예측으로 직접 COCOeval 결과를 만든다. 새 평가기에는 입력 좌표로 변환한 예측을 주고 12개 지표를 `rtol=0, atol=1e-12`로 비교한다. crowd, bbox 면적과 다른 공식 area, 홀수 크기, 비연속 category ID, 빈 예측, GT 없는 이미지, 중복 경로를 포함한다. 합성 데이터 결과는 성능 지표로 사용하지 않는다.
2. CPU Gloo 2-process로 중복 제거 및 모든 rank의 동일 결과를 확인한다. 실제 다중 GPU/NCCL 실행은 검증하지 않았다.
3. 검증 로더 GT를 비워도 공식 JSON으로 정답을 찾는지, epoch reset과 학습 초기 sanity validation, 2 epoch 학습, EMA 및 `ModelCheckpoint(monitor="map")`가 함께 동작하는지 검사한다.
4. 실제 v9-c 가중치로 COCO 이미지를 한 번 추론한다. 새 경로의 원본 좌표 예측 JSON을 저장하고, 별도로 직접 생성한 COCOeval에 같은 JSON을 입력해 12개 지표를 비교한다. 이는 원본 저장소가 사용하는 COCOeval API와 같은 경로다.
5. 같은 모델 예측을 이미지 한 장씩 역순으로 replay하여 배치 경계·순서가 결과를 바꾸지 않는지 확인한다. 모델을 다시 추론하는 검사는 아니며 추론 batch 크기별 출력 차이는 이 범위에 포함하지 않는다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  tests/test_utils/test_coco_eval.py \
  tests/test_tools/test_coco_validation_integration.py \
  tests/test_tools/test_validation_performance_equivalence.py \
  tests/test_tools/test_loss_performance_equivalence.py \
  tests/test_utils/test_nms_grouping.py \
  tests/test_tools/test_ema_candidate.py -q

# 실행마다 새 디렉터리를 사용하여 이전 증거를 보존한다.
.venv/bin/python scripts/verify_coco_evaluation.py \
  --dataset data/coco-1over20 --output runs/coco-evaluation-parity/subset-repeat
.venv/bin/python scripts/verify_coco_evaluation.py \
  --output runs/coco-evaluation-parity/full-repeat --batch-size 8
```

결과 디렉터리에는 설정, 원본 모델 예측 `batches.pt`, COCO 형식 `predictions.json`, 빈 예측 이미지도 포함한 `image_ids.json`, 12개 지표와 최대 오차를 담은 `verification.json`을 저장한다. annotation·가중치·소스의 SHA-256을 기록하고 실행 도중 변경되지 않았는지 검사한다.

## 실제 데이터에서 확인한 평가 방식 차이

250장 부분집합에서 같은 106,325개 예측을 평가했다. 아래 값은 0–1 척도이며 논문 전체 COCO 수치와 비교할 수 있는 실험이 아니다.

| 지표 | 기존 tensor GT | 공식 JSON |
|---|---:|---:|
| AP | 0.54309607 | 0.54548625 |
| AP50 | 0.70555532 | 0.70904654 |
| AP small | 0.29469797 | 0.41945348 |
| AP medium | 0.53615075 | 0.59573269 |
| AP large | 0.71723813 | 0.74333593 |

이 차이는 모델 정확도 향상이 아니라 평가 정답·좌표계·area/crowd 처리 변경의 결과다. 크기별 AP를 비교할 때 기존 방식의 차이가 특히 크게 나타났다. 모든 값은 `runs/coco-evaluation-parity/subset/legacy_comparison.json`에 있다. 새 평가 경로와 독립 COCOeval의 12개 지표 차이 및 역순 replay 차이는 모두 0이었다.

원본 평가 API 근거: [WongKinYiu/yolov9 val_dual.py](https://github.com/WongKinYiu/yolov9/blob/5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff/val_dual.py#L289). 원본 모델을 실행해 논문 AP를 재현한 실험과 동일 예측의 평가 일치 실험을 구분해야 한다.

## 2026-09-12 전체 COCO val2017 결과

v9-c, 기존 `weights/v9-c.pt`, 640×640, inference batch 8, FP16 mixed precision, 기존 confidence 0.0001·NMS IoU 0.7·max_bbox 1000을 사용했다. 모델 및 NMS 조건을 평가 수치에 맞춰 튜닝하지 않았다.

- 전체 이미지: **5,000장**
- 예측 수: **2,008,361개**
- AP/AR 12개 최대 절대 오차: **0.0**
- 이미지별 역순 replay 최대 절대 오차: **0.0**
- 추론·독립 평가·replay 포함 소요: **477.8초**

| 지표 | 새 통합 경로 | 독립 COCOeval |
|---|---:|---:|
| map | 0.4807885077 | 0.4807885077 |
| map_50 | 0.6377050373 | 0.6377050373 |
| map_75 | 0.5217423835 | 0.5217423835 |
| map_small | 0.3542154682 | 0.3542154682 |
| map_medium | 0.5613626249 | 0.5613626249 |
| map_large | 0.6220933693 | 0.6220933693 |
| mar_1 | 0.3919246296 | 0.3919246296 |
| mar_10 | 0.6570356653 | 0.6570356653 |
| mar_100 | 0.7207062381 | 0.7207062381 |
| mar_small | 0.5638782715 | 0.5638782715 |
| mar_medium | 0.7791141132 | 0.7791141132 |
| mar_large | 0.8646369676 | 0.8646369676 |

증거: `runs/coco-evaluation-parity/full/verification.json`, `predictions.json`, `image_ids.json`, `batches.pt`. 수치는 0–1 척도다. 이 실행의 AP는 약 0.48079이며 논문 0.53 달성을 의미하지 않는다.

전체 실행 뒤에는 DDP 수집 대상을 rank 0으로 제한하고 auto 모드의 TXT split 우선순위를 보완했다. 전체 수치 계산과 좌표 복원 함수의 AST가 그대로임을 `final-review.json`에 기록했으며, 최종 코드의 53개 회귀 테스트가 통과했다. 최종 실제 GPU 부분집합 검증은 `runs/coco-evaluation-parity/final-subset/verification.json`에 별도로 기록한다. 전체 실행 당시 두 변경 파일의 스냅샷도 `full/source/`에 보존했다.
