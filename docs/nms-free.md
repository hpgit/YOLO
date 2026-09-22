# YOLOv9 NMS-free 학습과 추론

## 실행

기본 설정은 기존 NMS 방식이다. YOLOv9 detection 모델에 `model.nms_free=true`를
주면 1:1 헤드를 추가하며, `model=v9-t-nms-free`는 같은 설정을 제공하는 별칭이다.
기본 이름은 `v9-t`로 유지하므로 사전학습 가중치 검색과 변환기 선택도 유지된다.

```bash
# COCO 예시. 다른 데이터셋은 dataset과 dataset.path를 변경한다.
python -m yolo.lazy task=train model=v9-t-nms-free dataset=coco \
  dataset.path=/path/to/coco name=v9-t-nms-free weight=false

# 다른 YOLOv9 detection/SR 크기에도 적용 가능
python -m yolo.lazy task=train model=v9-sr-t model.nms_free=true \
  dataset=coco dataset.path=/path/to/coco name=v9-sr-t-nms-free weight=false

# 기존 FP 사전학습 가중치에서 시작 (클래스 수가 다르면 기존 부분 로드 정책 적용)
python -m yolo.lazy task=train model=v9-t-nms-free dataset=coco \
  dataset.path=/path/to/coco name=v9-t-nms-free-pretrained weight=/path/to/v9-t.pt

# 학습 결과 검증 및 추론
python -m yolo.lazy task=validation model=v9-t-nms-free dataset=coco \
  dataset.path=/path/to/coco weight=runs/train/v9-t-nms-free/checkpoints/best.pt
python -m yolo.lazy task=inference model=v9-t-nms-free \
  weight=runs/train/v9-t-nms-free/checkpoints/best.pt task.data.source=/path/to/image.jpg
```

체크포인트의 실제 저장 위치는 실행 로그를 확인한다. 기존 이름으로 다시 학습하면
최신 `.ckpt`를 자동 재개하므로 실험마다 고유한 `name`을 사용한다. `best.pt`는 검증에
사용한 EMA 가중치이며, `.ckpt`는 optimizer/scheduler/EMA 상태를 포함한다.
NMS-free `.ckpt`를 재개할 때도 동일한 모델 크기, 클래스 수, `model.nms_free=true`가 필요하다.
기존 baseline `.pt`는 새 NMS-free 학습의 초기값으로 사용할 수 있다. 이것만으로
중복 억제를 학습한 모델이 되는 것은 아니며, 1:1 손실을 통한 재학습이 필요하다.

## 구현 구조

[YOLOv10 논문](https://arxiv.org/abs/2405.14458)의 dual assignment 아이디어를 이
저장소의 YOLOv9 헤드와 손실에 맞춰 구현했다. 다른 저장소 코드를 복사하지 않았으며,
YOLOv10 전체 아키텍처나 성능을 재현한다는 의미는 아니다.

- `Main.heads`: 기존 one-to-many 헤드. backbone/neck에 조밀한 학습 신호를 준다.
- `Main.one2one_heads`: 독립된 복제 헤드. `detach()`한 neck feature에서 학습한다.
- 기존 `AUX`: 원래 auxiliary supervision을 유지한다.
- 학습 출력: `Main`(one-to-many), `One2One`, 모델에 있는 경우 `AUX`.
- eval 출력: `Main`에 one-to-one 결과만 담는다. one-to-many와 AUX 경로는 실행하지 않는다.

사전학습 baseline을 부분 로드한 다음 one-to-many 가중치를 one-to-one에 복제한다.
이미 one-to-one 가중치가 들어 있는 체크포인트는 그대로 복원하며, 이를 일반 모델에
로드하거나 손상된 one-to-one 가중치를 로드하면 오류를 낸다.

### 매칭과 손실

1:1 매칭 점수는 기존 matcher와 같은 `sigmoid(class)^0.5 × max(CIoU, 0)^6`를
기본값으로 사용한다. 각 GT는 유효 anchor 중 최고 점수 하나를 선택한다. 유효 후보가
없으면 기존 matcher처럼 가장 좋은 양수 점수 후보로 fallback한다. 같은 anchor를
여러 GT가 선택하면 그중 IoU가 가장 높은 GT만 남기며 동률은 먼저 나온 인덱스로 결정한다.
따라서 GT당 최대 1개, anchor당 최대 1개 양성 할당이다. 충돌에서 탈락한 GT는
미할당 상태일 수 있다. 모든 GT를 강제로 매칭하는 Hungarian 방식은 아니다.

매칭은 gradient 없이 float32로 계산해서 AMP의 `IoU**6` underflow를 줄인다.
빈 이미지와 padding target은 배경으로 처리한다. 1:1 헤드도 기존 CIoU, DFL,
BCE 손실을 사용하고 class target 품질은 할당된 IoU이다.

```text
loss = Main O2M loss + 0.25 × AUX loss + task.loss.one2one × O2O loss
```

각 branch의 Box/DFL/BCE 계수는 기존 `7.5 / 1.5 / 0.5`이다.
`task.loss.one2one`의 기본값은 `1.0`이며 양수여야 한다.
`Loss/BoxLoss`, `Loss/DFLLoss`, `Loss/BCELoss`는 합산 항목이고,
`Loss/One2ManyLoss`, `Loss/One2OneLoss`는 가중치를 포함한 branch별 값이다.

### 후처리와 export

각 anchor에서 점수가 가장 높은 클래스 하나를 선택하고 confidence threshold와
점수 내림차순 stable top-k만 적용한다. 박스 간 IoU 비교나 NMS 호출은 없다.
좌표 복원과 결과 형식 `[class_id, x1, y1, x2, y2, score]`는 기존과 같다.
`task.nms.min_confidence`, `task.nms.max_bbox`는 계속 사용하고 `task.nms.min_iou`는
NMS-free 모드에서 사용하지 않는다. 서로 겹치는 박스도 top-k에 들면 그대로 보존된다.

```bash
python -m yolo.lazy task=export model=v9-t-nms-free \
  weight=runs/train/v9-t-nms-free/checkpoints/best.pt \
  task.format=onnx task.output=runs/export/v9-t-nms-free.onnx
python -m yolo.lazy task=inference weight=runs/export/v9-t-nms-free.onnx \
  task.data.source=/path/to/image.jpg
```

ONNX에는 one-to-one 헤드만 실행되는 그래프와 NMS-free 후처리 메타데이터가 들어간다.
기존 DFL probability/decoded xyxy 출력 계약은 유지하며, threshold/top-k는 호스트가
처리한다. 저장소의 portable ONNX inference가 메타데이터를 읽어 같은 후처리를 선택한다.
출력 메타데이터를 제거하거나 별도 런타임을 사용한다면 이 계약을 직접 적용해야 한다.

## 검증 범위

2026-09-22 검증 결과:

- 전체 CPU 회귀: **527 passed, 8 skipped** (PyTorch 2.6.0).
- RTX 2070 SUPER의 CUDA AMP 학습/EMA/체크포인트 재개: **1 passed**.
- YOLOv9와 SR의 t/s/m/c 총 8종 실제 eval shape와 유한값 확인.
- 고정/동적 batch ONNX Runtime와 PyTorch의 decoded 결과 및 최종 top-k 일치.
  export 그래프에는 O2O 가중치만 포함되며 NMS 연산 없이 tensor rank ≤ 4 유지.

단위 테스트는 1:1 할당의 유일성, 충돌, 빈 target, AMP dtype, gradient 분리,
체크포인트 왕복, 겹치는 박스 보존, NMS 미호출, ONNX 출력을 검사한다.
통합 테스트는 임시 합성 이미지로 실제 Trainer의 forward/backward/optimizer update,
EMA, validation, best.pt 저장과 `.ckpt` 재개를 확인한다.

재현 명령(선택 export 의존성과 pytest를 설치한 환경):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests/test_model/test_nms_free.py \
  tests/test_tools/test_nms_free_loss.py \
  tests/test_tools/test_nms_free_integration.py \
  tests/test_tools/test_nms_free_export.py
```

CUDA가 보이면 통합 테스트가 CPU와 GPU를 모두 실행한다. GPU smoke는 초기
GradScaler scale을 128로 고정해 짧은 두 optimizer step에서 갱신과 재개를 확인한다.
실제 학습 CLI의 기본 AMP 설정을 변경하지는 않는다. 실물 이미지 데이터셋 학습과
장시간 수렴 검증은 이 합성 smoke의 범위에 포함되지 않는다.

이 검증은 실행 경로의 정상 동작을 확인한다. COCO 전체 학습 AP, 중복률, 실제 장치
latency 개선은 별도 측정이 필요하다. NMS-free 자체는 중복 검출이 0임을 보장하지 않는다.
YOLOv7 anchor 기반 모델이나 classification/segmentation 헤드는 이 모드를 지원하지 않는다.
NMS-free와 QAT/QDQ 조합은 현재 지원하지 않으며 명시적으로 오류를 낸다.
