# COCO 1/20 학습·검증 속도 비교

후속 병목 진단과 EMA·NMS·지표 상태 최적화는 [추가 속도 개선 결과](performance-coco-followup.md)를 참고한다.

## 범위

이번 변경은 평가 기준과 학습 구조를 유지하면서 불필요한 계산을 제거한다.

- Box loss: 모든 양성 박스 쌍의 CIoU 행렬에서 대각선을 추출하는 대신 대응 쌍만 계산한다. Matcher의 전체 쌍 IoU는 유지한다.
- 손실 정규화: Python `max` 대신 `clamp_min`을 사용한다. 손실 로그에는 `.item()` 대신 detached tensor를 전달한다.
- 검증: Main 출력 뒤의 AUX 실행을 생략한다. 배치에서는 metric state만 갱신하고 전체 검증 종료 시 AP를 계산한다.
- 검증 진행 표시: 배치 AP 대신 처리한 배치 수를 표시한다. 이미지 로깅용 예측과 반환 tuple 구조는 유지하며 두 번째 값은 `None`이다. 외부 콜백이 배치 AP를 읽는다면 함께 수정해야 한다.

해상도, NMS threshold, 최대 검출 수, augmentation, optimizer, gradient accumulation, EMA, precision, deterministic 설정은 비교 간 동일하다. DataLoader shuffle 설정이 생성자에 전달되지 않는 기존 동작도 이번 속도 변경에서 바꾸지 않았다.

## 부분집합 생성

```bash
.venv/bin/python scripts/prepare_coco_subset.py
```

일반 학습 진입점에서도 `dataset=coco-subset`으로 선택할 수 있다. 이 설정은 자동 다운로드를 끄므로 부분집합을 먼저 생성해야 한다.

`data/coco-1over20`에 train2017 5,914장(118,287장의 약 5%), val2017 250장(5,000장의 5%)을 생성한다. Split마다 독립적인 고정 seed 샘플링으로 선택하고 이미지 ID 순서로 정렬한다. 원본 train/val 구분을 유지하며 split 사이 이미지 ID 중복을 검사한다.

이미지는 원본 파일로 연결하는 심볼릭 링크이고, COCO JSON은 선택 이미지와 해당 annotation만 포함한다. 80개 category 정의, crowd 등 원본 annotation 필드는 보존한다. `manifest.json`에는 이미지 ID 목록, 원본·부분집합 주석 SHA-256, seed와 개수를 기록한다. 같은 설정으로 다시 실행할 수 있으며 다른 부분집합이 이미 있는 디렉터리는 덮어쓰지 않는다. 원본 이미지 경로가 바뀌면 새 경로에서 부분집합을 다시 생성해야 한다.

원본 loader는 segmentation에서 box를 만들고 crowd를 제외하는 등 자체 평가 규약을 사용한다. 이번 결과는 그 규약을 유지한 비교이며 공식 COCO 원본 bbox 평가와의 일치 여부를 새로 검증한 것은 아니다.

## 재현

기준 소스는 변경 전 commit `c4cb5f6f56102eceeaa7d75e23e1125cd0373eaf`의 `yolo` 코드다. 이번 실행에서는 변경 직전 디렉터리를 `runs/performance/baseline-source/yolo`에 보관했다. 새 checkout에서는 다음과 같이 준비할 수 있다.

```bash
mkdir -p runs/performance/baseline-source
git archive c4cb5f6f56102eceeaa7d75e23e1125cd0373eaf yolo | tar -x -C runs/performance/baseline-source

.venv/bin/python scripts/benchmark_coco_subset.py \
  --source-root runs/performance/baseline-source --output runs/performance/baseline-1
.venv/bin/python scripts/benchmark_coco_subset.py --output runs/performance/optimized-1
.venv/bin/python scripts/benchmark_coco_subset.py \
  --source-root runs/performance/baseline-source --output runs/performance/baseline-2
.venv/bin/python scripts/benchmark_coco_subset.py --output runs/performance/optimized-2
.venv/bin/python scripts/benchmark_coco_subset.py \
  --source-root runs/performance/baseline-source --output runs/performance/baseline-3
.venv/bin/python scripts/benchmark_coco_subset.py --output runs/performance/optimized-3

.venv/bin/python scripts/benchmark_coco_subset.py --validation-only \
  --eval-state runs/performance/baseline-1/ema.pt --output runs/performance/fixed-optimized-restored
.venv/bin/python scripts/compare_coco_benchmarks.py \
  --baseline runs/performance/baseline-1 runs/performance/baseline-2 runs/performance/baseline-3 \
  --optimized runs/performance/optimized-1 runs/performance/optimized-2 runs/performance/optimized-3 \
  --fixed-reference runs/performance/baseline-1 \
  --fixed-optimized runs/performance/fixed-optimized-restored --output runs/performance/comparison.json
```

GPU 실행을 순차적으로 수행해 GPU 경쟁을 피한다. 결과를 보존하기 위해 완료된 출력 경로는 재사용하지 않는다.

기본 설정은 v9-t, 무작위 초기화, 입력 320×320, batch 32, workers 4, seed 10, FP16 mixed precision, GPU 1개다. 각 학습 실행은 185개 train batch와 8개 validation batch를 모두 처리한다. 이 실험은 속도와 파이프라인 회귀 확인용이며 충분히 학습한 모델의 정확도를 나타내지 않는다.

## 측정 방법

- 학습 시간: train epoch 시작부터 최종 validation 시작까지. 데이터 대기, 전송, forward/backward, optimizer, EMA, 로깅을 포함한다.
- 검증 시간: validation 시작부터 종료까지. forward, NMS, metric 갱신과 최종 compute, 비교용 CPU 예측 저장 준비를 포함한다.
- CUDA synchronize를 구간 경계에서 수행한다. 배치마다 측정을 위한 synchronize는 넣지 않는다.
- Dataset 생성과 annotation cache 준비는 구간 측정에서 제외한다. `fit_or_validate_seconds`는 Trainer 실행 전체를 측정해 setup과 sanity validation도 포함한다. 디스크 checkpoint 저장은 측정에서 제외하며, 결과 비교를 위해 종료 후 EMA와 예측을 저장한다.
- 메모리는 구간별 `max_memory_allocated`이며 드라이버 전체 사용량이나 reserved 메모리가 아니다.
- `result.json`에 실제 import 경로, Python 소스 SHA-256, 데이터 manifest SHA-256, 설정, 전체 batch loss, metric을 저장한다.
- 비교 도구는 소스·설정·데이터 식별자와 loss 개수 일치를 확인한다. 실행 도구는 소스와 loader cache의 실행 전후 변경도 검사하고, 고정 가중치 검증 시 입력 EMA의 SHA-256을 기록한다. 이 무결성 검사는 초기 비교 후 추가했으므로 초기 실행 JSON에는 cache hash 필드가 없다. 현재 cache의 이미지 ID와 manifest 일치 및 해시는 `runs/performance/dataset-cache-verification.json`에 별도 기록했다.
- 별도로 기준 실행의 동일 EMA를 개선 검증 경로에 넣어 250장 예측과 전체 AP/AR가 일치하는지 확인한다. 서로 다른 학습 결과의 AP 비교만으로 동등성을 주장하지 않는다.

## 회귀 테스트

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  tests/test_tools/test_loss_performance_equivalence.py \
  tests/test_tools/test_validation_performance_equivalence.py -q
```

CIoU 값과 gradient, 빈 양성 집합, 퇴화 box, 기존 pairwise 동작, detached tensor 로깅, v9-t/s/m/c와 v7 Main 출력, metric update/compute 동작을 검사한다.

기존 IoU 회귀 테스트도 다음 명령으로 확인했다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  tests/test_utils/test_bounding_box_utils.py \
  -k 'calculate_iou or calculate_diou or calculate_ciou' -q
```

## 2026-09-11~12 측정 결과

RTX 2070 SUPER / Python 3.13.12 / PyTorch 2.6.0+cu124 / Lightning 2.6.6 / TorchMetrics 1.9.0에서 실행했다. 패키지 버전은 `runs/performance/environment.json`에 기록되어 있다.

| 실행 | 기존 학습(초) | 개선 학습(초) | 기존 검증(초) | 개선 검증(초) | 기존 Trainer 전체(초) | 개선 Trainer 전체(초) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 68.136 | 57.415 | 26.657 | 17.427 | 98.884 | 77.922 |
| 2 | 59.512 | 55.593 | 26.077 | 17.141 | 88.824 | 75.706 |
| 3 | 58.063 | 58.077 | 25.922 | 16.307 | 87.080 | 77.205 |
| 중앙값 | **59.512** | **57.415** | **26.077** | **17.141** | **88.824** | **77.205** |

- 검증: 중앙값 시간 **34.3% 감소**, 처리량 **1.52배**. 세 실행 모두 감소했다.
- Trainer 전체: 중앙값 시간 **13.1% 감소**. Setup·sanity 검증을 포함하며 checkpoint 디스크 저장은 제외한다.
- 학습: 중앙값 시간 **3.5% 감소**. 세 번째 실행은 58.063→58.077초로 사실상 동일하다. 첫 기준 실행 중 짧은 CPU 회귀 테스트와 초기 이미지 캐시 효과가 있었으므로 첫 쌍의 18.7% 처리량 증가를 안정적인 학습 개선폭으로 해석하면 안 된다. 학습 속도 향상은 실행 편차와 구분하기 어렵다.
- 학습 최대 CUDA allocated 메모리: 세 번 모두 기존 **2,409,318,400 bytes (2.24 GiB)**, 개선 **2,027,572,224 bytes (1.89 GiB)**로 **15.8% 감소**했다.
- 3쌍 각각 185개 학습 batch loss의 최대 절댓값 차이는 **0**이다. 모든 실행에서 전체 부분집합 처리, 유한한 손실·가중치, 초기 가중치 변경을 확인했다.
- 같은 기준 EMA를 복원한 독립 검증에서 **250장 전체 예측 tensor가 bitwise 동일**하고 AP/AR 16개 지표가 정확히 같았다. 기준 subset mAP는 약 `4.434568e-6`으로, 1 epoch 무작위 초기화 모델의 속도 점검 결과다.
- 새 회귀 테스트 17개와 기존 IoU 테스트 4개가 통과했다. Box loss 구현과 테스트는 GPT-5.6 Sol에 위임하고, 통합·GPU 실험·비교 검토는 주 작업에서 수행했다.

기계 판독 요약은 `runs/performance/comparison.json`, 각 원시 결과는 `runs/performance/{baseline,optimized}-{1,2,3}/result.json`, 최종 고정 가중치 검증은 `runs/performance/fixed-optimized-restored/`에 있다. `runs/`와 `data/`는 Git 제외 대상이다. 스크립트·dataset YAML·테스트·이 문서는 저장소 변경에 포함된다.

### 고정 가중치 검사에서 발견한 기존 초기화 동작

첫 독립 검증(`runs/performance/fixed-optimized`)에서는 예측 일치 검사가 실패했다. 기존 `Vec2Box.create_auto_anchor()`가 train mode의 dummy forward로 BatchNorm running mean/variance와 batch counter를 변경하기 때문이다. 학습 도중 EMA 검증과, checkpoint 로드 후 새 모델의 setup을 거치는 단독 검증은 이 초기화 경로가 다르다.

벤치마크의 validation-only 모드는 setup 종료 후 입력 checkpoint를 다시 복원하고, 모든 state tensor가 checkpoint와 동일함을 확인하도록 수정했다. 실제로 변경된 버퍼 873개를 기록했고, 복원 후 예측과 지표 비교가 통과했다. 실패한 첫 독립 검증은 동등성 근거에서 제외한다. 이 보정은 비교 도구에 적용했으며 기존 제품 코드의 auto-anchor 초기화 동작 자체는 이번 속도 최적화에서 변경하지 않았다.

이번 결과는 320 입력의 v9-t와 해당 부분집합에서의 관찰이다. 전체 COCO, 640 입력, 장기간 학습이나 더 큰 모델에서 같은 개선율을 보장하지 않는다. NMS 기준이나 평가 이미지 수를 줄여 얻은 속도 향상은 아니다.
