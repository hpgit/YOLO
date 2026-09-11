# 추가 속도 최적화: EMA, NMS, 검증 지표 상태

이 비교의 기준은 [1차 개선](performance-coco-subset.md)을 이미 적용한 코드다. 원래 저장소 대비 개선율과 혼동하지 않도록 별도 결과 디렉터리 `runs/performance-round2/`를 사용한다.

## 병목 진단

같은 COCO 1/20 부분집합(train 5,914장, validation 250장), v9-t, 320×320, batch 32, workers 4, RTX 2070 SUPER에서 구간 경계마다 CUDA synchronize를 넣어 진단했다. 이 진단은 각 연산의 비용을 드러내는 용도이며 비동기 전체 처리량 측정과 다르다.

검증 250장의 진단:

| 구간 | 시간 |
|---|---:|
| 모델 forward | 0.405초 |
| NMS | 4.659초 |
| 지표 state 갱신 | 0.036초 |
| 최종 metric compute | 10.092초 |
| DataLoader 대기 | 0.437초 |

학습은 기준 EMA를 로드한 모델에 warmup 5배치 후 30배치를 진단했다. batch당 forward 76.1ms, backward 90.2ms, EMA 88.9ms, loss 19.4ms, optimizer 16.3ms, 데이터 대기 0.25ms였다. 이 진단의 단순 SGD 업데이트는 정식 비교의 warmup scheduler/Lightning loop와 다르므로 전체 학습 개선율의 근거로 쓰지 않는다. 현재 관측에서는 worker 수를 먼저 늘리는 것보다 EMA와 검증 후처리를 줄이는 것이 우선이다.

## 적용한 변경

### 1. 단일 프로세스 검증의 metric state를 CPU에 저장

설치된 TorchMetrics 1.9.0의 COCO 변환은 score, crowd, area를 검출마다 `.cpu().tolist()`로 읽는다. GPU에 누적한 state는 이때 작은 전송을 반복한다. `compute_on_cpu=True`로 batch 업데이트 때 CPU에 옮기면 이 비용을 줄일 수 있다.

같은 검증 진단에서 metric compute는 10.092→0.805초, state update는 0.036→0.199초였다. 전송 비용을 포함해도 큰 차이가 났으며 mAP는 동일했다. 임계값·검출 수·평가 이미지 수를 줄이지 않았다.

`ValidateModel.setup`은 Trainer의 world size가 1일 때만 CPU state를 켠다. 다중 GPU에서는 TorchMetrics/NCCL 동기화에 GPU state가 필요할 수 있어 기존 경로를 유지한다. world size에 따른 분기 테스트는 했지만 실제 다중 GPU 실행은 검증하지 않았다.

### 2. NMS 후보를 그룹별로 한 번 정렬

torchvision 0.21.0은 큰 입력에서 이미지·클래스 그룹마다 전체 후보를 `where`로 검색한다. 실제 부분집합 검증에서는 batch당 후보가 약 13만~35만 개였다.

새 구현은 stable sort로 그룹을 한 번 모으고 연속 구간에 같은 torchvision NMS를 적용한다. 그룹 안의 원래 순서와 마지막 score 정렬 방식도 유지해 동점 처리 결과를 보존한다. 작은 입력 및 tracing/scripting 경로는 기존 torchvision 구현을 사용한다.

실제 250장의 decode 결과를 저장해 두 번씩 순서를 바꾸어 비교했다. NMS 전체 시간 합계는 7.719→5.735초로 **1.346배** 빨랐고, 모든 최종 예측 tensor가 bitwise 동일했다. 입력 개수 경계, 빈 입력, 점수 동점, 단일 그룹도 회귀 테스트했다. 개별 후보의 효과이며 전체 검증 개선율과는 다르다.

### 3. EMA를 foreach 연산으로 묶기

EMA의 `current + (ema - current) * decay`를 tensor마다 따로 실행하는 대신 dtype/device별로 묶어 subtraction → multiplication → addition 세 단계를 실행한다. 연산 순서와 integer BatchNorm counter의 float 승격, 기존 decay/업데이트 주기를 유지한다. 텐서 storage는 현재 모델과 공유하지 않는다.

지원하지 않는 layout/device 또는 foreach 연산은 기존 scalar 식으로 처리한다. OOM은 숨기지 않고 그대로 전달한다. PyTorch 2.6에서는 `_foreach_*` API를 사용한다.

실제 v9-t의 1,776개 state tensor로 35회 업데이트(첫 5회 제외)를 비교했다. 업데이트당 42.60→20.15ms로 **2.11배** 빨랐고 GPU에서 값이 정확히 일치했다. 이 별도 측정은 `model.state_dict()` 생성 등의 콜백 비용을 포함하지 않으므로 위 학습 진단의 EMA 88.9ms와 직접 비교하지 않는다.

## 재현 명령

`runs/performance-round2/round1-source/yolo`에는 추가 변경 전의 1차 개선 소스 스냅샷을 보관했다. 커밋 `9f04098`의 YOLO Python 소스가 이 스냅샷과 정확히 일치한다. 각 실행 JSON에도 전체 Python 소스 해시를 저장했다.

```bash
# 새 checkout에서 1차 개선 기준 소스 복원
mkdir -p runs/performance-round2/round1-source
git archive 9f04098 yolo | tar -x -C runs/performance-round2/round1-source

# 동기화 구간 진단 (두 명령은 순서대로 실행)
.venv/bin/python scripts/profile_coco_speed.py \
  --source-root runs/performance-round2/round1-source \
  --output runs/performance-round2/profile.json
.venv/bin/python scripts/profile_coco_speed.py \
  --source-root runs/performance-round2/round1-source --train-batches 0 --metric-cpu \
  --output runs/performance-round2/profile-metric-cpu.json

# 저장된 실제 예측과 실제 모델 state 형태로 후보 비교
.venv/bin/python scripts/benchmark_nms_candidate.py
.venv/bin/python scripts/benchmark_ema_candidate.py

# GPU를 공유하지 않고 전체 부분집합 실행을 번갈아 수행
.venv/bin/python scripts/benchmark_coco_subset.py \
  --source-root runs/performance-round2/round1-source --output runs/performance-round2/before-1
.venv/bin/python scripts/benchmark_coco_subset.py --output runs/performance-round2/after-1
.venv/bin/python scripts/benchmark_coco_subset.py \
  --source-root runs/performance-round2/round1-source --output runs/performance-round2/before-2
.venv/bin/python scripts/benchmark_coco_subset.py --output runs/performance-round2/after-2
.venv/bin/python scripts/benchmark_coco_subset.py --validation-only \
  --eval-state runs/performance-round2/before-1/ema.pt --output runs/performance-round2/fixed-final
.venv/bin/python scripts/compare_coco_benchmarks.py \
  --baseline runs/performance-round2/before-1 runs/performance-round2/before-2 \
  --optimized runs/performance-round2/after-1 runs/performance-round2/after-2 \
  --fixed-reference runs/performance-round2/before-1 --fixed-optimized runs/performance-round2/fixed-final \
  --output runs/performance-round2/comparison.json
```

완료된 결과를 보존하므로 재실행할 때는 새로운 출력 디렉터리를 지정한다.

## 2026-09-12 전체 부분집합 결과

같은 seed, 데이터 manifest/cache, 이미지 크기, batch 크기, augmentation, FP16 mixed precision, deterministic 설정, gradient accumulation/EMA 스케줄을 유지했다. 기준→개선→기준→개선 순서로 실행했다. 1차 결과와 같은 벤치마크 도구를 사용했으며 Trainer 전체 시간에는 setup과 sanity validation이 포함되고 checkpoint 디스크 저장은 제외된다.

| 실행 | 기준 학습(초) | 개선 학습(초) | 기준 검증(초) | 개선 검증(초) | 기준 전체(초) | 개선 전체(초) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 61.712 | 59.857 | 16.963 | 6.141 | 81.584 | 68.981 |
| 2 | 61.253 | 56.471 | 17.526 | 6.211 | 81.876 | 65.728 |
| 중앙값 | **61.482** | **58.164** | **17.244** | **6.176** | **81.730** | **67.355** |

- 검증: 시간 **64.2% 감소**, 처리량 **2.79배**.
- 전체: 시간 **17.6% 감소**.
- 학습: 시간 **5.4% 감소**. 두 번 모두 감소했지만 개선 실행 간 편차가 있어 작은 개선폭은 보수적으로 해석해야 한다. EMA 단독 2.11배를 전체 학습 개선율로 환산하지 않는다.
- 학습 최대 CUDA allocated 메모리는 두 방식 모두 **2,027,572,224 bytes**로 같았다.
- 각 실행은 train 5,914장/185배치, val 250장/8배치를 모두 처리했다.
- 두 쌍 모두 185개 batch loss의 최대 차이가 **0**이고, 학습 후 EMA의 **1,776개 state tensor가 bitwise 동일**했다.
- 별도 고정 가중치 검증에서 250장 예측이 bitwise 동일하고 AP/AR 16개 지표가 정확히 같았다. 1차 실험에서 확인한 auto-stride probe의 BatchNorm 상태 변경을 배제하기 위해 setup 후 동일 checkpoint를 복원했다.
- 모든 측정 실행에서 소스·데이터 cache 해시를 기록하고 실행 중 변경이 없음을 확인했다.

기계 판독 요약: `runs/performance-round2/comparison.json`. 구간 진단은 `profile.json`, CPU metric 진단은 `profile-metric-cpu.json`, 후보별 비교는 `nms-comparison.json`과 `ema-comparison.json`이다.

## 회귀 검사와 검증 범위

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  tests/test_utils/test_nms_grouping.py \
  tests/test_tools/test_validation_performance_equivalence.py \
  tests/test_tools/test_ema_candidate.py -q
```

해당 범위의 **25개 테스트**가 통과했다(NMS 7, validation 9, EMA 9). EMA 후보·통합과 CPU 테스트는 GPT-5.6 Sol에 위임했고, 주 작업에서 GPU 값 일치·병목 진단·통합 실행과 NMS/metric 변경을 검증했다.

성능 측정 뒤 TorchScript 검사에서 NMS 함수의 float threshold 타입 주석 누락을 발견해 보완했다. 이 타입 주석은 Python 실행의 계산을 바꾸지 않는다. 최종 코드로 `fixed-final` 고정 가중치 검증을 다시 실행해 예측과 지표 일치를 확인했다. 초기 고정 가중치 결과 `fixed-after`도 보존했다.

현재 결과는 v9-t·320 입력·이 부분집합·GPU 1개에서 확인한 값이다. 전체 COCO 또는 다른 모델에서 동일한 개선율을 보장하지 않으며, 실제 DDP 성능/동기화는 추가 검증 범위다. DataLoader worker 증설, channels-last, torch.compile, batch 크기 변경은 이번 실행에서 비교하지 않았다. 새 병목 분포를 측정한 뒤 검토할 수 있다.
