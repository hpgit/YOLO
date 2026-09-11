# COCO 2017 1 epoch 검증

저장소 루트에서 실행합니다. 전체 COCO train2017 118,287장과 val2017 5,000장을 사용합니다.

## 환경 설치

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements.txt -e .
```

검증 당시 패키지 버전은 `runs/setup/requirements.lock.txt`에 저장됩니다.

## 데이터 준비

```bash
.venv/bin/python scripts/prepare_coco.py
```

공식 COCO S3 저장소에서 주석과 이미지 ZIP을 다운로드합니다. 분할 다운로드는 재실행하면 완료된 조각을 재사용합니다. 압축 해제 시 ZIP CRC를 검사하고 이미지 수와 주석의 파일 목록을 대조합니다. 결과는 `data/coco/verification.json`에 저장됩니다.

## 전체 1 epoch 학습

```bash
.venv/bin/python scripts/verify_coco_epoch.py
```

YOLOv9-t, 무작위 초기화(`weight=false`), 320×320, 배치 32, CUDA mixed precision, worker 4를 사용합니다. 프로젝트의 `TrainModel`, COCO 데이터 로더, 기본 augmentation/loss/optimizer/scheduler, gradient accumulation과 EMA를 사용하며 학습 및 검증 배치 수를 제한하지 않습니다. W&B 대신 로컬 CSV에 기록합니다.

이미지 크기·배치·출력 경로를 바꾸려면:

```bash
.venv/bin/python scripts/verify_coco_epoch.py --image-size 640 --batch-size 4 --output runs/train/coco-v9-t-640
```

결과는 기본적으로 `runs/train/coco-v9-t-1epoch/`에 저장됩니다.

- `config.yaml`: 실제 학습 설정
- `metrics/version_0/metrics.csv`: 손실 및 검증 지표. 재실행하면 버전 번호가 증가합니다.
- `checkpoints/last.ckpt`: 모델과 optimizer 상태를 포함한 Lightning 체크포인트
- `verification.json`: 전체 이미지 처리 수, 유한한 손실·가중치 검사, 가중치 변경, 1 epoch 완료 및 체크포인트 재읽기 확인

`--smoke` 옵션은 val2017에서 학습 20배치·검증 2배치만 실행하는 사전 점검입니다. 전체 COCO 1 epoch 결과와 구분하여 별도 출력 경로를 사용하세요.

```bash
.venv/bin/python scripts/verify_coco_epoch.py --smoke --output runs/train/coco-smoke
```

1 epoch 검증의 목적은 학습 파이프라인의 정상 완료 확인입니다. 무작위 초기화와 축소 입력 크기를 사용하므로 검증 AP는 충분히 학습한 모델의 정확도를 의미하지 않습니다.

## 2026-09-11 실행 결과

- RTX 2070 SUPER 8GB / Python 3.13 / PyTorch 2.6.0+cu124에서 프로세스 종료 코드 0, `status: passed`.
- 학습 118,287장 / 3,697배치, 검증 5,000장 전체 처리, 1 epoch 완료.
- 학습·검증·체크포인트 검사 약 42분 57초. 데이터 다운로드와 초기 캐시 생성 시간은 별도.
- 모든 배치 손실과 최종 가중치가 유한하며 초기 가중치와 달라진 것을 확인.
- 기록된 첫 10개/마지막 10개 학습 로그의 총 손실 평균: 16.14 → 10.07.
- epoch 평균 손실: Box 3.73196, DFL 3.92943, BCE 5.65957.
- 프로젝트 검증 지표: mAP@0.5:0.95 = 0.00144138, mAP@0.5 = 0.00357443 (0–1 척도).
- `checkpoints/last.ckpt` 재읽기 성공, epoch 인덱스 0, global step 3,697, optimizer/scheduler 상태 포함.

전체 실행 로그는 `runs/setup/full-epoch.log`, 기계 판독 가능한 결과는 `runs/train/coco-v9-t-1epoch/verification.json`에 있습니다.
