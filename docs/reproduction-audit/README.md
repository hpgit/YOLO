# YOLOv9 재현성 코드 비교 (2026-09-12)

비교 기준: 현재 저장소 `82b604ae5f0a01209add9df71412412935618b71`, 원본 WongKinYiu/yolov9 `5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff`.
원본은 `/tmp/yolov9-reference-20260912`에 읽기 비교용으로 clone했다. 아래 내용과 probe 결과는 수정 전 스냅샷이다.

후속 수정: [공식 COCO 평가(1번)](../coco-evaluation.md), [stride 탐색과 누적 학습 update(2·3번)](../training-reproduction.md), [학습 증강(4번, 마지막 mosaic 종료 제외)](../augmentation-reproduction.md).

우선순위는 COCO 논문 수치 재현에서의 진단 중요도이며, AP 하락 폭을 측정한 순위가 아니다. 전체 학습 및 같은 체크포인트의 양쪽 COCO AP 비교는 수행하지 않았다.

## 조건부 최우선: 표준 YOLO detection TXT 파싱

`yolo/tools/data_loader.py:139`는 `[class,xc,yc,w,h]`를 polygon 점으로 해석한다. 실제 호출에서 `[0,.5,.5,.2,.2]`가 `[0,.2,.2,.5,.5]`로 나왔다. 올바른 xyxy는 `[0,.4,.4,.6,.6]`이다. 공식 COCO JSON 경로에는 이 TXT 문제가 직접 적용되지 않는다. 데이터 루트에 split TXT 목록이 있으면 JSON보다 TXT 경로가 우선되는 분기도 확인해야 한다.

## 1. 공식 COCO 평가와 다른 GT 및 좌표계

원본 `val_dual.py:289`는 원본 좌표로 복원한 예측 JSON을 `instances_val2017.json`에 대해 COCOeval한다. 현재 `solver.py:51`은 resize/pad 좌표의 예측과 로더 GT를 TorchMetrics에 전달한다. 로더는 crowd를 제외하고, `to_metrics_format`은 boxes/labels만 전달하여 공식 area와 crowd 정보가 보존되지 않는다. 특히 AP_small/medium/large를 직접 비교하면 안 된다. 같은 COCO 알고리즘 계열 backend를 쓰는 것만으로 입력 GT 차이가 없어지지 않는다.

실제 val2017 5,000장 주석에서 crowd 446개가 확인됐다. noncrowd 36,335개의 polygon 외접 박스와 공식 bbox는 최대 약 1.14e-13 pixel 차이로 사실상 같았다. noncrowd GT가 100개를 넘는 이미지가 없어 collate의 GT 100개 제한은 이 검증셋을 자르지 않는다. 두 항목을 COCO val 성능 격차의 주원인으로 단정하지 않는다.

대응: 동일 예측을 원본 좌표로 복원하고 공식 annotation JSON으로 양쪽을 평가한다. category mapping, image IDs, crowd, area, maxDets를 동일하게 한다.

## 2. T/S/M stride 자동 탐색의 BatchNorm 상태 변경

현재 T/S/M 설정에는 strides가 없어 `Vec2Box.create_auto_anchor`가 호출된다. 이 함수는 train mode를 보존한 채 zero dummy forward를 수행한다. 체크포인트는 그 전에 로드된다. C에는 `[8,16,32]`가 지정되어 이 경로를 피한다.

CPU v9-t 무작위 초기화 모델과 64x64 dummy에서 873개 BN buffer 중 582개가 변경됐다. 첫 running_var는 1에서 약 .97, num_batches_tracked는 0에서 1로 바뀌었다. 이 검사는 상태 변경을 입증하며 pretrained COCO AP 하락 폭을 측정한 것은 아니다.

원본의 모델 생성 시 stride probe와 달리, 여기서는 로드된 상태를 이후에 바꾸는 실행 순서가 문제다. 대응: stride를 명시하거나 평가 모드에서 probe 후 원래 mode를 복원한다. no_grad만으로는 BN 통계 변경을 막지 못한다.

## 3. Gradient accumulation 및 DDP loss scale

원본 `train_dual.py:318`은 batch 크기가 반영된 loss로 backward하고 accumulation 동안 gradient를 합산한다. DDP에서는 WORLD_SIZE도 곱한다. 현재 `solver.py:108`도 loss에 batch 크기를 곱하지만 설치된 Lightning 2.6.6은 automatic optimization에서 accumulate_grad_batches로 나눈다.

동일 scalar loss의 4회 backward 검사에서 원본식 합산 gradient는 64, Lightning normalize=4 경로는 16이었다. 따라서 batch 16/nominal 64의 accumulation=4 구간은 clipping 전 gradient scale이 다르다. AP 또는 실제 전체 모델 update가 정확히 1/4이라는 주장은 아니다. 현재 코드에는 원본의 DDP WORLD_SIZE 보정도 없다.

`GradientAccumulation.on_train_epoch_start`는 microbatch 카운터를 optimizer step인 trainer.global_step으로 재설정한다. 100 batch/epoch 모의 실행에서 3 epoch warmup이 지난 다음에도 accumulation factor가 epoch 경계에서 4에서 3으로 돌아가는 구간이 나타났다. EMA의 배치 modulo 기준과 optimizer의 실제 step도 마지막 불완전 accumulation 및 동적 accumulation에서 어긋날 수 있다.

대응: 원본과 동일한 optimizer step 단위로 gradient 크기, 누적 횟수, clipping, EMA update를 비교한다. 단순 LR 배수 조정만으로 동일성을 가정하지 않는다.

## 4. 학습 증강 레시피와 구현

원본 hyp.scratch-high: HSV .015/.7/.4, translate .1, scale .9, fliplr .5, mosaic 1, mixup .15, copy_paste .3. Copy-paste의 실제 적용은 segmentation 데이터 가용성에 의존한다. README 학습 명령에는 마지막 15 epoch mosaic 종료가 있다.

현재 기본 train.yaml은 Mosaic 1, RandomCrop 1이며 MixUp/HorizontalFlip은 주석 처리되어 있다. HSV와 원본 random_perspective 레시피 및 마지막 mosaic 종료 경로가 없다. 현재 Mosaic는 원본 해상도 이미지를 고정 중심에 붙이며, RandomCrop은 가로와 세로를 절반으로 자른다. 원본은 먼저 이미지 크기를 조정하고 무작위 mosaic 중심 및 geometric transform을 적용한다.

대응: 원본 학습 재현에서는 임의의 AdamW/Cosine 변경보다 먼저 원본 증강 레시피와 실제 변환 분포를 맞춘다. 원본 기본 optimizer는 SGD, scheduler는 linear다.

## 5. EMA 체크포인트 선택과 재개

현재 학습 validation은 pl_module.ema를 사용하지만 `YOLO.save_load_weights`는 Lightning state_dict의 model.model.*를 선택한다. ema.model.*는 선택하지 않는다. 일반 weight=2, EMA weight=7의 최소 checkpoint 입력에서 실제 로드 값은 2였다. 학습 중 validation AP와 저장 후 standalone validation AP가 다른 직접적인 후보이다.

원본은 EMA를 validation에 사용하고 checkpoint의 ema 및 updates를 저장/복원한다. 현재 EMA callback은 step/ema_state_dict의 callback state 저장·복원 메서드가 없으며 lazy.py는 trainer.fit(model)에 ckpt_path를 주지 않는다. weight=checkpoint는 원본의 resume와 같지 않다. DDP에서 tau를 world_size로 나누는 것도 원본과 다르다.

대응: 평가용 EMA state 선택을 명시하고 재개 시 model/optimizer/scheduler/EMA/updates를 함께 복원한다.

## 6. BoxMatcher의 학습 목표 차이

현재 get_valid_matrix는 anchor가 GT 안에 있는지 외에도 거리 <= reg_max-1.01 조건을 적용한다. ensure_one_anchor는 유효 anchor가 없는 GT에 unmasked score로 하나를 배정한다. 중복 해소도 topk mask 안에서 IoU를 비교한다. 원본 utils/tal/assigner.py에는 이 회귀 범위 제한과 fallback이 없다.

이것은 제안된 개선이 반영된 알고리즘 차이이지 원본 재현과의 동일성은 아니다. 작은/큰 객체의 positive assignment와 target score가 달라진다. 효과의 방향은 실험 전에는 알 수 없다. 대응: 같은 prediction/GT에 대한 positive mask와 target score부터 비교하고 원본 일치 모드로 ablation한다.

## 7. 셔플, 전처리, NMS

현재 train.yaml의 shuffle=True가 create_dataloader의 DataLoader 인자로 전달되지 않는다. 단일 GPU에서는 순차 sampler가 된다. 원본 train_dual.py는 shuffle=True를 전달한다. DDP의 sampler 자동 교체는 별도로 확인해야 한다.

현재 validation은 dynamic_shape=False의 square 입력과 PIL LANCZOS, 원본은 rect=True/pad=.5 및 OpenCV resize/letterbox다. 현재 dynamic_shape의 공식도 원본 batch_shapes와 달라 단순히 True로 켜는 것으로 동일해지지 않는다. 캐시 .pache는 dynamic_shape 변경을 캐시 키에 반영하지 않으므로 정렬 정보도 확인해야 한다.

현재 confidence .0001/max_bbox 1000, 원본 평가 예제 .001/기본 max_det 300이다. 양쪽 IoU .7 및 multi-label 후보 방식은 공통이다. 대응: 입력 tensor, resize/letterbox 좌표, 후보 threshold와 개수 제한까지 통일한다. 평가 전처리 차이는 pretrained 평가만 할 때 우선순위를 더 높인다.

## 8. 초기화와 실험 출발점

원본 README의 학습은 --weights ''로 scratch 시작한다. 현재 general.yaml은 weight=True가 기본이다. 현재 detection class bias는 전 scale에서 -10, 원본은 log(5/nc/(640/stride)^2)다. COCO 80클래스/stride 8,16,32에서는 약 -11.54,-10.15,-8.76이다. 정상 로드된 pretrained bias에는 초기화 차이가 덮어써지지만 scratch/새 클래스 head에는 영향을 준다.

현재 lucky_number=10 설정과 set_seed 함수가 있지만 lazy.py/setup 호출 경로에서 set_seed 호출은 발견하지 못했다. 별도 검증 scripts는 seed_everything을 호출하므로 CLI와 구분한다. 원본은 seed 설정을 실행한다.

## 우선순위를 낮춘 항목

- 현재 validation은 metric.update 후 epoch 말 compute/reset을 사용하며 label=-1 padding을 제거한다. 옛 이슈의 매 batch AP/가짜 padding 문제는 현재 코드의 주원인으로 다시 적지 않는다.
- 현재 MixUp은 라벨 좌표에 lambda를 곱하지 않는다. 다만 두 이미지의 크기 정렬 처리가 없어 다른 크기 데이터에서 단순 활성화하면 shape 오류가 날 수 있다.
- 양쪽 box/cls/DFL 가중치 7.5/.5/1.5, AUX 가중치 .25, clip_grad_norm 10은 공통이다. 원본도 norm clipping을 사용하므로 value로 바꾸라는 과거 제안을 그대로 따르지 않는다.
- 전체 학습 그래프의 파라미터 수와 보조 branch가 제거된 converted 모델의 파라미터 수를 직접 비교해 구조 오류로 판단하지 않는다.

## 검증 실행

저장소 루트에서 `.venv/bin/python docs/reproduction-audit/probe.py`.
`probe_results.json`에 CPU 미니 검사, 실제 COCO val 주석 검사, 버전을 기록했다. 학습 파이프라인 코드는 수정하지 않았고 전체 COCO AP나 500 epoch 재학습은 실행하지 않았다.

원본 근거: [train_dual.py](https://github.com/WongKinYiu/yolov9/blob/5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff/train_dual.py), [val_dual.py](https://github.com/WongKinYiu/yolov9/blob/5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff/val_dual.py), [hyp.scratch-high.yaml](https://github.com/WongKinYiu/yolov9/blob/5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff/data/hyps/hyp.scratch-high.yaml), [assigner.py](https://github.com/WongKinYiu/yolov9/blob/5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff/utils/tal/assigner.py), [torch_utils.py](https://github.com/WongKinYiu/yolov9/blob/5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff/utils/torch_utils.py).
