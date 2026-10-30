# YOLOv9 grid-relative DFL pose baseline

공유 대화의 마지막 제안인 **bbox와 독립적인 grid 중심 좌표 분포**를 구현한다.
RTMO의 DCC/MLE 구현을 가져온 것이 아니며, 고정 signed offset bins를 사용하는 실험용 baseline이다.

## 모델과 좌표 계약

`v9-t-pose`, `v9-s-pose`, `v9-m-pose`, `v9-c-pose`는 기존 YOLOv9 backbone/FPN을 공유하고,
각 Main/AUX P3/P4/P5 head에 별도의 pose tower를 추가한다. 기존 cls/bbox parameter 이름을 유지하므로
검출 checkpoint의 일치하는 weight를 학습 초기화에 사용할 수 있다. class 수가 다르면 class 출력은 초기화된다.

SR 모델은 `v9-sr-t-pose`, `v9-sr-s-pose`, `v9-sr-m-pose`, `v9-sr-c-pose`로 선택한다.
동일 크기 SR 검출 모델의 backbone/FPN 및 AUX 구조를 상속하고 Main/AUX 모두 pose head로 바꾼다.
T/S/M의 AConv2, C의 ADown2 고정 smoothing과 SiLU를 유지하며, M의 확장 채널도 AUX까지 유지한다.
아래 학습·검증·추론·export 명령의 `model`을 해당 SR pose 이름으로 바꾸면 같은 경로를 사용한다.
학습 초기화에는 대응하는 SR 검출 checkpoint를 사용할 수 있고, 추론에는 해당 SR pose 모델로 학습한
`pose_config` 포함 checkpoint를 사용한다. SR pose도 QAT와 `model.nms_free=true`는 지원하지 않는다.

기본값:

- 사람 1 class, COCO 17 keypoints.
- bbox: 기존 L/T/R/B DFL, `reg_max=16`.
- pose: 관절별 x/y 각각 64 bins, stride 단위 `[-32,32]`.
- 관절별 confidence logit 1개. bbox와 pose의 최종 logits는 분리한다.

`b_i = -R + 2R*i/(M-1)`, `offset = sum(softmax(logits)_i*b_i)`이며
`keypoint_xy = grid_center_xy + stride*offset_xy`이다. bbox 좌표/크기는 이 계산에 들어가지 않는다.
특정 FPN level이 특정 사람 크기를 담당하도록 강제하지 않으며, 기존 TAL assignment를 따른다.

per-scale raw output은 다음 다섯 tensor다.

```
cls        [B,C,H,W]
bbox logits[B,reg_max,4,H,W]
bbox LTRB  [B,4,H,W]
pose logits[B,K*2*M,H,W]   # keypoint, x/y, bin 순서
confidence [B,K,H,W]
```

converter의 반환값은 `(cls, bbox_logits, boxes, pose_logits, confidence_logits, keypoints)`이다.
마지막 세 tensor는 각각 `[B,N,2K,M]`, `[B,N,K]`, `[B,N,K,3]`이다.
학습의 bbox logits에는 기존 rank-5 tensor가 남아 있다. rank ≤ 4 보장은 export graph에 적용한다.

postprocess는 class별 bbox NMS가 고른 **동일 grid의 전체 skeleton**을 함께 선택한다.
이미지별 결과 행은 `[class_id,x1,y1,x2,y2,score,k1_x,k1_y,k1_score,...]`, 즉 COCO 기준 `[instances,57]`이다.
좌표는 inference에서 원본 이미지 pixel로 복원하고, validation에서는 입력 좌표를 evaluator가 원본으로 복원한다.
정수 resize에서 발생하는 x/y gain 차이까지 반영한다. pose 좌표를 이미지 경계로 강제 clip하지 않는다.

## 학습

COCO `person_keypoints_train2017.json` / `person_keypoints_val2017.json`과 기존 이미지를 사용한다.
COCO 경로에는 cache나 subset 파일을 쓰지 않는다. class index는 person=0이며,
학습 target은 `[class,xyxy,(x,y,v)*17,area,gain_x,gain_y]` (59열), padding class는 -1이다.
`area`는 COCO annotation의 segmentation 면적을 기하 변환한 값이다. 유효 bbox의 annotation에는
양의 유한 `area`가 반드시 있어야 한다. `gain_x/gain_y`는 마지막 resize의 정확한 배율이다.
loss를 직접 호출하는 기존 56열 target은 호환을 위해 GT bbox 면적을 사용하지만, 이는 공식 annotation
area를 쓰는 경로와 다르다. COCO loader는 항상 59열을 전달한다. 예측·추론·export 출력 형식은 그대로다.

- bbox/cls loss: 기존 TAL, CIoU, DFL, BCE.
- pose: 분포의 기댓값을 `grid_center + stride * offset` 픽셀 좌표로 decode하고 **1-OKS**를 적용한다.
  같은 stride offset 오차도 실제 픽셀 크기, 관절별 sigma, annotation area에 따라 다른 페널티를 받는다.
  DFL은 보조 loss로 유지한다. 큰 오차에서 OKS gradient가 포화될 때도 DFL이 분포를 지도한다.
- 각 positive 안에서 labeled 관절의 OKS loss를 평균한다. DFL은 해당 positive의 지도 가능한 x/y 좌표를
  평균하며, visibility BCE는 전체 관절을 평균한다. 세 pose loss 모두 detached TAL target score 합을
  positive 가중치로 사용하고, `max(유효 positive의 가중치 합, 1)`로 정규화한다.
  좌표가 없는 사람은 좌표 loss의 분모에서도 제외하지만 visibility BCE에는 포함한다.
- `v=1`(가려졌지만 위치 annotation 있음), `v=2` 모두 좌표를 지도한다.
- confidence BCE target은 `v>0`이다. 따라서 출력은 **annotation 가능한 관절의 confidence**이며
  visible/occluded 이진 분류나 보정된 localization 확률이라고 해석하지 않는다.
- 범위 밖 좌표는 끝 bin으로 clamp하지 않고 해당 좌표의 **DFL에서만** 제외한다. OKS에는 labeled 관절을
  그대로 포함한다. 큰 범위 초과가 잦다면 bins/range와 학습 해상도를 확인해야 한다.
  `Pose/OutOfRangeCoordinates`, `Pose/LabeledCoordinates`, `Pose/OutOfRangeFraction`으로 범위를 관찰한다.
  Lightning의 epoch 값은 batch 가중 평균이며 count 항목도 epoch 총합은 아니다.
- person이 있지만 keypoint가 없는 annotation은 bbox supervision을 유지한다. 빈 이미지도 포함한다.
- crowd는 학습 target에서 제외한다. validation GT의 crowd/visibility/area는 공식 JSON 값을 유지한다.
- 학습 증강은 좌우 관절 swap을 포함한 flip, 색상 변화, 회전·이동·등방 확대/축소를 지원한다.
  bbox는 네 꼭짓점을 변환하고 clip한다. 폭/높이 1픽셀 이하 또는 변환 bbox 면적의 20% 미만만 남은
  사람은 제거한다. 화면 밖으로 나간 관절은 `(0,0,0)`으로 바꾸며, 남은 관절의 `v=1/2`는 보존한다.
  affine은 resize 전에 수행한다. area는 affine의 면적 배율과 정확한 resize 배율 곱으로 변환한다.
  crop 시에는 전체 인스턴스의 변환 면적을 유지한다. segmentation mask를 직접 자르지 않으므로
  잘린 segmentation의 실제 면적과는 다를 수 있는 학습 증강의 근사다.
  기존 bbox 전용 Mosaic/YOLOv9 recipe 및 validation 증강은 명시적으로 거부한다.

OKS 식은 [공식 COCO 구현](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)의
`e_i = squared_distance_i / (2 * (2 * sigma_i)^2 * area)` 및 `1 - mean(exp(-e_i))`를 따른다.
stretch와 정수 letterbox에서도 원본 좌표계의 평가와 맞도록, 좌표 오차를 x/y gain으로 나누고
area를 gain 곱으로 나눈 뒤 계산한다. 공식 COCO 17개 sigma를 기본 사용한다.
다른 관절 배치는 `task.loss.oks_sigmas`에 관절별 값을 지정해야 한다.

기본 학습 recipe는 다음과 같다. 계수만으로 실제 gradient 비중을 비교할 수 없으며,
이 값은 검증된 AP 최적값이 아닌 비교 학습의 출발점이다.

| 항목 | 기본값 |
| --- | --- |
| PoseOKSLoss / PoseDFLoss / PoseVisibilityLoss | 7.5 / 1.0 / 1.0 |
| BoxLoss / DFLoss / BCELoss | 7.5 / 1.5 / 0.5 |
| fliplr / degrees / translate / scale | 0.5 / 10 / 0.1 / 0.2 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.4 / 0.4 |

`degrees`는 ±도, `translate`는 각 축 이미지 크기의 ±비율, `scale`은 1±비율이다.
HSV 옵션은 hue의 ±이동, saturation/brightness의 1±배율이다. 0은 해당 증강을 끈다.
`task.loss.objective.PoseOKSLoss=0`으로 OKS만 끄는 ablation도 가능하다.
이 경우 custom layout의 sigma는 필요 없다. 이전 모델 weight는 동일한 구조로 로드할 수 있지만,
새 기본 recipe로 이어 학습한 결과는 이전 loss/증강 조건을 유지한 실험과 구분해야 한다.
로그의 `Loss/PoseOKSLoss`를 포함한 loss 값에는 해당 계수와 auxiliary head 가중치가 반영된다.

```bash
python -m yolo.lazy task=train-pose model=v9-t-pose dataset=coco-pose \
  weight=false name=pose-train task.epoch=100 use_wandb=false
```

기존 검출 weight로 초기화할 때는 `weight=/absolute/path/to/v9-t.pt`처럼 지정한다.
`weight=true` 자동 pose pretrained 다운로드는 지원하지 않는다. 새 pose head는 별도 학습이 필요하다.
`best.pt`는 pose AP로 선택되고 pose bins/range metadata를 함께 저장한다.
전체 `.ckpt`에는 optimizer/scheduler/EMA/학습 진행 상태도 저장된다.
기존 named-run resume 규칙과 명시적 `.ckpt` resume 경로를 그대로 사용한다.
범위만 달라 tensor shape가 같은 checkpoint도 metadata 불일치면 거부한다.

## 1/20 smoke test

`dataset=coco-pose-smoke`는 split 전체 image ID를 정렬한 뒤 seed=10으로 5%를 샘플링한다.
train 5,915/118,287장, validation 250/5,000장이다. person이 없는 이미지도 포함한다.
validation subset을 학습이나 hyperparameter 선택에 사용하지 않는다.

```bash
python scripts/smoke_pose.py --output runs/pose-smoke --epochs 1 \
  --size 128 --batch-size 16 --workers 4 --threads 4
```

CPU float32, random initialization, EMA off로 한 epoch 전체를 실행한다.
`config.yaml`, annotation SHA256와 전체 subset image ID의 `subset_manifest.json`,
`report.json`, `pose-smoke.pt`, full checkpoint 및 best weights가 저장된다.
유한 loss, optimizer step, bbox/pose weight 변화와 실제 COCO keypoint 평가 실행을 검증한다.
합성 테스트는 별도로 pose branch의 실제 gradient 전달과 좌표 의미를 검증한다.
이 짧은 학습은 **실행 검증**이며 수렴, 실사용 pose 품질 또는 논문 AP 재현이 아니다.

## 추론과 검증

```bash
python -m yolo.lazy task=inference model=v9-t-pose dataset=coco-pose \
  weight=runs/pose-smoke/pose-smoke.pt task.data.source=demo/images/inference/image.png \
  image_size='[128,128]' accelerator=cpu device=1 precision=32-true \
  use_wandb=false name=pose-demo

python -m yolo.lazy task=validation-pose model=v9-t-pose dataset=coco-pose-smoke \
  weight=runs/pose-smoke/pose-smoke.pt image_size='[128,128]' \
  accelerator=cpu device=1 precision=32-true use_wandb=false
```

추론은 `runs/inference/<name>/frame########.jpg`와 같은 이름의 `.json`을 저장한다.
JSON은 원본 `image_size`, 인스턴스별 `class_id`, `bbox_xyxy`, `score`, `keypoints[K][3]`를 포함한다.
`task.save_predict=false`로 그림, `task.save_json=false`로 JSON 저장을 각각 끌 수 있다.
`task.keypoint_confidence`는 그리기 threshold다. numerical 결과 자체에는 threshold를 적용하지 않는다.
Python에서는 `PostProcess` 결과 또는 `InferenceModel.last_predictions`로 tensor를 받을 수 있다.
단일 이미지·디렉터리·video 파일 입력은 기존 stream loader를 사용한다.
실시간 카메라/RTMP의 기존 표시 경로는 이번 구현·검증 범위에 포함하지 않았다.
기존 `task.fast_inference` backend 선택은 CLI에 연결되어 있지 않으며 pose 가속 backend를 추가하지 않았다.

검증은 **pycocotools COCOeval keypoints/OKS**와 공식 17-joint sigmas, maxDets=20을 사용한다.
`map`은 pose AP이며 `mar_20`은 AR20이다. bbox AP와 혼동하지 않는다.
일반적인 custom K의 head/decoder는 가능하지만 제공하는 dataset/evaluator는 COCO 17 관절 전용이다.

## Export

```bash
python -m yolo.lazy task=export task.format=onnx model=v9-t-pose dataset=coco-pose \
  weight=runs/pose-smoke/pose-smoke.pt image_size='[128,128]' name=pose-export
```

기존 검출 exporter의 두 계약을 확장한다. 두 경우 모두 단일 pre-NMS tensor이며 Main만 실행한다.

| 경로 | 출력 순서 | K17, M64, C1 |
|---|---|---|
| ONNX 기본 probability 모드 | LTRB softmax `4R`, cls sigmoid `C`, pose softmax `2KM`, confidence sigmoid `K` | `[B,N,2258]` |
| decoded wrapper / TFLite | pixel xyxy `4`, class scores `C`, `(x,y,confidence)*K` | `[B,N,56]` |

ONNX의 pose 분포 flatten 순서는 keypoint, x/y, bin이다. 후처리 소비자는 동일한 bins/range/stride/grid를
사용해야 하며 shape만으로 offset 범위를 추측하면 안 된다. 설정 metadata는 checkpoint와 실행 config에 있다.
ONNX shape inference로 **모든 내부 tensor rank ≤ 4**를 검사한다.
640 입력에서 관절 logits만 이미지당 Main 또는 AUX 출력 전체 기준 약 73 MB(float32)이므로 큰 batch의 메모리 사용은 별도 확인해야 한다.
`model.pose.pose_bins`와 `model.pose.pose_range`로 bins/range를 조정할 수 있다.
실제 ONNX Runtime에서 probability/decoded, 고정/동적 batch 출력을 비교한다.
TFLite는 별도 Python 3.12 환경에 `pip install -e '.[export-tflite]'`로 설치한다.
이 extra는 검증용 LiteRT Torch/PyTorch/TorchAO 버전을 고정한다. CPU 설치와 테스트 명령은
[export 안내](../3_custom/3_task.rst)를 참고한다. TFLite CI는 backend import를 필수로 검사하고,
`v9-t-pose` 및 `v9-sr-{t,s,m,c}-pose`의 COCO 17관절 출력, PyTorch와의 수치 일치 및
모든 내부 tensor rank ≤ 4를 검사한다. 이 검증은 CPU float32 변환·실행 검사이며 모바일 NPU
성능이나 학습된 pose 정확도를 측정하지 않는다.

## 참고

- [설계의 출발점인 공유 대화](https://chatgpt.com/share/6aad60ba-8690-83e8-8279-c2ea3652fd78)
- [RTMO CVPR 2024 원문](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_RTMO_Towards_High-Performance_One-Stage_Real-Time_Multi-Person_Pose_Estimation_CVPR_2024_paper.html):
  dual 1D coordinate classification, DCC와 전용 loss를 제안한다. 여기의 fixed grid-relative DFL은 별도 baseline이다.

새 구현은 기존 저장소 코드 위에 작성했으며 외부 pose 구현/weight를 복사하지 않았다.
배포 시 기존 저장소 LICENSE와 데이터/weight 이용 조건을 별도로 따른다.
