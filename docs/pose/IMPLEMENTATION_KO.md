# YOLOv9 grid-relative DFL pose baseline

공유 대화의 마지막 제안인 **bbox와 독립적인 grid 중심 좌표 분포**를 구현한다.
RTMO의 DCC/MLE 구현을 가져온 것이 아니며, 고정 signed offset bins를 사용하는 실험용 baseline이다.

## 모델과 좌표 계약

`v9-t-pose`, `v9-s-pose`, `v9-m-pose`, `v9-c-pose`는 기존 YOLOv9 backbone/FPN을 공유하고,
각 Main/AUX P3/P4/P5 head에 별도의 pose tower를 추가한다. 기존 cls/bbox parameter 이름을 유지하므로
검출 checkpoint의 일치하는 weight를 학습 초기화에 사용할 수 있다. class 수가 다르면 class 출력은 초기화된다.

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
학습 target은 `[class,xyxy,(x,y,v)*17]`, padding class는 -1이다.

- bbox/cls loss: 기존 TAL, CIoU, DFL, BCE.
- pose: 같은 TAL positive 전체에 인접 두 bin의 soft CE(DFL)를 적용한다.
- `v=1`(가려졌지만 위치 annotation 있음), `v=2` 모두 좌표를 지도한다.
- confidence BCE target은 `v>0`이다. 따라서 출력은 **annotation 가능한 관절의 confidence**이며
  visible/occluded 이진 분류나 보정된 localization 확률이라고 해석하지 않는다.
- 범위 밖 좌표는 끝 bin으로 clamp하지 않고 해당 좌표 loss에서 제외한다.
  `Pose/OutOfRangeCoordinates`, `Pose/LabeledCoordinates`, `Pose/OutOfRangeFraction`으로 범위를 관찰한다.
  Lightning의 epoch 값은 batch 가중 평균이며 count 항목도 epoch 총합은 아니다.
- person이 있지만 keypoint가 없는 annotation은 bbox supervision을 유지한다. 빈 이미지도 포함한다.
- crowd는 학습 target에서 제외한다. validation GT의 crowd/visibility/area는 공식 JSON 값을 유지한다.
- 첫 baseline augmentation은 letterbox와 좌우 관절 swap을 포함한 horizontal flip이다.
  기존 bbox 전용 Mosaic/YOLOv9 recipe를 pose에 잘못 적용하지 않도록 다른 recipe는 명시적으로 거부한다.
  이 설정은 upstream 검출 학습 recipe 재현을 주장하지 않는다.

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
TFLite runtime은 optional `litert_torch` 의존성이 필요하다. 이번 환경에서 설치되지 않아 실행 검증하지 않았다.

## 참고

- [설계의 출발점인 공유 대화](https://chatgpt.com/share/6aad60ba-8690-83e8-8279-c2ea3652fd78)
- [RTMO CVPR 2024 원문](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_RTMO_Towards_High-Performance_One-Stage_Real-Time_Multi-Person_Pose_Estimation_CVPR_2024_paper.html):
  dual 1D coordinate classification, DCC와 전용 loss를 제안한다. 여기의 fixed grid-relative DFL은 별도 baseline이다.

새 구현은 기존 저장소 코드 위에 작성했으며 외부 pose 구현/weight를 복사하지 않았다.
배포 시 기존 저장소 LICENSE와 데이터/weight 이용 조건을 별도로 따른다.
