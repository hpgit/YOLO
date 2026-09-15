# Bounding box 전용 annotation 지원과 검증

Segmentation 없이 **bbox와 class만 있는 annotation으로 학습과 validation이 가능하다.**
기본 YOLOv9 증강 학습, 기존 증강 학습, 학습 중 validation, 독립 validation에 적용된다.
이미지 경로·크기와 class 정보 등 데이터셋 형식에 필요한 메타데이터는 있어야 한다.

## 지원 형식

### YOLO TXT

`labels/<split>/<image stem>.txt`에 정규화한 `class cx cy width height`를 기록한다.
예를 들어 너비 64, 높이 32인 이미지에서 다음 라벨은 pixel xyxy `[8, 8, 40, 24]`다.

```text
0 0.375 0.5 0.5 0.5
```

`images/<split>` 디렉터리 또는 데이터셋 루트의 `<split>.txt` 이미지 목록을 사용한다.
같은 라벨 파일에 `class x1 y1 x2 y2 x3 y3 ...` polygon 행을 섞을 수도 있다.
빈 행은 무시하며 빈 파일이나 라벨 파일이 없는 이미지는 배경 이미지로 유지한다.
TXT validation은 기본 `evaluator=auto`에서 TorchMetrics bbox 평가를 사용한다.

### COCO JSON

`annotations/instances_<split>.json`의 최소 예시다. `categories`의 ID를 정렬한 순서가
모델의 0부터 시작하는 class index이며, category 수는 `dataset.class_num`과 일치해야 한다.

```json
{
  "images": [{"id": 1, "file_name": "sample.png", "width": 64, "height": 32}],
  "categories": [{"id": 7, "name": "object"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [8, 8, 32, 16]}]
}
```

COCO bbox는 정규화하지 않은 pixel `x_left, y_top, width, height`다.
`segmentation`이 없거나 빈 배열, `null`, RLE여도 bbox를 읽는다.
polygon이 함께 있어도 detection 라벨은 JSON의 bbox를 기준으로 한다.
YOLOv9 학습에서는 실제 polygon이 있는 객체에만 Copy-Paste를 적용한다.
bbox 전용 데이터에서도 나머지 증강은 적용되며 별도의 증강 설정 변경은 필요 없다.

COCO JSON validation은 원본 JSON을 COCOeval에 전달한다. 누락된 `iscrowd`는 `0`,
누락된 `area`는 원본 bbox의 `width × height`로 메모리에서만 보완한다.
이미 제공된 `area`와 `iscrowd`는 그대로 보존하고 원본 파일은 수정하지 않는다.
`area`를 생략한 데이터의 small/medium/large 구분은 bbox 면적 기준이다.

## 로딩 검증 규칙

| 입력 | 처리 |
| --- | --- |
| TXT class | 유한한 0 이상 정수, `class_num` 미만이어야 한다. |
| TXT bbox | 정확히 5열이며 모든 xywh가 유한한 `[0, 1]` 값이고 w/h가 양수여야 한다. |
| TXT polygon | 최소 3개 xy 점, 유한한 `[0, 1]` 좌표, 0이 아닌 polygon 면적이어야 한다. |
| 잘못된 TXT 행 | 파일명과 줄 번호를 포함한 `ValueError`로 중단한다. clip으로 잘못된 원본 좌표를 숨기지 않는다. |
| TXT에서 유도한 bbox가 이미지 경계를 걸침 | YOLOv9는 기하 증강 단계에서 clip하고, 기존 로더는 전처리 전에 clip한다. |
| COCO bbox | 4개 유한한 xywh 값과 양수 w/h를 확인하고 이미지 경계로 clip한다. |
| 잘못된 COCO bbox 또는 이미지와 겹치지 않는 bbox | 해당 객체를 경고와 함께 로더에서 제외한다. 이미지 자체는 유지한다. |
| COCO image 크기 | 양수이고 유한해야 한다. |
| 알 수 없는 category 또는 유효하지 않은 class | 오류로 중단한다. |
| COCO crowd | 로더 target에서는 제외하고, JSON 평가에서는 원본 crowd 의미를 보존한다. |

COCOeval은 로더에서 제외한 객체도 **원본 JSON 기준**으로 평가한다. 원본의 유효하지 않은
bbox를 정제하는 도구는 아니며, `area`를 유도할 수 없는 bbox는 명시적으로 오류를 낸다.

기존 로더의 `.pache`에는 parser 버전·class 수·dynamic shape 설정을 기록한다.
이전 list 형식 캐시와 설정이 다른 캐시는 자동 재생성하므로 기존의 잘못된 bbox 해석이 남지 않는다.
이 변경은 annotation 파일의 수정 여부까지 추적하지 않는다. 이후 라벨이나 split을 편집하면
해당 `<split>.pache`를 삭제해 다시 읽어야 한다. 기본 YOLOv9 학습 로더는 캐시 없이 원본을 읽는다.

## 수정한 문제와 확인 범위

- 기존 로더가 TXT 5열 xywh를 두 polygon 점으로 해석해 잘못된 xyxy를 만드는 문제를 수정했다.
- clip 이후 검사로 좌표 오류를 숨기고, 0 면적 bbox나 잘못된 class를 허용하던 경로를 수정했다.
- COCO의 빈 segmentation이 bbox를 가리거나, segmentation의 외곽이 bbox를 덮어쓰던 문제를 수정했다.
- `iscrowd` 누락 시 로더 오류, `area`/`iscrowd` 누락 시 COCOeval 오류를 수정했다.
- 두 로더가 TXT·COCO bbox 검증 함수를 공유한다.

`tests/test_tools/test_bbox_only_dataset.py`는 TXT 디렉터리·split 목록·최소 COCO JSON,
좌표 변환, 잘못된 라벨, 작은 양수 bbox, 혼합 polygon, 빈 이미지, 캐시 재생성을 검사한다.
TXT/JSON × 기본 YOLOv9/기존 로더 4개 조합으로 실제 CPU YOLOv9-t 학습과 독립 validation을
실행하며 양수 box/DFL loss, 유한한 지표, 실제 parameter 변경을 확인한다.
정답과 일치하는 합성 예측의 AP/AR=1 검사도 포함한다. 이는 실행 경로와 좌표의 검증이며
학습 성능 측정이나 COCO benchmark 재현 결과가 아니다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  tests/test_tools/test_bbox_only_dataset.py \
  tests/test_tools/test_data_loader.py \
  tests/test_tools/test_yolov9_dataset.py \
  tests/test_tools/test_yolov9_augmentation.py \
  tests/test_tools/test_augmentation_recipe_wiring.py \
  tests/test_tools/test_data_augmentation.py \
  tests/test_tools/test_coco_validation_integration.py \
  tests/test_tools/test_validation_performance_equivalence.py \
  tests/test_tools/test_loss_performance_equivalence.py \
  tests/test_utils/test_coco_eval.py -q
```
