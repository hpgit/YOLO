# YOLOv9 학습 증강 재현

기본 학습 설정은 원본 WongKinYiu/yolov9의
[`hyp.scratch-high.yaml`](https://github.com/WongKinYiu/yolov9/blob/5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff/data/hyps/hyp.scratch-high.yaml)을
따르는 `data_augment.YOLOv9` recipe를 사용한다. 요청에 따라 **학습 마지막 구간의 mosaic 종료는 적용하지 않는다.**
설정된 mosaic 확률은 마지막 epoch까지 유지된다. 원본 소스를 프로젝트에 복사하지 않고 동작을 독립적으로 구현했다.
현재 기본 학습 설정에는 요청에 따라 약한 motion blur도 추가되어 있다. 원본 증강과의
parity 비교에서는 아래 motion blur를 비활성화한다.

| 항목 | 기본값 |
| --- | --- |
| HSV hue / saturation / value | 0.015 / 0.7 / 0.4 |
| rotation / translate / scale | 0 / 0.1 / 0.9 |
| shear / perspective | 0 / 0 |
| 상하 / 좌우 반전 확률 | 0 / 0.5 |
| mosaic / MixUp | 1.0 / 0.15 |
| segment copy-paste 비율 | 0.3 |
| MixUp 분포 | Beta(32, 32) |
| Motion blur 확률 / 커널 / 혼합 강도 | 0.1 / 3×3 / 0.5 |

Copy-paste의 0.3은 모든 이미지에 대한 30% gate가 아니라, 뒤집은 박스와 기존 박스의 IOA가
모두 0.3 미만인 후보 중 `round(0.3 × 후보 수)`를 선택하는 비율이다. 실제 polygon이 있는 객체만
대상이며 bbox를 사각형 마스크로 꾸며 copy-paste하지 않는다.

## 실행 순서

1. OpenCV로 이미지를 읽고 긴 변을 학습 크기에 맞춰 INTER_LINEAR resize한다.
2. Mosaic 분기에서는 임의 중심과 섞인 네 이미지로 canvas를 구성하고, segment copy-paste 및
   affine/perspective 변환을 적용해 학습 크기로 자른다. 이전의 고정 중심 Mosaic + 절반 RandomCrop은 사용하지 않는다.
3. MixUp 분기에서는 별도로 만든 두 번째 mosaic과 Beta(32,32) 비율로 섞고 라벨을 합친다.
4. Mosaic를 쓰지 않는 분기에서는 letterbox 후 기하 변환을 적용한다.
5. 선택적 Albumentations 이후 HSV와 상하/좌우 반전을 적용한다.
6. 확률 0.1로 motion blur를 적용한다. 수평·수직·두 대각선 중 한 방향의 중심 대칭
   3픽셀 커널로 흐리게 한 결과를 원본과 50:50으로 혼합한다. 이미지 크기와 라벨은 유지한다.

기하 변환에는 segment resampling, 경계 clipping, 변환 후 작은 박스·낮은 잔존 면적·과도한 종횡비
제거가 포함된다. RGB float tensor와 pixel xyxy 학습 라벨로 반환하며, 학습용 변환 결과에
validation용 PIL PadAndResize를 다시 적용하지 않는다.

## 데이터와 설정

기본 recipe는 polygon을 보존하는 별도 학습 dataset을 사용한다. COCO JSON, 명시적 split TXT, 또는 `images/<phase>`와 `labels/<phase>` 디렉터리를
읽으며 기존 bbox 전용 `.pache`를 재사용하지 않는다. TXT의 5-column detection xywh와 polygon
형식을 구분한다. 기존 validation/inference의 dataset 및 전처리는 유지한다.
COCO multipart polygon은 가까운 점 사이를 왕복하는 연결로 합쳐 모든 구성 요소를 보존한다.
RLE 또는 bbox-only 주석은 실제 polygon이 없으므로 copy-paste 대상으로 만들지 않는다.
Mosaic/MixUp으로 객체가 늘어날 수 있으므로 이 recipe의 batch collate는 100개를 넘는 라벨도 보존한다.

`task.data.data_augment.YOLOv9` 아래 설정값을 변경할 수 있다. 이는 완전한 파이프라인이므로 같은
설정에 이전 Mosaic/RandomCrop 등을 덧붙이면 오류를 낸다. 정사각형 입력을 기준으로 하며
지원하지 않는 rectangular/dynamic shape 조합은 명시적으로 거부한다.

원본의 Albumentations는 코드상 선택 기능이지만 requirements에도 포함된다. 기본 recipe는
`albumentations: true`이며 upstream API와 호환되는 `albumentations==1.3.1`을 고정했다.
실제 패키지의 Blur/MedianBlur/ToGray/CLAHE를 각각 확률 0.01로 적용한다. 원본과 같은
BGR 입력 조건과 확률 0인 변환들의 난수 소비도 유지한다. 패키지가 없으면 조용히 생략하지 않고
설치 오류를 알린다. 선택적으로 `albumentations: false`를 설정할 수 있다.

Motion blur는 Albumentations 설정과 독립적으로 동작하며, 최종 학습 해상도에서 한 번 적용한다.
`task.data.data_augment.YOLOv9.motion_blur=0`으로 끌 수 있다.
`motion_blur_kernel_size`는 3 이상의 홀수, `motion_blur_strength`는 0~1이다.
확률이나 강도가 0이면 이미지와 난수 상태를 그대로 유지한다. 직접 `YOLOv9Augmentation`을
생성할 때는 기존 원본 비교를 위해 motion blur가 기본적으로 꺼져 있고, `train.yaml`에서 켠다.
기존 legacy 증강 설정에서는 `data_augment: {MotionBlur: 0.1}`로 같은 약한 기본값을 사용할 수 있다.
legacy 경로에서는 다른 변환처럼 최종 PadAndResize 전에 적용된다.

## 검증

2026-09-20 motion blur 검증: 방향별 픽셀 변화·중심 대칭·라벨 보존·비활성화 시 난수 보존과
legacy/YOLOv9의 worker 0/2 재현성을 확인했다. 기본 확률 0.1로 COCO val2017 일부를 사용한
v9-t / 128px / batch 4 / CUDA AMP 20배치 진단에서 80장 중 8장에 적용되었다.
loss와 가중치는 유한했고 가중치 갱신도 확인했다. optimizer 시도 10회 중 성공 3회,
AMP skip 7회였다. 결과는 `runs/train/motion-blur-smoke-20260920/verification.json`에 있다.
이는 실행 경로 진단이며 AP 개선이나 장기 학습 성능 검증은 아니다.

원본 비교 스크립트는 별도로 준비된 upstream checkout의 고정 commit에서 함수들을 읽는다.
원본 코드를 저장소나 테스트에 포함하지 않는다. 동일한 입력 및 Python/NumPy 난수 상태에서
픽셀·박스·난수 소비를 비교하며 결과 JSON에 원본 commit과 함수 소스 해시를 기록한다.

```bash
.venv/bin/python scripts/verify_yolov9_augmentation.py \
  --reference /tmp/yolov9-reference-20260912 --seeds 32 \
  --output docs/reproduction-audit/augmentation-parity.json
```

결과: **39개 조건, 총 1,124회 비교 통과**. 픽셀과 Python/NumPy RNG 상태가 정확히 같고,
정규화 박스 좌표의 최대 절대 오차는 `8.94e-8`이었다. Albumentations 켜짐/꺼짐,
각 photometric 변환 강제 적용, box/segment 입력, 빈 라벨, Mosaic, MixUp을 포함한다.
세부 결과와 버전은 [augmentation-parity.json](reproduction-audit/augmentation-parity.json)에 있다.
외부 원본을 지정한 parity 테스트 및 로더·평가·학습·CUDA/DDP 회귀 테스트의 최종 실행은
**47 passed**였으며 skip은 없었다.

RTX 2070 SUPER에서 v9-t scratch / 128px / batch 4 / worker 2 / CUDA AMP로 20 batch를
실행했다. loss와 가중치가 유한하고 실제 가중치 변경을 확인했다. optimizer 호출 10회 중
성공 2회, AMP skip 8회였으며 이 차이는 검증 JSON에 분리해서 기록했다.
`runs/train/yolov9-augmentation-smoke/verification.json` 및 checkpoint를 참조한다.
이는 val2017 일부를 이용한 실행 경로 진단이며 AP 비교나 하이퍼파라미터 선택 실험이 아니다.

이 검증은 증강 구현 및 학습 연결의 검증이다. 전체 COCO 장기 학습이나 논문 AP 재현 완료를 뜻하지 않는다.
