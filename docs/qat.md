# YOLOv9 QAT와 QDQ ONNX

`qat.enabled=true`로 기존 학습 경로에서 QAT fine-tuning을 수행하고,
`task=export task.qdq=true`로 학습한 양자화 파라미터를 포함한 ONNX를 생성합니다.
기존 FP 학습·export는 기본 설정에서 그대로 동작합니다.

## 지원 범위와 양자화 방식

현재 프로파일 `conv_w8a8`은 YOLOv9 DFL detection 모델의 **convolution 경계 QAT**입니다.

| 대상 | 처리 |
| --- | --- |
| RepConv | 두 Conv-BN branch를 하나의 convolution으로 합친 후 QAT |
| 나머지 Conv-BN | pretrained running statistics로 BN folding; 이후 BN 재학습 없음 |
| 학습 head | Main만 사용; AUX 제거, 기존 BCE/IoU/DFL loss 사용 |
| Conv2d weight | 출력 채널별 symmetric INT8, axis=0, 범위 -128…127 |
| Conv2d 입력·출력 | 텐서별 affine UINT8, 범위 0…255 |
| Bias | float; 정수 bias 표현은 대상 컴파일러 단계에서 결정 |
| SiLU, Add, Concat, Pool, resize | float 연산; 이어지는 convolution 입력에서 다시 fake quantization |
| 최종 Sigmoid·Softmax·Concat | float 확률 출력 |
| 학습 precision / EMA | FP32, EMA 사용 안 함 |
| 분산 학습 | 현재 단일 device만 지원; observer 동기화 없는 DDP는 거부 |

PyTorch의 `FakeQuantize`와 moving-average min/max observer를 사용합니다.
Forward에는 clipping/rounding 오차가 반영되고, backward에는 STE gradient가 흐릅니다.
Scale과 zero-point는 학습 데이터 통계로 갱신되며 gradient로 직접 최적화하는 LSQ 방식은 아닙니다.
출력 확률까지 UINT8로 바꾸거나 모든 연산을 INT8로 실행한다고 가정하지 않습니다.

이 ONNX는 표준 Q/DQ 노드를 포함하는 교환 형식입니다. Qualcomm QNN/AIMET,
삼성 Exynos, MediaTek NeuroPilot의 구체적인 quantization 규칙·지원 연산·partition은
각 SDK에서 확인해야 합니다. NPU compile, 전체 INT8 실행, AP 유지, latency 개선을
이 구현만으로 보장하지 않습니다. FP 및 PTQ 기준선과 실제 기기 비교가 별도로 필요합니다.

## 학습

수렴한 FP 체크포인트에서 시작하고, 학습 데이터로 ranges를 수집합니다.
모델·클래스 설정은 원본 체크포인트와 일치시켜야 합니다.

```bash
python yolo/lazy.py task=train model=v9-c dataset=coco \
  weight=weights/v9-c.pt name=v9-c-qat qat.enabled=true \
  task.epoch=10 task.optimizer.args.lr=0.0001 device=1
```

위 epoch/LR은 실행 예시이며 성능 검증된 최적 설정이 아닙니다.
기본 일정은 0-based epoch 기준입니다.

- epoch 0: observer 수집, fake quantization 비활성화.
- epoch 1–2: observer 수집과 fake quantization 활성화.
- epoch 3 이후: observer 고정, fake quantization 유지하며 fine-tuning.

`qat.fake_quant_start_epoch`, `qat.observer_freeze_epoch`,
`qat.averaging_constant`로 조절합니다. 시작 epoch보다 freeze epoch가 커야 합니다.
검증·stride probe는 observer를 갱신하지 않습니다. 첫 학습 이전의 sanity validation은
아직 ranges가 없으므로 FP로 실행됩니다. 학습 CLI는 QAT일 때 FP32를 선택하고,
EMA callback을 제외합니다. Direct Trainer 사용 시 `precision="32-true"`와 EMA 비활성화가 필요합니다.
QAT에서는 FP 학습의 초기 bias warmup LR 0.1 대신 설정된 fine-tuning LR에서 시작합니다.

RepConv fusion은 **fake quantization 이전**에 수행합니다.
`quantize(branch1 + branch2)`와 `quantize(branch1) + quantize(branch2)`는 일반적으로
같지 않으므로, QAT 이후 임의로 branch fusion을 수행하지 않습니다.

## 저장·재개

QAT `best.pt`에는 모델 구조를 재구성할 metadata, fused weights, observer min/max,
scale, zero-point, enable flags가 저장됩니다. 일반 FP weight-only 파일과 형식이 다릅니다.
QAT 모델은 구조·클래스·reg_max를 검사하고 state를 strict하게 로드합니다.

```bash
# full optimizer/scheduler/observer 상태에서 재개
python yolo/lazy.py task=train model=v9-c dataset=coco \
  weight=runs/train/v9-c-qat/checkpoints/epoch0009-step00012345.ckpt \
  name=v9-c-qat task.epoch=20 device=1
```

체크포인트 metadata로 QAT 설정을 자동 복원하므로 `qat.enabled=true`를 다시 지정할
필요가 없습니다. 명시한 QAT 설정이 저장된 설정과 다르면 오류를 냅니다.
`best.pt`를 학습 weight로 주면 weight/observer 상태에서 새 optimizer로 fine-tuning합니다.
FP `.ckpt`에 `qat.enabled=true`를 지정하면 FP model weights로 새 QAT 학습을 시작합니다.
기존 FP run의 이름으로 자동 resume하면서 QAT를 켜는 경우에는 새 run 이름을 요구합니다.

## QDQ ONNX export

```bash
pip install -e '.[export-onnx]'
python yolo/lazy.py task=export task.format=onnx task.qdq=true \
  model=v9-c dataset=coco \
  weight=runs/train/v9-c-qat/checkpoints/best.pt \
  'image_size=[640,640]' task.output=runs/export/v9-c-qat.onnx
```

`best.pt`는 fake quantization 활성화 이후의 검증 결과만 후보로 사용합니다.
미초기화 observer, warmup 상태의 `.ckpt`, 잘못된 scale 또는 FP weight를
QDQ export에 주면 오류가 발생합니다. Export의 dummy input으로 ranges를 수집하거나
학습한 encodings를 재계산하지 않습니다.
QAT weight를 일반 FP/TFLite export에 주는 경우도 오류를 내어 QDQ의 조용한 유실을 막습니다.

- 하나의 float32 출력 `[B,N,4*reg_max+C]`.
- 순서: `[left bins, top bins, right bins, bottom bins, class 확률]`.
- 방향별 bin Softmax, class Sigmoid. DFL 기대값·box decode·NMS는 외부 처리.
- 기본 COCO 640×640, reg_max=16: `[1,8400,144]`.
- 내부 텐서·가중치·상수를 포함해 rank ≤ 4 검증.
- ONNX opset ≥ 13, 기본 17. `task.dynamic_batch=true` 지원, 공간 크기는 고정.
- 각각의 fake quantizer를 `QuantizeLinear` → `DequantizeLinear`로 내보냄.
- Export 복사본의 weight를 학습한 양자화 격자에 맞춰 저장하여 half-bin 반올림 차이를 방지.
  학습 원본 weight와 scale/zero-point는 변경하지 않음.
- ONNX metadata `yolo.qat`, `yolo.qat.encodings`에 프로파일과 모든 scale/zero-point 기록.

Float weights와 Q/DQ를 함께 저장하므로 파일 크기가 반드시 FP 대비 1/4이 되지는 않습니다.
검증은 먼저 ONNX Runtime의 graph optimization을 끄고 Q/DQ 의미를 비교한 뒤,
대상 backend 최적화/compile 결과를 별도로 비교하는 방식으로 수행합니다.

## 검증과 근거

`tests/test_tools/test_qat.py`는 FP fusion 동등성, 실제 detection loss의 gradient,
observer freeze 및 validation 중 통계 보존, best/full checkpoint와 학습 재개,
모든 Q/DQ scale/zero-point/dtype/axis, 정적·동적 batch의 ONNX Runtime 결과,
그래프 전체 rank 제한을 검사합니다. 작은 모델/데이터에서의 smoke 검증은
COCO 성능 또는 모바일 기기 성능을 입증하지 않습니다.

- [PyTorch FakeQuantize](https://docs.pytorch.org/docs/main/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)
- [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
- [ONNX DequantizeLinear](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html)

### 2026-09-19 실행 결과와 남은 검증

환경: PyTorch 2.6.0+cu124, ONNX 1.22.0, ONNX Runtime 1.30.0, CPU 실행.
관련 회귀 테스트 74개 통과, 3개 skip. 테스트의 작은 모델에서는 PyTorch/QDQ
출력이 설정된 수치 허용오차 이내였으며, 모든 scale·zero-point·axis·dtype와
양자화 weight 값의 보존도 확인했습니다.

별도로 `weights/v9-c.pt`와 mock 데이터에서 다음 경로를 실행했습니다.

- 64×64에서 QAT 2 epoch → full checkpoint 재개 1 epoch.
- 640×640에서 3 epoch: observer-only warmup → QAT → observer 고정.
- 마지막 QAT `.ckpt`에서 640×640 ONNX export 및 ONNX Runtime 실행.
- 출력 `[1,8400,144]`, QuantizeLinear 432개, DequantizeLinear 432개,
  모든 내부 tensor/initializer/constant rank ≤ 4.

**큰 모델의 출력 수치 동등성은 아직 확인되지 않았습니다.** 위 640×640 smoke
체크포인트와 데모 이미지 한 장에서 측정한 절대 확률 차이는 다음과 같습니다.

| 비교 | 평균 차이 | 최대 차이 | class 최대 차이 |
| --- | ---: | ---: | ---: |
| PyTorch 기본 CPU vs ORT, graph optimization 비활성화 | 0.002064 | 0.338507 | 0.074614 |
| PyTorch 기본 CPU vs ORT, graph optimization 활성화 | 0.010440 | 0.906926 | 0.211022 |
| PyTorch 내부에서 MKLDNN 활성/비활성 비교 | 0.002066 | 0.328011 | 0.074614 |

따라서 이 smoke 결과를 실모델 PyTorch↔ONNX 동등성 통과, 검출 AP 유지,
또는 NPU 배포 준비 완료로 해석하면 안 됩니다. Kernel의 작은 float 차이가
quantization 경계에서 확대되는 현상이 관찰됐지만 잔여 차이의 원인이 모두
규명된 것은 아닙니다. Export weight의 half-bin 반올림 차이는 격자 정렬로
보완했고, scale/zero-point를 재보정해 차이를 감추지는 않았습니다.

QDQ 구조/encoding 검사와 실제 backend의 출력·AP 검사는 별개의 검증 단계입니다.
충분한 대표 데이터로 fine-tuning하고, FP/PTQ 기준선과 backend AP를 비교한 후,
대상 NPU의 compile/partition/numerical/latency 검증을 수행해야 합니다.
ORT의 U8S8 integer 경로에도 CPU별 saturation 제약이 있으므로, graph optimization을
켜고 끈 결과를 함께 확인합니다. 이 제약은 모든 잔여 차이의 원인이라고 단정한 것이
아닙니다. [ONNX Runtime quantization 문서](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

추가 GPU 실험에서는 양자화 반올림, cuDNN 알고리즘, SiLU 계산 방식의 차이와
ORT 최적화의 bias 양자화를 분리했습니다. 진단 프로세스에서 계산 방식을 정렬하자
동일 이미지의 box 확률은 완전히 일치하고 class 최대 차이는 `8.12e-8`로 줄었습니다.
이 정렬은 현재 학습·export 구현에는 아직 반영하지 않았습니다.
재현 조건과 대조 실험은 [GPU 수치 불일치 분석](qat-gpu-numerical-analysis-2026-09-19.md)을 참고하세요.
