# GPU QAT / QDQ 수치 불일치 분석

2026-09-19 실제 RTX 2070 SUPER 측정. 최종 확률 차이의 원인을 중간 출력 추적과 한 가지 조건만 바꾸는 대조 실험으로 분리했다. **양자화 반올림, cuDNN 알고리즘, SiLU 계산 방식을 정렬하자 box 확률은 완전히 일치하고 class 확률의 최대 차이는 8.12e-8로 감소했다.** ORT 최적화 ON에서 추가로 발생하는 차이는 bias 양자화 변환으로 재현했다.

## 측정 범위와 조건

- PyTorch 2.6.0+cu124, ORT GPU 1.26.0, cuDNN 9.1, FP32, TF32 OFF, PyTorch cuDNN benchmark OFF.
- ORT 1.26은 현재 CUDA 12 환경에 맞춰 임시 경로 `/tmp/yolo-ort-gpu-cu12`에 설치했다. 기존 프로젝트 ORT 설치는 변경하지 않았다.
- YOLOv9-c, QAT 3 epoch / 15 step 진단 체크포인트. 충분히 학습된 배포 모델의 정확도 평가가 아니다.
- 입력: `demo/images/inference/image.png` 한 장을 RGB 640×640으로 resize, `/255`, NCHW로 변환.
- 관측 통계와 scale/zero point는 고정. export용 weight snapping을 동일하게 적용했다.
- 출력: `[1,8400,144]`, 앞 64채널은 left/top/right/bottom 각각 16개 bin 확률, 뒤 80채널은 class 확률.
- 기본 원인 추적은 ORT graph optimization OFF. 이 조건에서 Conv와 Q/DQ는 CUDA에서 실행되는 것을 이전 프로파일에서 확인했다.
- 아래 PyTorch forward 변경은 진단 프로세스 안에서만 적용했다. 학습용 STE, exporter, 체크포인트 또는 production 구현을 수정한 결과가 아니다.

정량 결과는 [JSON](qat-gpu-numerical-analysis-2026-09-19.json)에 저장했다.

## 1. 동일 입력에서도 fake quantization의 반올림 결과가 다르다

원래 구현에서 첫 Conv의 입력/weight/출력 QDQ는 모두 일치했다. 그 뒤 SiLU의 미세한 FP32 차이는 다음 입력 양자화에서 사라졌다. **두 번째 Conv의 양자화 전 출력도 완전히 같지만, 출력 양자화에서 3,276,800개 중 4개가 달라졌다.**

실제 양수 예시:

| 항목 | PyTorch CUDA | ORT CUDA |
|---|---:|---:|
| 입력 x | 1.495746374130249 | 1.495746374130249 |
| scale | 0.42735612392425537 | 0.42735612392425537 |
| 정규화 계산 | x × float32(1/scale) | x / scale |
| 정규화 결과 | 3.5 | 3.499999761581421 |
| 반올림 결과, zero point 제외 | 4 | 3 |
| 역양자화 결과 | 1.7094244956970215 | 1.2820683717727661 |

zero point는 양쪽 모두 119이다. 같은 원본 tensor를 PyTorch GPU quantizer에 다시 입력해도 이 4개의 불일치가 재현됐다. 따라서 이 지점은 Conv 오차가 아니라 **양자화 연산 자체의 차이**다. 두 구현 모두 ties-to-even 계열 반올림을 쓰지만, 반올림에 들어가는 값이 다르다.

관련 구현:

- [PyTorch 2.6 CUDA fake quantization](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/quantized/cuda/FakeQuantizeCore.cu#L98): reciprocal을 먼저 계산한 후 곱하고 반올림한다.
- [ORT 1.26 CUDA QuantizeLinear](https://github.com/microsoft/onnxruntime/blob/v1.26.0/onnxruntime/core/providers/cuda/tensor/quantize_linear.cu#L64): 나눗셈 결과를 반올림한다.

## 2. 양자화 계산을 맞춰도 Conv 알고리즘의 오차가 새로 유입된다

진단용 PyTorch fake quantizer를 나눗셈 기반으로 바꾸면 위 불일치는 사라진다. ORT의 `cudnn_conv_algo_search=DEFAULT`에서는 다섯 번째 Conv인 `model.model.2.conv2.0.bottleneck.0.conv1.reparam`에서 다음 차이가 발생했다.

- Conv 입력 양자화 결과 동일.
- Conv 출력 최대 차이: `7.62939453125e-6`.
- 출력 양자화 후: 819,200개 중 2개 불일치, 최대 차이 `0.07914447784`.
- 예: `-2.5721940994`와 `-2.5721960068`이 scale로 나누면 `-32.49999237`과 `-32.50001526`이 되어 서로 다른 bin으로 반올림된다.

입력·가중치·bias를 고정한 독립 Conv 실험:

| ORT 알고리즘 선택 | PyTorch 대비 Conv 최대 차이 |
|---|---:|
| DEFAULT | 7.62939453125e-6 |
| HEURISTIC | 0 |
| EXHAUSTIVE | 0 |

bias를 Conv 밖에서 더하는 대조 실험은 결과를 바꾸지 않았다. PyTorch deterministic 설정만 켜도 결과는 바뀌지 않았다. 이 Conv에서는 **cuDNN 계산 경로 선택**이 차이를 만든다는 근거다. 이것이 모든 레이어·GPU에서 HEURISTIC의 bitwise 일치를 보장한다는 뜻은 아니다. `DEFAULT`는 실험에 사용한 옵션 값이며 ORT 설정을 생략했을 때의 기본값을 뜻하지 않는다. [ORT 알고리즘 선택 문서](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cudnn_conv_algo_search)

## 3. SiLU의 미세한 차이도 후속 양자화 경계를 넘는다

나눗셈 기반 quantizer와 ORT HEURISTIC을 함께 사용하면 초기 Conv들의 양자화 출력이 모두 일치한다. 첫 61개 Conv의 입력/출력 quantizer를 촘촘히 추적한 결과, 다음 불일치는 46번째 Conv의 입력인 `model.model.6.conv3.0.conv3.conv.input_fake_quant`에서 시작했다.

- 상류 Conv들의 양자화 출력은 동일.
- SiLU와 residual Add를 지난 입력의 특정 위치: `3.913001298904419` 대 `3.91300106048584`.
- scale `0.037090059369802475`로 나눈 결과: `105.5` 대 `105.49999237060547`.
- 출력: `3.931546211242676` 대 `3.894456148147583`.
- 입력 quantizer 409,600개 중 1개 불일치. 같은 ORT 입력으로 quantizer만 비교하면 차이는 0.

PyTorch의 SiLU는 `x / (1 + exp(-x))`를 계산한다. ONNX는 Sigmoid 뒤 Mul로 분해되며, ORT CUDA Sigmoid는 음수 입력에 `1 - 1/(1 + exp(x))` 형태를 사용한다. 실수 수학에서는 같지만 FP32 연산 순서가 다르다.

- [PyTorch SiLU CUDA](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/ActivationSiluKernel.cu#L33)
- [ORT Sigmoid CUDA](https://github.com/microsoft/onnxruntime/blob/v1.26.0/onnxruntime/core/providers/cuda/activation/activations_impl.cu#L48)

PyTorch SiLU를 단순히 `x * torch.sigmoid(x)`로 바꾸는 것만으로는 결과가 개선되지 않았다. ORT의 부호별 Sigmoid 식과 곱셈 순서까지 맞추자 큰 불일치가 사라졌다.

**ORT HEURISTIC, optimization OFF를 고정한 비교:**

| PyTorch 진단 조건 | 전체 확률 최대 차이 | 평균 차이 | class 최대 차이 |
|---|---:|---:|---:|
| 원래 fake quant / SiLU | 0.31289515 | 0.001880847 | 0.09668186 |
| 모든 fake quant를 나눗셈으로 정렬 | 0.32376576 | 0.001656824 | 0.09668186 |
| 추가로 `x * torch.sigmoid(x)` 적용 | 0.32376576 | 0.001656824 | 0.09668186 |
| 추가로 ORT의 SiLU 계산 방식 적용 | **8.1199687e-8** | **1.6275130e-8** | **8.1199687e-8** |

마지막 조건에서 **box bin 확률의 최대 차이는 0**이다. 남은 차이는 마지막 class Sigmoid 출력에 있다. 일부 원인만 제거한다고 최댓값이 단조롭게 감소하지는 않는다. 다른 위치의 bin 선택이 달라지고 최대 오차 위치도 바뀌기 때문이다.

## 4. ORT 최적화 ON은 float bias까지 양자화한다

최적화된 그래프를 저장해 확인하니 `WeightBiasQuantization` 변환이 모든 144개 Conv의 float bias를 다음 grid로 바꿨다.

`bias_scale = input_scale × weight_scale`

`bias_int32 = round(bias / bias_scale)`

현재 QAT 구현은 bias를 float로 학습하고 그대로 사용한다. 이 변환은 PyTorch reference에 없던 별도 양자화다. 전체 bias 24,944개 중 24,931개 값이 바뀌었고, 최대 차이는 `0.0008458122611`이었다. 첫 Conv bias에서도 같은 최대 차이가 관찰됐다.

ORT DEFAULT, 원래 PyTorch reference 조건에서 변환을 독립적으로 검증했다.

| 실험 | 결과 |
|---|---|
| ORT optimization OFF vs PyTorch | 최대 0.32823604, 평균 0.002014679 |
| ORT optimization ON vs PyTorch | 최대 0.43471181, 평균 0.002610541 |
| ON에서 `WeightBiasQuantization`만 비활성화 | **OFF 출력과 완전히 동일** |
| OFF 그래프에서 bias만 ON의 양자화 값으로 교체 | **ON 출력과 완전히 동일** |

따라서 이 입력에서 ON/OFF의 추가 차이는 bias 양자화로 완전히 재현된다. GPU에서 관찰한 이 현상을 CPU U8S8 포화 문제로 설명해서는 안 된다.

- [ORT WeightBiasQuantization 구현](https://github.com/microsoft/onnxruntime/blob/v1.26.0/onnxruntime/core/optimizer/qdq_transformer/weight_bias_quantization.cc#L208)
- [변환 계약 설명](https://github.com/microsoft/onnxruntime/blob/v1.26.0/onnxruntime/core/optimizer/qdq_transformer/weight_bias_quantization.h)

## 구현에 대한 판단

이번 차이는 출력 concat 순서나 softmax 축의 문제가 아니라, **실수 수학에서는 같은 계산이 FP32에서는 달라지고 양자화가 그 차이를 정수 bin 차이로 확대하는 문제**다. bias 양자화는 여기에 추가되는 명시적 모델 변환이다.

후속 구현에서는 목표 backend의 반올림·bias·activation 처리와 QAT forward의 계약을 맞춰야 한다. 진단용 나눗셈/round 코드를 그대로 학습에 넣으면 STE가 사라지므로 별도 backward와 ONNX QDQ symbolic을 갖춰야 한다. ORT에 맞춘 SiLU 수식이 Qualcomm/Samsung/MediaTek에서도 일치를 보장하는 것은 아니다.

이번 결과는 한 이미지와 짧은 진단 학습 체크포인트에서 원인을 분리한 결과다. 여러 이미지의 검출 정확도, 충분한 QAT 학습, 휴대폰 NPU 컴파일 및 실기기 검증을 대신하지 않는다.

## 실행 증거

현재 세션의 진단 스크립트와 원본 결과:

- `/tmp/trace_qdq_gpu_cause.py`, `/tmp/qat-gpu-cause-layerwise.json`
- `/tmp/trace_qdq_gpu_div.py`, `/tmp/qat-gpu-div-layerwise.json`
- `/tmp/qat_gpu_conv5.py`, `/tmp/qat-gpu-conv5.json`
- `/tmp/trace_qdq_gpu_dense.py`, `/tmp/qat-gpu-div-heuristic-dense.json`
- `/tmp/qat_gpu_ablate_silu.py`, `/tmp/qat-gpu-heuristic-ablation.json`
- `/tmp/qat_gpu_optimization.py`, `/tmp/qat-gpu-optimization.json`

GPU 진단은 샌드박스 밖에서 `PYTHONPATH=/tmp/yolo-ort-gpu-cu12:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python <script>`로 실행했다. 스크립트는 위 체크포인트와 ONNX의 세션 임시 경로를 참조한다.
