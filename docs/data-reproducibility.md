# Shuffle과 seed 적용

학습의 `task.data.shuffle=True`는 실제 DataLoader에 전달된다. 각 epoch는 전체 데이터를
한 번씩 순회하면서 새로운 순서를 만든다. `shuffle=False`는 순차 순회를 유지하며,
validation의 기본값은 `False`이다. DDP에서는 Lightning이 분산 sampler로 교체하고
epoch를 설정한다. 데이터 수가 world size로 나누어떨어지지 않을 때는 분산 sampler의
기본 padding으로 일부 sample이 중복될 수 있다.

```bash
.venv/bin/python -m yolo.lazy task=train lucky_number=10 task.data.shuffle=True
```

기존 seed 설정 이름은 `lucky_number`이며 기본값은 `10`이다. CLI 진입 시 모델·데이터·Trainer
생성 전에 Python, NumPy, PyTorch CPU/CUDA와 Lightning worker seeding을 설정한다.
각 DataLoader는 이 seed에서 시작하는 전용 `torch.Generator`를 사용하므로 모델 초기화나
validation의 torch 난수 소비가 학습 sample 순서를 바꾸지 않는다. Generator는 epoch 사이에
계속 진행하며, 매 epoch 같은 seed로 재설정하지 않는다.

worker에는 Lightning의 `pl_worker_init_function`을 명시적으로 연결한다. 따라서 Trainer 없이
DataLoader만 순회해도 worker별 Python/NumPy/PyTorch 증강 난수가 초기화된다. DDP에서는 rank도
반영된다. `cpu_num=0`에서는 메인 프로세스의 seed를 사용한다. bbox 시각화 색상은 별도
`random.Random` 객체로 생성하므로 학습용 Python 난수를 재설정하지 않는다.

Python API를 직접 사용한다면 모델과 DataLoader를 생성하기 **전에** 호출한다.

```python
from yolo.utils.logging_utils import set_seed

set_seed(10)
# 이후 TrainModel(cfg), create_dataloader(...) 등을 생성한다.
```

legacy `dynamic_shape=True`는 sample index별로 이미지 크기를 정하므로 임의 shuffle과 함께 쓰면
한 batch의 이미지 크기가 달라질 수 있다. 이 조합은 명확한 오류로 거부한다.
기본 YOLOv9 recipe는 고정 크기이며 `shuffle=True`를 사용한다.

검증은 다음을 포함한다.

- worker 0/2, legacy/YOLOv9 증강 각각에서 같은 seed의 두 번 실행 결과 비교:
  두 epoch의 sample 순서, 이미지, 라벨, 역변환 tensor가 정확히 일치한다.
- 다른 seed 및 다음 epoch에서 순서/증강 변화, 전체 sample 순회, `shuffle=False` 순서 유지.
- worker별 Python/NumPy/PyTorch 난수 분리, CUDA seed, 시각화 전후 난수 상태 보존.
- 실제 Lightning 단일 프로세스 및 CPU 2-process DDP의 두 epoch 학습 반복:
  같은 seed의 sample/난수/최종 가중치 일치와 rank 간 sample 분할 및 난수 분리.

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  tests/test_tools/test_data_reproducibility.py \
  tests/test_tools/test_data_loader.py
```

2026-09-14 검증: 새 재현성 테스트 **16 passed**(CUDA 포함), 기존 loader·증강·logging·checkpoint·
export 진입점 회귀 테스트 **57 passed**. CPU DDP spawn 테스트는 이 WSL/PyTorch 2.6.0 환경의
Generator FD 공유 오류를 피하기 위해 테스트 안에서만 `file_system` 공유 방식을 사용하고 복원한다.
직접 `ddp_spawn`을 사용하는 API 실행에서 같은 오류가 발생하면 이 공유 방식이 필요하다.
일반 CLI의 DDP 실행과 구분하며, production의 전역 multiprocessing 설정은 변경하지 않는다.

재현성 비교에는 같은 데이터, 라이브러리, 하드웨어, worker 수, world size 및 실행 조건이 필요하다.
`cpu_num=0`에서 다른 코드가 전역 Python/NumPy/torch 난수를 소비하면 이후 증강은 달라질 수 있다.
checkpoint에 DataLoader generator와 전체 RNG 상태를 추가 저장하는 변경은 포함하지 않으므로,
중단 후 재개를 연속 실행과 bitwise 동일하게 만드는 보장은 없다. 이 검증은 AP 성능 검증이 아니다.
