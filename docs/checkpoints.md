# 체크포인트 저장과 학습 재개

`yolo task=train name=my-run`의 저장 위치는 `runs/train/my-run/checkpoints/`입니다.
`out_path`를 지정하면 `runs` 대신 해당 경로를 사용합니다. W&B, TensorBoard, quiet 설정과 무관하게 동일합니다.

- `epoch0003-step00000042.ckpt`: 가장 최근 epoch의 전체 학습 상태. 기존처럼 최신 체크포인트 하나를 유지합니다.
  epoch는 Lightning의 0부터 시작하는 인덱스이며 4자리, step은 global step이며 8자리로 zero-fill합니다.
  자리수를 초과하면 숫자를 자르지 않습니다. 모델, optimizer, scheduler, EMA와 누적 학습 상태를 포함합니다.
- `best.pt`: 검증 mAP@0.5:0.95 (`map`)가 이전 최고값보다 높을 때 갱신하는 가중치 전용 파일입니다.
  EMA를 사용하면 검증에 사용한 EMA 가중치를 저장합니다. 동점·NaN·sanity validation에서는 갱신하지 않습니다.
  최고 점수도 `.ckpt`에 저장하므로 재개 후 낮은 점수로 덮어쓰지 않습니다.

## 이름으로 자동 재개

```bash
yolo task=train name=my-run model=v9-t
```

명시적인 `weight` 설정이 없으면 해당 실행 폴더 안의 `.ckpt`를 하위 폴더까지 검색하고,
파일 내부의 `(epoch, global_step)`이 가장 큰 체크포인트를 재개합니다. epoch를 먼저 비교하고,
같은 epoch에서는 step을 비교합니다. 기존 `epoch=...-step=...ckpt`, `last.ckpt`,
로거의 `version_*/checkpoints/` 경로도 지원합니다. 손상되어 읽을 수 없는 파일은 경고 후 건너뜁니다.
체크포인트가 없으면 기존 가중치 초기화 방식으로 새 학습을 시작합니다.

자동 재개 시 `exist_ok=False`여도 같은 실행 폴더를 사용합니다. 기본 이름 `v9-dev`도 같은 규칙을 적용합니다.
`task.epoch`는 추가 epoch 수가 아닌 총 목표 epoch 수입니다. 재개에는 기존 모델·데이터 설정을 유지하세요.

## weight 우선순위

```bash
# 지정한 체크포인트로 전체 학습 상태를 복원
yolo task=train name=my-run model=v9-t weight=/path/to/epoch0003-step00000042.ckpt

# 가중치만 읽고 epoch 0부터 새 학습
yolo task=train name=fine-tune model=v9-t weight=runs/train/my-run/checkpoints/best.pt

# 자동 재개를 끄고 무작위 / 기본 사전학습 가중치로 새 학습
yolo task=train name=new-run model=v9-t weight=False
yolo task=train name=new-run model=v9-t weight=True
```

명시한 가중치 경로와 CLI의 `weight=True/False/null` 설정은 이름 기반 재개보다 우선합니다.
기본 설정 파일의 `weight: True`는 자동 재개를 막지 않습니다. 가중치만 담긴 `best.pt`에는
optimizer나 학습 진행 상태가 없으므로 전체 학습 재개에는 `.ckpt`를 사용합니다.

최고 점수를 기록하지 않던 과거 체크포인트는 재개 후 첫 유효한 검증 점수를 best 기준으로 삼습니다.
