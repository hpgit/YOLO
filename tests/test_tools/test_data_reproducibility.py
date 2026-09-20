import hashlib
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning import LightningModule, Trainer
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

from yolo.tools import data_loader
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.utils.logging_utils import set_seed


@pytest.fixture
def image_configs(tmp_path):
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rng = np.random.default_rng(731)
    for index in range(16):
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(images / f"{index:02}.png")
        (labels / f"{index:02}.txt").write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")
    return OmegaConf.create(
        {
            "shuffle": True,
            "batch_size": 4,
            "cpu_num": 0,
            "pin_memory": False,
            "image_size": [64, 64],
            "data_augment": {},
        }
    ), OmegaConf.create({"path": str(tmp_path), "train": "train", "validation": "train", "class_num": 1})


def _epoch(loader):
    paths, digest = [], hashlib.sha256()
    for _, images, targets, reverse, batch_paths in loader:
        paths.extend(str(path) for path in batch_paths)
        for tensor in (images, targets, reverse):
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return paths, digest.hexdigest()


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("recipe", ["legacy", "yolov9"])
def test_shuffle_and_augmented_batches_repeat_across_runs(image_configs, workers, recipe):
    data_cfg, dataset_cfg = image_configs
    data_cfg.cpu_num = workers
    data_cfg.data_augment = (
        {"YOLOv9": {"mixup": 1.0, "motion_blur": 0.5}}
        if recipe == "yolov9"
        else {"HorizontalFlip": 0.5, "MotionBlur": 0.5}
    )

    def run(seed):
        set_seed(seed)
        loader = create_dataloader(data_cfg, dataset_cfg)
        assert isinstance(loader.sampler, RandomSampler)
        return [_epoch(loader), _epoch(loader)]

    first = run(10)
    assert first == run(10)  # Paths, pixels, labels, reverse transforms: both epochs.
    assert first != run(11)
    assert first[0][0] != first[1][0]
    assert first[0][1] != first[1][1]
    expected = sorted(str(path) for path in (Path(dataset_cfg.path) / "images/train").glob("*.png"))
    for paths, _ in first:
        assert sorted(paths) == expected
        assert paths != expected


@pytest.mark.parametrize("task", ["train", "validation"])
def test_shuffle_false_keeps_order(image_configs, task):
    data_cfg, dataset_cfg = image_configs
    data_cfg.shuffle = False
    set_seed(10)
    loader = create_dataloader(data_cfg, dataset_cfg, task)
    assert isinstance(loader.sampler, SequentialSampler)
    first = _epoch(loader)
    assert first == _epoch(loader)
    assert first[0] == sorted(first[0])


def test_model_rng_and_validation_do_not_change_training_order(image_configs):
    data_cfg, dataset_cfg = image_configs

    def run(consume_rng):
        set_seed(10)
        if consume_rng:
            torch.rand(1000)
        loader = create_dataloader(data_cfg, dataset_cfg)
        first = _epoch(loader)
        if consume_rng:
            torch.rand(1000)
            _epoch(create_dataloader(data_cfg, dataset_cfg, "validation"))
        return first, _epoch(loader)

    assert run(False) == run(True)


def test_dynamic_shapes_reject_shuffled_batches(image_configs):
    data_cfg, dataset_cfg = image_configs
    data_cfg.dynamic_shape = True
    with pytest.raises(ValueError, match="requires shuffle=False"):
        create_dataloader(data_cfg, dataset_cfg)


class RandomDataset(Dataset):
    def __init__(self, *args):
        pass

    def __len__(self):
        return 16

    def __getitem__(self, index):
        worker = torch.utils.data.get_worker_info()
        values = torch.tensor([random.random(), np.random.random(), torch.rand(()).item()], dtype=torch.float64)
        return values, torch.zeros(0, 5), torch.tensor([worker.id if worker else -1]), str(index)


def test_workers_have_distinct_repeatable_rng_streams(image_configs, monkeypatch):
    data_cfg, dataset_cfg = image_configs
    data_cfg.cpu_num = 2
    data_cfg.batch_size = 1
    data_cfg.shuffle = False
    monkeypatch.setattr(data_loader, "YoloDataset", RandomDataset)

    def run():
        set_seed(10)
        loader = create_dataloader(data_cfg, dataset_cfg)
        return [list(loader), list(loader)]

    first, repeated = run(), run()
    for epoch, other in zip(first, repeated):
        values = torch.cat([batch[1] for batch in epoch])
        assert {batch[3].item() for batch in epoch} == {0, 1}
        assert all(torch.unique(values[:, index]).numel() == 16 for index in range(3))
        assert torch.equal(values, torch.cat([batch[1] for batch in other]))
    assert not torch.equal(first[0][0][1], first[1][0][1])


def test_set_seed_initializes_all_global_rngs():
    def sample(seed):
        set_seed(seed)
        return random.random(), np.random.random(), torch.rand(4)

    first, repeated, changed = sample(0), sample(0), sample(1)
    assert first[:2] == repeated[:2] != changed[:2]
    assert torch.equal(first[2], repeated[2])
    assert not torch.equal(first[2], changed[2])
    assert os.environ["PL_SEED_WORKERS"] == "1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_set_seed_initializes_cuda_rng():
    set_seed(10)
    first = torch.rand(8, device="cuda")
    set_seed(10)
    assert torch.equal(first, torch.rand(8, device="cuda"))
    set_seed(11)
    assert not torch.equal(first, torch.rand(8, device="cuda"))


def test_drawing_does_not_reset_training_rng():
    set_seed(10)
    state = random.getstate()
    draw_bboxes(Image.new("RGB", (64, 64)), torch.tensor([[0, 10, 10, 40, 40]]))
    assert random.getstate() == state


def test_cli_seeds_before_trainer_and_model_construction(monkeypatch, tmp_path):
    import yolo.lazy as lazy

    samples = []

    def construct(*args, **kwargs):
        samples.append((random.random(), np.random.random(), torch.rand(1).item()))
        return SimpleNamespace(fit=lambda *args, **kwargs: None)

    monkeypatch.setattr(lazy, "resolve_training_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(lazy, "setup", lambda *args, **kwargs: ([], [], tmp_path))
    monkeypatch.setattr(lazy, "Trainer", construct)
    monkeypatch.setattr(lazy, "TrainModel", construct)
    cfg = OmegaConf.create({"lucky_number": 10, "device": 1, "task": {"task": "train", "epoch": 1}})
    lazy.main.__wrapped__(cfg)
    lazy.main.__wrapped__(cfg)
    cfg.lucky_number = 11
    lazy.main.__wrapped__(cfg)
    assert samples[:2] == samples[2:4] != samples[4:6]


def test_default_hydra_shuffle_and_seed():
    with initialize_config_dir(config_dir=str(Path(data_loader.__file__).parents[1] / "config"), version_base=None):
        cfg = compose(config_name="config", overrides=["task=train", "lucky_number=27"])
    assert cfg.task.data.shuffle is True
    assert cfg.task.validation.data.shuffle is False
    assert cfg.lucky_number == 27


class SeededTrainingModel(LightningModule):
    """Observe actual Lightning sampler replacement, worker RNGs, and updates."""

    def __init__(self, loader, output_dir):
        super().__init__()
        self.layer = torch.nn.Linear(3, 1, dtype=torch.float64)
        self.loader = loader
        self.output_dir = output_dir
        self.records = []

    def train_dataloader(self):
        return self.loader

    def training_step(self, batch, batch_idx):
        self.records.append((self.current_epoch, batch[4], batch[1].tolist()))
        return self.layer(batch[1]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def on_train_end(self):
        torch.save(
            {"records": self.records, "weights": self.layer.state_dict()},
            self.output_dir / f"rank-{self.global_rank}.pt",
        )


@pytest.mark.parametrize("devices", [1, 2])
def test_lightning_training_repeats_with_seed_including_ddp(image_configs, monkeypatch, tmp_path, devices, request):
    if devices == 2:
        # PyTorch 2.6's FD sharing cannot unpickle a Generator on this WSL host.
        # File-system sharing exercises real spawn without changing production defaults.
        previous = torch.multiprocessing.get_sharing_strategy()
        torch.multiprocessing.set_sharing_strategy("file_system")
        request.addfinalizer(lambda: torch.multiprocessing.set_sharing_strategy(previous))
    data_cfg, dataset_cfg = image_configs
    data_cfg.cpu_num = 2
    monkeypatch.setattr(data_loader, "YoloDataset", RandomDataset)

    def run(name, seed):
        set_seed(seed)
        output = tmp_path / name
        output.mkdir()
        model = SeededTrainingModel(create_dataloader(data_cfg, dataset_cfg), output)
        trainer = Trainer(
            accelerator="cpu",
            devices=devices,
            max_epochs=2,
            precision="64-true",
            strategy=DDPStrategy(process_group_backend="gloo", start_method="spawn") if devices == 2 else "auto",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            default_root_dir=output,
        )
        trainer.fit(model)
        return [torch.load(output / f"rank-{rank}.pt", weights_only=True) for rank in range(devices)]

    first, repeated, changed = run("first", 10), run("repeat", 10), run("changed", 11)
    for rank in range(devices):
        assert first[rank]["records"] == repeated[rank]["records"] != changed[rank]["records"]
        for name, weights in first[rank]["weights"].items():
            assert torch.equal(weights, repeated[rank]["weights"][name])
    for epoch in range(2):
        rank_paths = [
            [path for current, paths, _ in result["records"] if current == epoch for path in paths] for result in first
        ]
        all_paths = sum(rank_paths, [])
        assert len(all_paths) == len(set(all_paths)) == 16
    if devices == 2:
        assert first[0]["records"][0][2] != first[1]["records"][0][2]
