from queue import Queue
from threading import Event
from time import sleep
from types import SimpleNamespace

import numpy as np
import pytest
from lightning import LightningModule, Trainer
from PIL import Image

from yolo.tools import data_loader
from yolo.tools.data_loader import StreamDataLoader


def config(source):
    return SimpleNamespace(source=str(source), image_size=[32, 32])


class FrameCounter(LightningModule):
    def __init__(self):
        super().__init__()
        self.seen = []

    def predict_step(self, batch, batch_idx):
        self.seen.append(batch[2].getpixel((0, 0))[0])
        return batch_idx


@pytest.mark.parametrize("prefetched", [0, 10])
def test_lightning_consumes_all_images_regardless_of_prefetch(tmp_path, monkeypatch, prefetched):
    for index in range(37):
        folder = tmp_path / str(index % 2)
        folder.mkdir(exist_ok=True)
        Image.new("RGB", (32, 32), (index, 0, 0)).save(folder / f"{index}.png")

    # Freeze the producer at the queue size Lightning used to mistake for the
    # total input count. Release it only after Trainer has inspected the loader.
    ready, release = Event(), Event()
    original = StreamDataLoader.process_image
    produced = 0

    def gated(self, path):
        nonlocal produced
        if produced == prefetched:
            ready.set()
            assert release.wait(10)
        original(self, path)
        produced += 1

    monkeypatch.setattr(StreamDataLoader, "process_image", gated)
    # Allow ten prefetched frames in this regression, independent of the normal
    # queue capacity. The producer blocks before it can add frame eleven.
    monkeypatch.setattr(data_loader, "Queue", lambda **kwargs: Queue(maxsize=16))
    original_next = StreamDataLoader.__next__

    def consume(self):
        release.set()
        return original_next(self)

    monkeypatch.setattr(StreamDataLoader, "__next__", consume)
    loader = StreamDataLoader(config(tmp_path))
    model = FrameCounter()
    trainer = Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    try:
        assert ready.wait(10)
        trainer.predict(model, dataloaders=loader, return_predictions=False)
        assert sorted(model.seen) == list(range(37))
        assert not loader.thread.is_alive()
    finally:
        release.set()
        loader.stop()


def test_slow_producer_is_not_end_of_input(tmp_path, monkeypatch):
    path = tmp_path / "slow.png"
    Image.new("RGB", (32, 32)).save(path)
    original = StreamDataLoader.process_image

    def delayed(self, path):
        sleep(1.2)  # Longer than the old one-second EOF timeout.
        original(self, path)

    monkeypatch.setattr(StreamDataLoader, "process_image", delayed)
    loader = StreamDataLoader(config(path))
    try:
        assert len(list(loader)) == 1
        assert list(loader) == []
    finally:
        loader.stop()


@pytest.mark.parametrize("source", ["empty", "missing.png", "broken.png"])
def test_end_of_input_and_read_errors(tmp_path, source):
    path = tmp_path / source
    if source == "empty":
        path.mkdir()
    elif source == "broken.png":
        path.write_text("not an image")
    loader = StreamDataLoader(config(path))
    try:
        if source == "empty":
            assert list(loader) == []
        else:
            with pytest.raises(OSError):
                list(loader)
        assert not loader.thread.is_alive()
    finally:
        loader.stop()


def test_stop_releases_producer_with_full_queue(tmp_path, monkeypatch):
    full = Event()

    def produce(self, folder):
        for _ in range(100):
            if self.queue.full():
                full.set()
            if self.stop_event.is_set():
                break
            self.process_frame(Image.new("RGB", (32, 32)))

    monkeypatch.setattr(StreamDataLoader, "load_image_folder", produce)
    loader = StreamDataLoader(config(tmp_path))
    try:
        assert full.wait(10)
        loader.stop()
        assert not loader.thread.is_alive()
        assert loader.source_done.is_set()
        assert list(loader) == []
    finally:
        loader.stop()


@pytest.mark.parametrize("source", [0, "video.mp4"])
def test_video_and_live_inputs_consume_through_eof(tmp_path, monkeypatch, source):
    import cv2

    frames = iter(np.full((32, 32, 3), index, dtype=np.uint8) for index in range(19))
    released = Event()

    def read():
        frame = next(frames, None)
        return frame is not None, frame

    monkeypatch.setattr(cv2, "VideoCapture", lambda source: SimpleNamespace(read=read, release=released.set))
    cfg = config(tmp_path / "video.mp4")
    cfg.source = source if isinstance(source, int) else cfg.source
    loader = StreamDataLoader(cfg)
    try:
        assert [frame.getpixel((0, 0))[0] for _, _, frame in loader] == list(range(19))
        assert released.is_set()
    finally:
        loader.stop()


@pytest.mark.parametrize("fails", [False, True])
def test_cli_discards_predictions_and_stops_loader(tmp_path, monkeypatch, fails):
    import yolo.lazy as lazy

    calls = []

    def predict(model, **kwargs):
        assert kwargs == {"return_predictions": False}
        calls.append("predict")
        if fails:
            raise RuntimeError("prediction failed")

    model = SimpleNamespace(predict_loader=SimpleNamespace(stop=lambda: calls.append("stop")))
    monkeypatch.setattr(lazy, "set_seed", lambda seed: None)
    monkeypatch.setattr(lazy, "resolve_training_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(lazy, "setup", lambda *args, **kwargs: ([], [], tmp_path))
    monkeypatch.setattr(lazy, "Trainer", lambda **kwargs: SimpleNamespace(predict=predict))
    monkeypatch.setattr(lazy, "InferenceModel", lambda cfg: model)
    cfg = SimpleNamespace(task=SimpleNamespace(task="inference"), lucky_number=10, device="auto")
    if fails:
        with pytest.raises(RuntimeError, match="prediction failed"):
            lazy.main.__wrapped__(cfg)
    else:
        lazy.main.__wrapped__(cfg)
    assert calls == ["predict", "stop"]
