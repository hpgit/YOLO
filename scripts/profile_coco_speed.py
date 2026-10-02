"""Synchronized diagnostic stage timings on the fixed COCO subset (not throughput)."""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source_parser = argparse.ArgumentParser(add_help=False)
source_parser.add_argument("--source-root", type=Path, default=ROOT)
source_args, _ = source_parser.parse_known_args()
sys.path.insert(0, str(source_args.source_root.resolve()))
import torch
from hydra import compose, initialize_config_dir
from lightning import seed_everything
from torchmetrics.detection import MeanAveragePrecision

from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import Vec2Box, bbox_nms, to_metrics_format
from yolo.utils.model_utils import EMA, create_optimizer


def timed(totals, name, fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    totals[name] += time.perf_counter() - start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/performance-round2/profile.json")
    parser.add_argument("--state", type=Path, default=ROOT / "runs/performance/baseline-1/ema.pt")
    parser.add_argument("--train-batches", type=int, default=30)
    parser.add_argument(
        "--metric-cpu", action="store_true", help="Store metric states on CPU to avoid per-detection CUDA scalar copies"
    )
    args = parser.parse_args()
    seed_everything(10, workers=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with initialize_config_dir(config_dir=str(args.source_root.resolve() / "yolo/config"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=train",
                "model=v9-t",
                "dataset=coco-subset",
                "weight=false",
                "cpu_num=4",
                "image_size=[320,320]",
                "task.data.batch_size=32",
                "task.validation.data.batch_size=32",
                "+model.anchor.strides=[8,16,32]",
            ],
        )
    model = create_model(cfg.model, weight_path=False).cuda()
    model.load_state_dict(torch.load(args.state, map_location="cuda", weights_only=True))
    model.eval()
    converter = Vec2Box(model, cfg.model.anchor, cfg.image_size, torch.device("cuda"))
    loader = create_dataloader(cfg.task.validation.data, cfg.dataset, "validation")
    metric = MeanAveragePrecision(
        iou_type="bbox", box_format="xyxy", backend="faster_coco_eval", compute_on_cpu=args.metric_cpu
    ).cuda()
    metric.warn_on_many_detections = False
    val = defaultdict(float)
    candidates, images_seen = [], 0
    # Warm up model kernels, without running BN in training mode.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            model(torch.zeros(32, 3, 320, 320, device="cuda"), shortcut="Main")
        iterator = timed(val, "loader_start", lambda: iter(loader))
        for index in range(len(loader)):
            batch = timed(val, "data_wait", lambda: next(iterator))
            _, images, targets, *_ = batch
            images, targets = timed(
                val, "transfer", lambda: (images.cuda(non_blocking=True), targets.cuda(non_blocking=True))
            )
            output = timed(val, "forward", lambda: model(images, shortcut="Main"))
            cls, _, boxes = timed(val, "decode", lambda: converter(output["Main"]))
            preds = timed(val, "nms", lambda: bbox_nms(cls, boxes, cfg.task.validation.nms))
            # Candidate counting and disk capture are outside every measured stage.
            candidates.append(int((cls.sigmoid() > cfg.task.validation.nms.min_confidence).sum()))
            cache = args.output.parent / "nms-inputs"
            cache.mkdir(parents=True, exist_ok=True)
            torch.save({"cls": cls.cpu(), "boxes": boxes.cpu()}, cache / f"{index:02d}.pt")
            formatted = timed(
                val,
                "metric_format",
                lambda: ([to_metrics_format(p) for p in preds], [to_metrics_format(t) for t in targets]),
            )
            timed(val, "metric_update", lambda: metric.update(*formatted))
            images_seen += batch[0]
        metrics = timed(val, "metric_compute", metric.compute)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "val_images": images_seen,
        "metric_cpu": args.metric_cpu,
        "state_sha256": hashlib.sha256(args.state.read_bytes()).hexdigest(),
        "validation_stage_seconds": dict(val),
        "nms_candidates_per_batch": candidates,
        "map": float(metrics["map"]),
    }
    if args.train_batches:
        model.train()
        train_loader = create_dataloader(cfg.task.data, cfg.dataset, "train")
        loss_fn = create_loss_function(cfg, converter)
        optimizer = create_optimizer(model, cfg.task.optimizer)
        scaler = torch.amp.GradScaler("cuda")
        ema = EMA()
        ema.ema_state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}
        from types import SimpleNamespace

        module = SimpleNamespace(model=model)
        train = defaultdict(float)
        iterator = iter(train_loader)
        for step in range(args.train_batches + 5):
            current = train if step >= 5 else defaultdict(float)
            batch = timed(current, "data_wait", lambda: next(iterator))
            _, images, targets, *_ = batch
            images, targets = timed(
                current, "transfer", lambda: (images.cuda(non_blocking=True), targets.cuda(non_blocking=True))
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = timed(current, "forward", lambda: model(images))
                aux, main = timed(current, "decode", lambda: (converter(output["AUX"]), converter(output["Main"])))
                loss, _ = timed(current, "loss", lambda: loss_fn(aux, main, targets))
                loss = loss * batch[0]
            timed(current, "backward", lambda: scaler.scale(loss).backward())

            def update():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                scaler.step(optimizer)
                scaler.update()

            scale_before = scaler.get_scale()
            timed(current, "optimizer", update)
            if hasattr(ema, "update"):
                # GradScaler leaves parameters unchanged when it skips an overflow.
                if scaler.get_scale() >= scale_before:
                    timed(current, "ema", lambda: ema.update(module))
            else:
                timed(
                    current, "ema", lambda: ema.on_train_batch_end(SimpleNamespace(accumulate_grad_batches=1), module)
                )
        result["train_batches_after_warmup"] = args.train_batches
        result["training_stage_seconds"] = dict(train)
        result["training_stage_ms_per_batch"] = {key: value / args.train_batches * 1000 for key, value in train.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
