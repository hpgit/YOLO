import sys
from copy import deepcopy
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel
from yolo.utils.checkpoint_utils import resolve_training_checkpoint
from yolo.utils.logger import logger
from yolo.utils.logging_utils import set_seed, setup


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    # Seed before constructing models, datasets, callbacks, or the Trainer.
    set_seed(cfg.lucky_number)
    if cfg.task.task == "export":
        from yolo.tools.export import export_model

        return export_model(cfg)

    overrides = HydraConfig.get().overrides.task if HydraConfig.initialized() else []
    weight_explicit = any(item.lstrip("+").split("=", 1)[0] == "weight" for item in overrides)
    checkpoint_path = resolve_training_checkpoint(cfg, weight_explicit=weight_explicit)
    callbacks, loggers, save_path = setup(cfg, resume=checkpoint_path is not None)

    trainer = Trainer(
        accelerator=getattr(cfg, "accelerator", "auto"),
        devices=cfg.device,
        max_epochs=getattr(cfg.task, "epoch", None),
        precision="16-mixed",
        callbacks=callbacks,
        sync_batchnorm=True,
        logger=loggers,
        log_every_n_steps=1,
        deterministic=True,
        enable_progress_bar=not getattr(cfg, "quiet", False),
        default_root_dir=save_path,
    )

    if cfg.task.task == "train":
        model_cfg = deepcopy(cfg)
        if checkpoint_path is not None:
            model_cfg.weight = False
            logger.info(f"Resuming training from {checkpoint_path}")
        model = TrainModel(model_cfg)
        trainer.fit(model, ckpt_path=checkpoint_path)
    if cfg.task.task == "validation":
        model = ValidateModel(cfg)
        trainer.validate(model)
    if cfg.task.task == "inference":
        model = InferenceModel(cfg)
        try:
            # Predictions are displayed/saved per frame; do not retain all COCO
            # visualizations in memory for a return value the CLI never uses.
            trainer.predict(model, return_predictions=False)
        finally:
            model.predict_loader.stop()


if __name__ == "__main__":
    main()
