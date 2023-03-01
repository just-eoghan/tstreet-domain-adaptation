from dataclasses import dataclass
from typing import List, Optional

import os
import PIL
from functools import partial
from flash.image import ObjectDetectionData, ObjectDetector
import flash
import albumentations as A
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningModule,
    Trainer,
    seed_everything,
)

from pytorch_lightning.loggers import Logger
from flash.core.integrations.icevision.transforms import IceVisionTransformAdapter
from icevision.tfms import A

from src import utils
from albumentations.pytorch.transforms import ToTensorV2
from flash.core.data.transforms import ApplyToKeys

from src.datamodules.datasets.detect_dataset import DetectDataset
from torch import nn
import torchvision
import numpy as np

log = utils.get_logger(__name__)

transform = A.Compose([
        A.Resize(256, 256)
    ], bbox_params=A.BboxParams(
    format="pascal_voc"
))

@dataclass
class ResizeTransform(flash.InputTransform):
    def per_sample_transform(self):
        return IceVisionTransformAdapter(
            [transform]
        )

def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: ObjectDetectionData = hydra.utils.call(config.datamodule)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    # Send some parameters from config to all lightning loggers
    # Train the model
    log.info("Starting training!")

    # datamodule = ObjectDetectionData.from_coco(
    #     train_folder=os.path.dirname(os.path.abspath(__file__))[:-3] + "data/streets_griffith/images",
    #     train_ann_file=os.path.dirname(os.path.abspath(__file__))[:-3] + "data/streets_griffith/annotations.json",
    #     val_folder=os.path.dirname(os.path.abspath(__file__))[:-3] + "data/streets_thomas/images",
    #     val_ann_file=os.path.dirname(os.path.abspath(__file__))[:-3] + "data/streets_thomas/annotations.json",
    #     transform_kwargs=dict(image_size=(256, 256)),
    #     batch_size=24
    # )

    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    # Make sure everything closed properly|
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score
