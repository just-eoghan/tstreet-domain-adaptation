import datetime
import os
from typing import Optional, Tuple

import albumentations as A
import boto3
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils import data
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CocoDetection
from tqdm import tqdm

s3 = boto3.resource("s3")
cw = boto3.client("cloudwatch")


class Collater:
    # https://shoarora.github.io/2020/02/01/collate_fn.html
    def __call__(self, batch):
        return tuple(zip(*batch))


class DetectDataModule(LightningDataModule):
    """
    LightningDataModule for drone danbuoy 1920x1080.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        images_directory: str = "data/danbuoy.data.v1/images",
        annotations_path: str = "data/danbuoy.data.v1/danbuoy_v1_annotations.json",
        val_split_percent: int = 30,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes: int = 2,
        image_width: int = 160,
        image_height: int = 120
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transforms = A.Compose(
            [
                A.Resize(self.hparams.image_height,self.hparams.image_width),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco", label_fields=["category_ids"]
            ),
        )

        self.notransforms = A.Compose(
            [
                A.Resize(self.hparams.image_height,self.hparams.image_width),
                A.Normalize(),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(
                format="coco", label_fields=["category_ids"]
            ),
        )

        self.collater = Collater()

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.hparams.image_width, self.hparams.image_height)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val:
            dataset = DetectDataset(
                self.hparams.data_dir + "images",
                self.hparams.data_dir + "danbuoy_v1_annotations.json",
                transform=self.transforms,
            )

            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=[(1 - (self.hparams.val_split_percent/100)), self.hparams.val_split_percent/100],
                generator=torch.Generator().manual_seed(42),
            )

            self.data_val.dataset.transfrom = self.notransforms

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            # https://github.com/pytorch/vision/issues/2624#issuecomment-681811444
            collate_fn=self.collater,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collater,
        )