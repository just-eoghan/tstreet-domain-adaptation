import os
import os.path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.ops import box_convert


class FrcnnDataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Tuple[Image.Image, str]:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return (Image.open(os.path.join(self.root, path)).convert("RGB"), path)

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        id = self.ids[index]
        image, file_name = self._load_image(id)
        targets = self._load_target(id)

        bboxes = []
        category_ids = []

        for target in targets:
            category_ids.append(target["category_id"])
            bboxes.append(target["bbox"])

        # Handle the no gt image case
        if len(bboxes) < 1:
            bboxes = [[0, 1, 2, 3]]
            category_ids = [0]

        if self.transform is not None:
            transformed = self.transform(
                image=np.array(image), bboxes=bboxes, category_ids=category_ids
            )
            
            # faster-rcnn wants xyxy but we have xywh so convert
            for idx, tup in enumerate(transformed["bboxes"]):
                transformed["bboxes"][idx] = np.array(tup)

            if len(transformed["bboxes"]) < 1:
                transformed["bboxes"] = [[0, 1, 2, 3]]
                transformed["category_ids"] = [0]

            targets = {
                "boxes": box_convert(torch.FloatTensor(transformed["bboxes"]), "xywh", "xyxy"),
                "labels": torch.tensor(transformed["category_ids"], dtype=torch.int64),
            }

        return transformed["image"], targets, file_name

    def __len__(self) -> int:
        return len(self.ids)
