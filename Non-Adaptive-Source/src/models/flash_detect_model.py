from pytorch_lightning import LightningModule
import torch

from flash.core.data.io.input import DataKeys

from flash.image import ObjectDetector
from icevision.models.ross import efficientdet
from icevision.models.ultralytics import yolov5
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP



class FlashDetectLitModel(ObjectDetector):
    def __init__(
        self,
        num_classes,
        backbone,
        head,
        learning_rate,
        optimizer,
        image_size=None
    ):
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            head=head,
            learning_rate=learning_rate,
            optimizer=optimizer,
            image_size=image_size
        )
        self.save_hyperparameters()
        self.val_map = MAP().to(torch.device("cuda", 0))
        self.test_map = MAP().to(torch.device("cuda", 0))

    def validation_step(self, batch, batch_idx):
        (xb, yb), records = batch[DataKeys.INPUT]

        if self.hparams.head == "faster_rcnn":
            with torch.no_grad():
                raw_preds = self.adapter.model(xb)
                preds = self.adapter.icevision_adapter.convert_raw_predictions(
                    batch=batch, raw_preds=raw_preds, records=records
                )

        if self.hparams.head == "retinanet":
            with torch.no_grad():
                raw_preds = self.adapter.model(xb)
                preds = self.adapter.icevision_adapter.convert_raw_predictions(
                    batch=batch, raw_preds=raw_preds, records=records
                )

        if self.hparams.head == 'yolov5':
            with torch.no_grad():
                inference_out, training_out = self.adapter.model(xb)
                preds = yolov5.convert_raw_predictions(
                    batch=xb,
                    raw_preds=inference_out,
                    records=records,
                    detection_threshold=0.001,
                    nms_iou_threshold=0.6,
                )
            
        if self.hparams.head == 'efficientdet':
            with torch.no_grad():
                raw_preds = self.adapter.model(xb, yb)
                preds = efficientdet.convert_raw_predictions(
                    batch=(xb, yb),
                    raw_preds=raw_preds["detections"],
                    records=records,
                    detection_threshold=0.0,
                )

            self.adapter.icevision_adapter.accumulate_metrics(preds)

            for k, v in raw_preds.items():
                if "loss" in k:
                    self.log(f"valid/{k}", v)

        with torch.no_grad():
            tensor_preds = [pred.as_dict()["detection"] for pred in preds]
            map_preds = []
            for pred_obj in tensor_preds:
                pred = {}
                pred["scores"] = torch.tensor(pred_obj["scores"])
                pred["labels"] = torch.tensor(pred_obj["label_ids"])
                if len(pred_obj["bboxes"]) == 0:
                    pred["boxes"] = torch.tensor([])
                else:
                    pred["boxes"] = torch.stack([torch.tensor(box.xyxy) for box in pred_obj["bboxes"]])
                map_preds.append(pred)

            batch_recs = [record.as_dict()["detection"] for record in records]
            targets = []
            for batch_obj in batch_recs:
                target = {}
                target["labels"] = torch.tensor(batch_obj["label_ids"])
                target["boxes"] = torch.stack([torch.tensor(box.xyxy) for box in batch_obj["bboxes"]])
                targets.append(target)

            self.val_map.update(map_preds, targets)
    
    def validation_epoch_end(self, outputs) -> None:
        val_map = self.val_map.compute()
        self.log("val/map50", val_map["map_50"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/map", val_map, on_step=False, on_epoch=True, prog_bar=False)
        self.val_map.reset()

    def test_step(self, batch, batch_idx):
        (xb, yb), records = batch[DataKeys.INPUT]
        if type(xb) is not list:
            test_images = torch.unbind(xb)
        else:
            test_images = xb
        test_targets = []
        for ice_record in records:
            gt = ice_record.as_dict()['detection']
            gt_boxes = []
            for box in gt['bboxes']:
                gt_boxes.append(box.to_tensor())
            test_targets.append(
                {
                    "labels": torch.tensor(gt['label_ids']).to(self.device),
                    "boxes" : torch.stack(gt_boxes).to(self.device)
                }
            )

        if self.hparams.head == "faster_rcnn":
            with torch.no_grad():
                raw_preds = self.adapter.model(xb)
                preds = self.adapter.icevision_adapter.convert_raw_predictions(
                    batch=batch, raw_preds=raw_preds, records=records
                )
                test_outs = raw_preds

        if self.hparams.head == "retinanet":
            with torch.no_grad():
                raw_preds = self.adapter.model(xb)
                preds = self.adapter.icevision_adapter.convert_raw_predictions(
                    batch=batch, raw_preds=raw_preds, records=records
                )
                test_outs = raw_preds

        if self.hparams.head == 'yolov5':
            with torch.no_grad():
                inference_out, training_out = self.adapter.model(xb)
                preds = yolov5.convert_raw_predictions(
                    batch=xb,
                    raw_preds=inference_out,
                    records=records,
                    detection_threshold=0.1,
                    nms_iou_threshold=0.6,
                )

                test_outs = []
                for ice_pred in preds:
                    pred = ice_pred.pred.as_dict()['detection']
                    boxes = []
                    for box in pred['bboxes']:
                        boxes.append(box.to_tensor())
                    if(len(boxes) > 0):
                        torch_boxes = torch.stack(boxes).to(self.device)
                    else:
                        torch_boxes = torch.tensor([]).to(self.device)
                    test_outs.append(
                        {
                            "labels": torch.tensor(pred['label_ids']).to(self.device),
                            "scores": torch.tensor(pred['scores']).to(self.device),
                            "boxes" : torch_boxes
                        }
                    )

            
        if self.hparams.head == 'efficientdet':
            with torch.no_grad():
                raw_preds = self.adapter.model(xb, yb)
                preds = efficientdet.convert_raw_predictions(
                    batch=(xb, yb),
                    raw_preds=raw_preds["detections"],
                    records=records,
                    detection_threshold=0.1,
                )
                test_outs = []
                for ice_pred in preds:
                    pred = ice_pred.pred.as_dict()['detection']
                    boxes = []
                    for box in pred['bboxes']:
                        boxes.append(box.to_tensor())
                    if(len(boxes) > 0):
                        torch_boxes = torch.stack(boxes).to(self.device)
                    else:
                        torch_boxes = torch.tensor([]).to(self.device)
                    test_outs.append(
                        {
                            "labels": torch.tensor(pred['label_ids']).to(self.device),
                            "scores": torch.tensor(pred['scores']).to(self.device),
                            "boxes" : torch_boxes
                        }
                    )
            self.adapter.icevision_adapter.accumulate_metrics(preds)

        with torch.no_grad():
            tensor_preds = [pred.as_dict()["detection"] for pred in preds]
            map_preds = []
            for pred_obj in tensor_preds:
                pred = {}
                pred["scores"] = torch.tensor(pred_obj["scores"])
                pred["labels"] = torch.tensor(pred_obj["label_ids"])
                if len(pred_obj["bboxes"]) == 0:
                    pred["boxes"] = torch.tensor([])
                else:
                    pred["boxes"] = torch.stack([torch.tensor(box.xyxy) for box in pred_obj["bboxes"]])
                map_preds.append(pred)

            batch_recs = [record.as_dict()["detection"] for record in records]
            targets = []
            for batch_obj in batch_recs:
                target = {}
                target["labels"] = torch.tensor(batch_obj["label_ids"])
                target["boxes"] = torch.stack([torch.tensor(box.xyxy) for box in batch_obj["bboxes"]])
                targets.append(target)

            self.test_map.update(map_preds, targets)
        return {
            "test_images": test_images,
            "test_gt": test_targets,
            "test_outs": test_outs
        }
    
    def test_epoch_end(self, outputs) -> None:
        test_map = self.test_map.compute()
        self.log("test/map50", test_map["map_50"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/map", test_map, on_step=False, on_epoch=True, prog_bar=False)
        self.test_map.reset()