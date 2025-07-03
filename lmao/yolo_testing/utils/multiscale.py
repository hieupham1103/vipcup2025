import cv2
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import *
import torch

class YOLOMS:
    def __init__(
        self,
        model: YOLO,
        segment_ratio: float = 0.55,
        iou_thr: float = 0.2,
        conf_thr: float = 0.001,
        weight = None,
        device = 'cpu'
    ):
        """
        model     : một instance của ultralytics.YOLO đã load weights
        segment_ratio: tỉ lệ để crop mỗi vùng (mặc định 55%)
        iou_thr   : IoU threshold cho soft-NMS
        conf_thr  : confidence threshold để lọc box trước khi soft-NMS
        """
        self.model = model
        self.segment_ratio = segment_ratio
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr
        self.weight = weight
        self.device = device

    def _get_crops(self, image: np.ndarray):
        """
        Trả về list các vùng crop và tọa độ gốc của chúng trong ảnh.
        """
        h, w = image.shape[:2]
        sw, sh = int(w * self.segment_ratio), int(h * self.segment_ratio)
        xs = [0, w - sw]
        ys = [0, h - sh]

        crops, boxes = [], []
        # for y in ys:
        #     for x in xs:
        #         x2, y2 = x + sw, y + sh
        #         crops.append(image[y:y2, x:x2])
        #         boxes.append((x, y, x2, y2))
        
        cx = (w - sw) // 2
        cy = (h - sh) // 2
        crops.append(image[cy:cy+sh, cx:cx+sw])
        boxes.append((cx, cy, cx+sw, cy+sh))
        return crops, boxes

    def _run_inference(self, images):
        """
        Chạy YOLO trên batch ảnh, tắt NMS nội bộ để lấy raw predictions.
        """
        if isinstance(images, np.ndarray):
            images = [images]
        # tắt NMS, giữ raw boxes
        results = self.model.predict(images,
                                     iou=1.0,
                                     imgsz=[320, 256],
                                     conf=self.conf_thr,
                                     verbose=False,
                                     device=self.device,
                                     batch=16,
                                     )

        preds = []
        for res in results:
            boxes = res.boxes
            preds.append((
                boxes.xyxy.cpu().numpy(),               # [N,4]
                boxes.conf.cpu().numpy(),               # [N,]
                boxes.cls.cpu().numpy().astype(int)     # [N,]
            ))
        return preds

    def _remap_crops(self, crop_preds, crop_boxes):
        """
        Dịch trực tiếp tọa độ trong crop_preds, rồi trả về crop_preds luôn.
        Mỗi phần tử crop_preds là (boxes, scores, labels), ta chỉ +offset lên boxes.
        """
        for idx, ((boxes, scores, labels), (x0, y0, _, _)) in enumerate(zip(crop_preds, crop_boxes)):
            boxes += np.array([x0, y0, x0, y0], dtype=boxes.dtype)
            crop_preds[idx] = (boxes, scores, labels)

        return crop_preds


    def _apply_nms(self, all_preds, image_shape):
        """
        Áp dụng soft-NMS (ensemble-boxes) cho toàn bộ box đã gộp.
        Đầu vào: boxes pixel-format [x1,y1,x2,y2], scores, labels
        """
        boxes_list = []
        scores_list = []
        labels_list = []

        height, width = image_shape[:2]
        weights = []
        for idx, (pred_boxes, pred_scores, pred_labels) in enumerate(all_preds):
            if len(pred_boxes) == 0:
                continue
                
            norm_boxes = pred_boxes.copy()
            norm_boxes[:, 0] /= width
            norm_boxes[:, 2] /= width
            norm_boxes[:, 1] /= height
            norm_boxes[:, 3] /= height
            if not self.weight is None:
                weights.append(self.weight[idx] if idx < len(self.weight) else 1)
            boxes_list.append(norm_boxes.tolist())
            scores_list.append(pred_scores.tolist())
            labels_list.append(pred_labels.tolist())
            # print(f"Crop {idx}: {len(pred_boxes)} boxes, {len(pred_scores)} scores, {len(pred_labels)} labels")
        # print("Boxes:", len(boxes_list), "Scores:", len(scores_list), "Labels:", len(labels_list))
        if len(boxes_list) == 0:
            return [], [], []
        nms_boxes, nms_scores, nms_labels = nms(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=self.iou_thr
        )
        # nms_boxes, nms_scores, nms_labels = soft_nms(
        #     boxes_list,
        #     scores_list,
        #     labels_list,
        #     weights=weights,
        #     iou_thr=self.iou_thr,
        #     sigma=0.1,
        #     thresh=self.conf_thr
        # )
        # nms_boxes, nms_scores, nms_labels = non_maximum_weighted(
        #     boxes_list,
        #     scores_list,
        #     labels_list,
        #     weights=weights,
        #     iou_thr=self.iou_thr,
        #     skip_box_thr=self.conf_thr
        # )
        # nms_boxes, nms_scores, nms_labels = weighted_boxes_fusion(
        #     boxes_list,
        #     scores_list,
        #     labels_list,
        #     iou_thr=self.iou_thr,
        #     skip_box_thr=self.conf_thr,
        #     weights=weights,
        # )


        # Chuyển ngược về pixel-format
        out_boxes = []
        for bx in nms_boxes:
            x1, y1, x2, y2 = bx
            out_boxes.append([
                x1 * width,
                y1 * height,
                x2 * width,
                y2 * height
            ])

        return out_boxes, nms_scores, nms_labels

    def detect(self, image: np.ndarray, device=None):
        """
        Chạy multi-scale inference:
        1. Crop 4 vùng + toàn ảnh.
        2. Lấy raw preds từ YOLO (nms=False).
        3. Remap coords, gộp preds.
        4. Áp dụng soft-NMS cuối.
        """
        if device is not None:
            self.device = device
        # 1. Sinh crops
        crops, crop_boxes = self._get_crops(image)
        images = [image] + crops

        # 2. Inference raw
        all_preds = self._run_inference(images)
        crop_preds = all_preds[1:]
        # 3. Remap crop preds → toạ độ gốc
        crop_preds = self._remap_crops(crop_preds, crop_boxes)
        all_preds = all_preds[0:1] + crop_preds
        
        # all_preds = self._run_inference(crops)
        # all_preds = self._remap_crops(all_preds, crop_boxes)
        

        # all_preds = all_preds[0:1] + crop_preds
        return self._apply_nms(all_preds, image.shape)
