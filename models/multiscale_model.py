import cv2
import numpy as np
import ultralytics
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.cfg import get_cfg
import torch
from torchvision.ops import nms as torch_nms
from ensemble_boxes import *
from .model import DetectionModel as BaseDetectionModel


class DetectionModel(BaseDetectionModel):
    def __init__(self,
                 model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model(self.model_path)
        self.device = device
    
    def _load_model(self, model_path):
        self.model = ultralytics.YOLO(model_path)
        return self.model
    
    def image_detect(self,
                     image,
                     scales=[1.0],
                     crop_ratio=0.65,
                     weights = [
                            0.75,  # original image
                            1.0,  # top-left
                            1.0,  # top-right
                            1.0,  # bottom-left
                            1.0,  # bottom-right
                            2.0   # center
                        ]
                    ,
                    conf_threshold=None,
                    iou_threshold=None
                     ):
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        # print(f"Running detection on image with conf: {conf_threshold}, iou: {iou_threshold}")
        detections = {
            "boxes": [],
            "scores": [],
            "labels": []
        }
        
        h, w = image.shape[:2]
        
        batch_images = []
        batch_metadata = []
        total_weights = []
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(image, (new_w, new_h))
            
            # ảnh gốc
            batch_images.append(scaled_image)
            batch_metadata.append({
                'scale': scale,
                'crop_size': (new_w, new_h),
                'offset': (0, 0),
                'original_size': (w, h)
            })
            
            crop_w = int(new_w * crop_ratio)
            crop_h = int(new_h * crop_ratio)
                
            crop_positions = [
                (0, 0),
                (new_w - crop_w, 0),
                (0, new_h - crop_h),
                (new_w - crop_w, new_h - crop_h),
                ((new_w - crop_w) // 2, (new_h - crop_h) // 2)
            ]
            
            
            # print(f"Processing scale: {scale}, new size: ({new_w}, {new_h})")
            # print(f"Crop size: ({crop_w}, {crop_h}), positions: {crop_positions}")
            
            for idx, (x_offset, y_offset) in enumerate(crop_positions):
                crop = scaled_image[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                batch_images.append(crop)
                batch_metadata.append({
                    'scale': scale,
                    'crop_size': (crop_w, crop_h),
                    'offset': (x_offset, y_offset),
                    'original_size': (w, h)
                })
                
        # 3. Run batch inference
        batch_results = self.model.predict(
            batch_images,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            device=self.device,
            stream=True,
        )
        
        
        detections_per_view = []
        
        for idx, (result, metadata) in enumerate(zip(batch_results, batch_metadata)):
            # print(f"Processing result {idx + 1}/{len(batch_metadata)}")
            # if result.boxes is None:
            #     print("0 boxes detected, skipping...")
            # else:
            #     print(f"Detected {len(result.boxes)} boxes")
            if result.boxes is None or len(result.boxes) == 0:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            
            scale = metadata['scale']
            x_offset, y_offset = metadata['offset']
            w_orig, h_orig = metadata['original_size']
            
            # Step 1: Add offset to convert from crop coordinates to scaled image coordinates
            boxes[:, 0] = boxes[:, 0] + x_offset
            boxes[:, 1] = boxes[:, 1] + y_offset
            boxes[:, 2] = boxes[:, 2] + x_offset
            boxes[:, 3] = boxes[:, 3] + y_offset
            
            # Step 2: Convert from scaled image to original image and normalize
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] / scale) / w_orig
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] / scale) / h_orig
            
            boxes = np.clip(boxes, 0, 1)
            
            if len(boxes) > 0:
                detections_per_view.append({
                    'boxes': boxes.tolist(),
                    'scores': scores.tolist(),
                    'labels': labels.tolist()
                })
                
                total_weights.append(weights[idx])
            # print(f"View {idx + 1}/{len(batch_metadata)}: Detected {len(boxes)} boxes with weight {weights[idx]}")

        # 5. Apply Non-Maximum Weighted fusion
        if detections_per_view:
            boxes_list = [det['boxes'] for det in detections_per_view]
            scores_list = [det['scores'] for det in detections_per_view]
            labels_list = [det['labels'] for det in detections_per_view]

            # Apply Weighted Boxes Fusion
            wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=total_weights,
                iou_thr=iou_threshold,
                skip_box_thr=conf_threshold
            )
            #filter out boxes with low scores
            wbf_keep = wbf_scores > conf_threshold
            # wbf_boxes = wbf_boxes.tolist()
            # wbf_scores = wbf_scores.tolist()
            # wbf_labels = wbf_labels.tolist()
            final_boxes = wbf_boxes[wbf_keep]
            final_scores = wbf_scores[wbf_keep]
            final_labels = wbf_labels[wbf_keep]

            nms_keep_indices = torch_nms(
                torch.tensor(final_boxes),
                torch.tensor(final_scores),
                iou_threshold
            )
            final_boxes = [final_boxes[i] for i in nms_keep_indices]
            final_scores = [final_scores[i] for i in nms_keep_indices]
            final_labels = [final_labels[i] for i in nms_keep_indices]
            
            # if len(final_labels) > 1:
            #     print("======= Results =======")
            #     print(f"Before WBF: {len(boxes_list)} views")
            #     for i, boxes in enumerate(boxes_list):
            #         print(f"View {i + 1}: {len(boxes)} boxes")
            #         print(boxes)
            #         print(scores_list[i])
            #         print(labels_list[i])
            #     print(f"After WBF: {len(wbf_boxes)} boxes")
            #     print(wbf_boxes)
            #     print(wbf_scores)
            #     print(wbf_labels)
            #     print(wbf_keep)
            #     print(nms_keep_indices)
            #     print(f"After NMS: {len(final_boxes)} boxes")
            #     print(final_boxes)
            #     print(final_scores)
            #     print(final_labels)
            #     for i, box in enumerate(final_boxes):
            #         for j, box2 in enumerate(final_boxes):
            #             if i < j:
            #                 iou = self.bb_intersection_over_union(box, box2)
            #                 print(f"Box {i} and Box {j} have IoU {iou:.2f}")

            for box, score, label in zip(final_boxes, final_scores, final_labels):
                x1 = box[0] * w
                y1 = box[1] * h
                x2 = box[2] * w
                y2 = box[3] * h
                detections["boxes"].append([x1, y1, x2, y2])
                detections["scores"].append(score)
                detections["labels"].append(label)

        return detections
    
    def video_detect(self,
                    video_path,
                    scales=[1.0],
                    crop_ratio=0.65,
                    weights = [
                            0.75,  # original image
                            1.0,  # top-left
                            1.0,  # top-right
                            1.0,  # bottom-left
                            1.0,  # bottom-right
                            2.0   # center
                        ]
                    ,
                    conf_threshold=None,
                    iou_threshold=None
                     
                     ) -> list:
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
            
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            det = self.image_detect(frame,
                                    scales=scales,
                                    crop_ratio=crop_ratio,
                                    weights=weights,
                                    conf_threshold=conf_threshold,
                                    iou_threshold=iou_threshold
                                )
            
            frames.append(det)
            # if len(frames) >= 30:
            #     break
            
        cap.release()
        return frames

    
    def bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
