import cv2
import numpy as np
import ultralytics
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.cfg import get_cfg
import torch
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
    
    def image_detect(self, image):
        detections = {
            "boxes": [],
            "scores": [],
            "labels": []
        }
        
        h, w = image.shape[:2]
        
        scales = [0.8, 1.0, 1.2]
        crop_ratio = 0.6
        
        batch_images = []
        batch_metadata = []
        
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(image, (new_w, new_h))
            
            batch_images.append(scaled_image)
            batch_metadata.append({
                'scale': scale,
                'crop_size': (new_w, new_h),
                'offset': (0, 0),
                'weight': 2.0 if scale == 1.0 else 1.0,
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
            
            for x_offset, y_offset in crop_positions:
                crop = scaled_image[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                batch_images.append(crop)
                batch_metadata.append({
                    'scale': scale,
                    'crop_size': (crop_w, crop_h),
                    'offset': (x_offset, y_offset),
                    'weight': 1.8 if scale == 1.0 else 1.2,
                    'original_size': (w, h)
                })
        
        # 3. Run batch inference
        print(f"Processing batch of {len(batch_images)} images ({len(scales)} scales × 6 images each)...")
        batch_results = self.model.predict(
            batch_images,
            conf=self.conf_threshold,
            iou=0.01,
            verbose=False,
            device=self.device,
            stream=True,
        )
        
        all_detections = []
        
        for i, (result, metadata) in enumerate(zip(batch_results, batch_metadata)):
            if result.boxes is None or len(result.boxes) == 0:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            
            # UNIFIED coordinate conversion
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
                all_detections.append({
                    'boxes': boxes.tolist(),
                    'scores': scores.tolist(),
                    'labels': labels.tolist(),
                    'weight': metadata['weight']
                })
        
        # 5. Apply Non-Maximum Weighted fusion
        if all_detections:
            boxes_list = [det['boxes'] for det in all_detections]
            scores_list = [det['scores'] for det in all_detections]
            labels_list = [det['labels'] for det in all_detections]
            weights = [det['weight'] for det in all_detections]
            
            # Apply Non-Maximum Weighted
            final_boxes, final_scores, final_labels = non_maximum_weighted(
                boxes_list, 
                scores_list, 
                labels_list, 
                weights=weights, 
                iou_thr=self.iou_threshold,
                skip_box_thr=0.01
            )
            
            # Convert back to pixel coordinates and tensor format
            for i in range(len(final_boxes)):
                box = final_boxes[i]
                pixel_box = [
                    box[0] * w,  # x1
                    box[1] * h,  # y1
                    box[2] * w,  # x2
                    box[3] * h   # y2
                ]
                
                detections["boxes"].append(torch.tensor(pixel_box))
                detections["scores"].append(torch.tensor(final_scores[i]))
                detections["labels"].append(torch.tensor(final_labels[i]))
        
        return detections

    
class TrackingModel:
    def __init__(self,
                 detection_model: DetectionModel,
                 config_path='configs/bytetrack.yml',
                 fps = 25
                 ):
        self.det_model = detection_model
        
        self.tracker = BYTETracker(
                            args=get_cfg(config_path),
                            frame_rate=fps
                            )
    
    def video_track(self, video_path: str) -> list:
        cap = cv2.VideoCapture(video_path)
        tracked_frames = []

        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            det = self.det_model.image_detect(frame)
            
            boxes = np.array(det["boxes"]) if det["boxes"] else np.zeros((0, 4))
            scores = np.array(det["scores"]) if det["scores"] else np.zeros((0,))
            labels = np.array(det["labels"]) if det["labels"] else np.zeros((0,), dtype=int)

            dets = np.concatenate([boxes, scores.reshape(-1, 1)], axis=1) if boxes.size else np.zeros((0, 5))

            online_targets = self.tracker.update(dets, [frame.shape[0], frame.shape[1]])

            tracks = []
            for t in online_targets:
                x1, y1, x2, y2, track_id = t[:5]
                tracks.append([int(x1), int(y1), int(x2), int(y2), int(track_id)])

            tracked_frames.append({
                "frame": frame,
                "tracks": tracks,
                
            })

        cap.release()
        return tracked_frames