import cv2
import numpy as np
import ultralytics
import torch
from torchvision.ops import nms as torch_nms
from ensemble_boxes import *
from .model import DetectionModel as BaseDetectionModel
from collections import Counter, defaultdict


class KalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.kf = cv2.KalmanFilter(8, 4)
        
        # State transition matrix (position and velocity for x, y, w, h)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position and size)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.1
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        
        # Error covariance
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        
    def init_state(self, bbox):
        """Initialize Kalman filter with first detection"""
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.kf.statePre = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        
    def predict(self):
        """Predict next state"""
        prediction = self.kf.predict()
        x, y, w, h = prediction[0], prediction[1], prediction[2], prediction[3]
        return [x, y, x + w, y + h]
        
    def update(self, bbox):
        """Update with new measurement"""
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        measurement = np.array([x, y, w, h], dtype=np.float32)
        self.kf.correct(measurement)

class DetectionModel(BaseDetectionModel):
    def __init__(self,
                 model_path: str,
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.1,
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
        # print(type(image))
        if isinstance(image, np.ndarray):
            image = [image]
        h, w = image[0].shape[:2]
        
        batch_images = []
        batch_metadata = []
        total_weights = []
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = [cv2.resize(img, (new_w, new_h)) for img in image]
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
                crop = []
                for img in scaled_image:
                    crop.append(img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w])
                batch_images.append(crop)
                batch_metadata.append({
                    'scale': scale,
                    'crop_size': (crop_w, crop_h),
                    'offset': (x_offset, y_offset),
                    'original_size': (w, h)
                })
                
        # 3. Run batch inference\
        if len(batch_images[0]) == 1:
            batch_images = [img[0] for img in batch_images]
        else:
            RGB_batch_images = [img[0] for img in batch_images]
            IR_batch_images = [img[1] for img in batch_images]
            batch_images = [RGB_batch_images, IR_batch_images]
            
        batch_results = self.model.predict(
            batch_images,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            device=self.device,
            stream=True
        )
        
        detections_per_view = []
        
        # print(batch_images[0][1].shape, batch_images[0][1].shape)
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
                    iou_threshold=None,
                    min_detection_frames=4,
                    max_missing_frames=5
                ) -> list:
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        
        frames = []
        if isinstance(video_path, str):
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
            cap.release()
        elif isinstance(video_path, list):
            cap_1 = cv2.VideoCapture(video_path[0])
            cap_2 = cv2.VideoCapture(video_path[1])
            while cap_1.isOpened() or cap_2.isOpened():
                ret, frame_1 = cap_1.read()
                ret2, frame_2 = cap_2.read()
                if not ret and not ret2:
                    break
                det = self.image_detect([frame_1, frame_2],
                                        scales=scales,
                                        crop_ratio=crop_ratio,
                                        weights=weights,
                                        conf_threshold=conf_threshold,
                                        iou_threshold=iou_threshold
                                    )
                frames.append(det)
            cap_1.release()
            cap_2.release()
        else:
            raise ValueError("video_path must be a string or a list of strings")
        
        # final_frames = frames
        final_frames = self._postprocess_frames(frames, min_detection_frames, max_missing_frames, iou_threshold)

        return final_frames
    
    def _postprocess_frames(self, frames, min_detection_frames, max_missing_frames, iou_threshold):
        """Apply post-processing with Kalman filter, label correction, and spurious detection removal"""
        
        # Track storage
        tracks = {}  # track_id: {'kalman': KalmanFilter, 'last_seen': frame_idx, 'detections': [], 'labels': []}
        track_id_counter = 0
        
        # First pass: Build tracks and identify missing frames
        for frame_idx, frame_detections in enumerate(frames):
            matched_tracks = set()
            new_detections = []
            
            for i, (box, score, label) in enumerate(zip(
                frame_detections["boxes"], 
                frame_detections["scores"], 
                frame_detections["labels"]
            )):
                best_match_id = None
                best_iou = 0
                
                # Find best matching track
                for track_id, track_data in tracks.items():
                    if track_data['last_seen'] < frame_idx - max_missing_frames:
                        continue
                        
                    # Predict current position
                    predicted_box = track_data['kalman'].predict()
                    iou = self.bb_intersection_over_union(box, predicted_box)
                    
                    if iou > iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_match_id = track_id
                
                if best_match_id is not None:
                    # Update existing track
                    tracks[best_match_id]['kalman'].update(box)
                    tracks[best_match_id]['last_seen'] = frame_idx
                    tracks[best_match_id]['detections'].append(box)
                    tracks[best_match_id]['labels'].append(label)
                    tracks[best_match_id]['scores'].append(score)
                    tracks[best_match_id]['frame_indices'] = tracks[best_match_id].get('frame_indices', [])
                    tracks[best_match_id]['frame_indices'].append(frame_idx)
                    matched_tracks.add(best_match_id)
                else:
                    # New detection
                    new_detections.append((box, score, label))
            
            # Create new tracks for unmatched detections
            for box, score, label in new_detections:
                kf = KalmanFilter()
                kf.init_state(box)
                
                tracks[track_id_counter] = {
                    'kalman': kf,
                    'last_seen': frame_idx,
                    'detections': [box],
                    'labels': [label],
                    'scores': [score],
                    'frame_indices': [frame_idx]
                }
                track_id_counter += 1
        
        # Second pass: Process frames with smart interpolation
        processed_frames = []
        
        for frame_idx, frame_detections in enumerate(frames):
            current_detections = {
                "boxes": [],
                "scores": [],
                "labels": []
            }
            
            # Add actual detections first
            matched_tracks = set()
            
            for i, (box, score, label) in enumerate(zip(
                frame_detections["boxes"], 
                frame_detections["scores"], 
                frame_detections["labels"]
            )):
                best_match_id = None
                best_iou = 0
                
                # Find best matching track
                for track_id, track_data in tracks.items():
                    if frame_idx in track_data['frame_indices']:
                        # This detection belongs to this track
                        for track_box in track_data['detections']:
                            iou = self.bb_intersection_over_union(box, track_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_match_id = track_id
                
                if best_match_id is not None:
                    matched_tracks.add(best_match_id)
                    # Get corrected label (most common label in track)
                    corrected_label = Counter(tracks[best_match_id]['labels']).most_common(1)[0][0]
                    
                    current_detections["boxes"].append(box)
                    current_detections["scores"].append(score)
                    current_detections["labels"].append(corrected_label)
                else:
                    # Keep original detection if no track found
                    current_detections["boxes"].append(box)
                    current_detections["scores"].append(score)
                    current_detections["labels"].append(label)
            
            # Add interpolated detections for missing tracks (only if they reappear later)
            for track_id, track_data in tracks.items():
                if track_id not in matched_tracks and self._should_interpolate(track_id, frame_idx, tracks, max_missing_frames):
                    # Create a new Kalman filter for prediction from last known position
                    kf_pred = KalmanFilter()
                    last_detection_idx = max([idx for idx in track_data['frame_indices'] if idx < frame_idx], default=None)
                    
                    if last_detection_idx is not None:
                        # Find the detection at last_detection_idx
                        detection_idx = track_data['frame_indices'].index(last_detection_idx)
                        last_box = track_data['detections'][detection_idx]
                        
                        # Initialize predictor and predict forward
                        kf_pred.init_state(last_box)
                        frames_to_predict = frame_idx - last_detection_idx
                        
                        # Predict forward
                        predicted_box = last_box
                        for _ in range(frames_to_predict):
                            predicted_box = kf_pred.predict()
                        
                        # Use most common label and average score
                        corrected_label = Counter(track_data['labels']).most_common(1)[0][0]
                        avg_score = np.mean(track_data['scores'][-3:])  # Average of last 3 scores
                        
                        current_detections["boxes"].append(predicted_box)
                        current_detections["scores"].append(avg_score * 0.7)  # Reduce confidence for interpolated
                        current_detections["labels"].append(corrected_label)
            
            processed_frames.append(current_detections)
        
        # Third pass: Remove spurious detections (tracks with too few detections)
        valid_track_ids = set()
        for track_id, track_data in tracks.items():
            if len(track_data['detections']) >= min_detection_frames:
                valid_track_ids.add(track_id)
        
        # Final pass: Filter out spurious detections from processed frames
        final_frames = []
        
        for frame_idx, frame_detections in enumerate(processed_frames):
            filtered_detections = {
                "boxes": [],
                "scores": [],
                "labels": []
            }
            
            for i, (box, score, label) in enumerate(zip(
                frame_detections["boxes"], 
                frame_detections["scores"], 
                frame_detections["labels"]
            )):
                # Check if this detection belongs to a valid track
                belongs_to_valid_track = False
                
                for track_id in valid_track_ids:
                    if track_id in tracks:
                        for track_box in tracks[track_id]['detections']:
                            iou = self.bb_intersection_over_union(box, track_box)
                            if iou > 0.3:
                                belongs_to_valid_track = True
                                break
                    if belongs_to_valid_track:
                        break
                
                if belongs_to_valid_track:
                    filtered_detections["boxes"].append(box)
                    filtered_detections["scores"].append(score)
                    filtered_detections["labels"].append(label)
            
            final_frames.append(filtered_detections)
        
        return final_frames
    
    def _should_interpolate(self, track_id, current_frame_idx, tracks, max_missing_frames):
        """Check if we should interpolate for a missing track by looking at future frames"""
        track_data = tracks[track_id]
        frame_indices = track_data['frame_indices']
        
        # Check if track was seen recently enough
        last_seen = max([idx for idx in frame_indices if idx < current_frame_idx], default=-1)
        if last_seen == -1 or current_frame_idx - last_seen > max_missing_frames:
            return False
        
        # Check if track reappears in future frames (within max_missing_frames)
        future_appearances = [idx for idx in frame_indices if idx > current_frame_idx]
        if not future_appearances:
            return False
        
        # Check if the next appearance is within reasonable distance
        next_appearance = min(future_appearances)
        if next_appearance - current_frame_idx <= max_missing_frames:
            return True
        
        return False
    
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