import cv2
import numpy as np
import ultralytics
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.cfg import get_cfg
from types import SimpleNamespace
import torch
from collections import defaultdict, deque
from .compensation_tracker import CompensationTracker

class TrackingModel:
    def __init__(self,
                 detection_model,
                 config_path='configs/bytetrack.yml',
                 fps = 25,
                 motion_history_length=3,
                 approach_threshold=1.05,
                 use_compensation=True,
                 compensation_config=None
                 ):
        self.det_model = detection_model
        
        self.tracker = BYTETracker(
                            args=get_cfg(config_path),
                            frame_rate=fps
                            )
        
        self.motion_history_length = motion_history_length
        self.approach_threshold = approach_threshold
        self.track_history = defaultdict(lambda: deque(maxlen=motion_history_length))
        
        # Compensation tracker
        self.use_compensation = use_compensation
        if self.use_compensation:
            comp_config = compensation_config or {}
            self.comp_tracker = CompensationTracker(
                img_size=(None, None),  # Will be set in video_track
                max_lost_frames=comp_config.get('max_lost_frames', 10),
                cf_thresh=comp_config.get('cf_thresh', 0.5),
                boundary_weight=comp_config.get('boundary_weight', 0.5),
                iou_thresh=comp_config.get('iou_thresh', 0.7)
            )
        
        # Track lost objects from previous frame
        self.prev_active_tracks = set()
        # Store last known positions for each track
        self.track_positions = {}
    
    def _calculate_bbox_area(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # Fixed: bbox2[2] instead of bbox2
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _is_valid_recovery(self, recovered_bbox, track_id, active_tracks):
        """Check if recovered track is valid and not overlapping with active tracks"""
        
        # Check if bbox is reasonable size
        width = recovered_bbox[2] - recovered_bbox[0]
        height = recovered_bbox[3] - recovered_bbox[1]
        
        if width <= 5 or height <= 5:  # Too small
            return False
        
        if width > 500 or height > 500:  # Too large (adjust based on your data)
            return False
        
        # Check for overlap with active tracks (avoid duplicates)
        for active_track in active_tracks:
            iou = self._calculate_iou(recovered_bbox, active_track['bbox'])
            if iou > 0.3:  # High overlap with existing track
                return False
        
        # Check if track has moved too far from last known position
        if track_id in self.track_positions:
            last_bbox = self.track_positions[track_id]
            
            # Calculate center distance
            last_center = [(last_bbox[0] + last_bbox[2])/2, (last_bbox[1] + last_bbox[3])/2]
            current_center = [(recovered_bbox[0] + recovered_bbox[2])/2, (recovered_bbox[1] + recovered_bbox[3])/2]
            
            distance = np.sqrt((last_center[0] - current_center[0])**2 + 
                             (last_center[1] - current_center[1])**2)
            
            # Check if moved too far (adjust threshold based on your data)
            max_movement = max(width, height) * 2  # Allow movement up to 2x bbox size
            if distance > max_movement:
                return False
        
        return True
    
    def _predict_motion(self, track_id, current_bbox):
        current_area = self._calculate_bbox_area(current_bbox)
        history = self.track_history[track_id]
        history.append(current_bbox)
        
        if len(history) < 2:
            return "receding"
        
        ratio_sum = 0.0
        ratio_count = 0
        
        for i in range(len(history) - 1, 0, -1):
            current_frame_area = self._calculate_bbox_area(history[i])
            prev_frame_area = self._calculate_bbox_area(history[i-1])
            
            if prev_frame_area > 0:
                ratio_sum += current_frame_area / prev_frame_area
                ratio_count += 1
        
        if ratio_count == 0:
            return "receding"
        
        avg_ratio = ratio_sum / ratio_count
        
        if avg_ratio > self.approach_threshold:
            return "approaching"
        return "receding"
    
    def video_track(self, video_path,    
                conf_threshold=None,
                iou_threshold=None,
                return_det = False) -> list:
        
        detection_frames = self.det_model.video_detect(video_path,
                                                       conf_threshold=conf_threshold,
                                                       iou_threshold=iou_threshold
                                                       )
        tracked_frames = []
        if isinstance(video_path, list):
            video_path = video_path[0]
        cap = cv2.VideoCapture(video_path)
        
        # Set compensation tracker image size
        if self.use_compensation:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.comp_tracker.img_h, self.comp_tracker.img_w = height, width
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx >= len(detection_frames):
                break
                
            det = detection_frames[frame_idx]
            
            boxes = np.array(det["boxes"]) if det["boxes"] else np.zeros((0, 4))
            scores = np.array(det["scores"]) if det["scores"] else np.zeros((0,))
            labels = np.array(det["labels"]) if det["labels"] else np.zeros((0,))
            print(f"Frame {frame_idx}: Detected {len(boxes)} objects")
            
            xywh = np.column_stack([
                (boxes[:, 0] + boxes[:, 2]) / 2,  # center_x
                (boxes[:, 1] + boxes[:, 3]) / 2,  # center_y
                (boxes[:, 2] - boxes[:, 0]),      # width
                (boxes[:, 3] - boxes[:, 1])       # height
            ])
            
            fake_result = SimpleNamespace(
                conf=scores,
                cls=labels,
                xywh=xywh
            )
            
            online_targets = self.tracker.update(fake_result, img=frame)
            
            # Process active tracks
            active_tracks = []
            current_active_tracks = set()
            
            for det in online_targets:
                x1, y1, x2, y2, track_id, score, class_id, _ = det
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                bbox = [x1, y1, x2, y2]
                
                motion_type = self._predict_motion(track_id, bbox)
                
                result = {
                    "frame_idx": frame_idx,
                    "track_id": int(track_id),
                    "bbox": bbox,
                    "score": score,
                    "label": class_id,
                    "source": "primary"  # Mark as primary tracker
                }
                
                if motion_type:
                    result["motion"] = motion_type
                
                tracked_frames.append(result)
                active_tracks.append({'id': int(track_id), 'bbox': bbox})
                current_active_tracks.add(int(track_id))
                
                # Update track positions
                self.track_positions[int(track_id)] = bbox
            
            # Use compensation tracker for lost objects
            if self.use_compensation and frame_idx > 0:
                # Find lost tracks
                lost_tracks = []
                lost_track_ids = self.prev_active_tracks - current_active_tracks
                
                # For lost tracks, use stored positions
                for lost_id in lost_track_ids:
                    if lost_id in self.track_positions:
                        last_bbox = self.track_positions[lost_id]
                        lost_tracks.append({'id': lost_id, 'bbox': last_bbox})
                
                # Get recovered tracks
                recovered_tracks = self.comp_tracker.step(lost_tracks, active_tracks, frame)
                
                # Validate and filter recovered tracks
                valid_recovered = []
                for recovered in recovered_tracks:
                    if self._is_valid_recovery(recovered['bbox'], recovered['id'], active_tracks):
                        valid_recovered.append(recovered)
                    else:
                        # Remove invalid tracks from compensation tracker
                        if recovered['id'] in self.comp_tracker.trackers:
                            del self.comp_tracker.trackers[recovered['id']]
                
                print(f"Frame {frame_idx}: Active: {len(active_tracks)}, Lost: {len(lost_tracks)}, "
                      f"Recovered: {len(recovered_tracks)}, Valid: {len(valid_recovered)}")
                
                # Add valid recovered tracks to results
                for recovered in valid_recovered:
                    motion_type = self._predict_motion(recovered['id'], recovered['bbox'])
                    
                    result = {
                        "frame_idx": frame_idx,
                        "track_id": recovered['id'],
                        "bbox": recovered['bbox'],
                        "score": 0.5,  # Default score for recovered tracks
                        "label": 0,    # Default label
                        "source": "compensation"  # Mark as compensation tracker
                    }
                    
                    if motion_type:
                        result["motion"] = motion_type
                    
                    tracked_frames.append(result)
                    current_active_tracks.add(recovered['id'])
                    
                    # Update track positions for valid recoveries
                    self.track_positions[recovered['id']] = recovered['bbox']
            
            self.prev_active_tracks = current_active_tracks
            frame_idx += 1

        cap.release()
        return tracked_frames