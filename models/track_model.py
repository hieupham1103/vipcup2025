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
                 approach_threshold=1.1,
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
    
    def _calculate_bbox_area(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def _predict_motion(self, track_id, current_bbox):
        current_area = self._calculate_bbox_area(current_bbox)
        history = self.track_history[track_id]
        history.append(current_bbox)
        
        if len(history) < 2:
            return None
        
        ratio_sum = 0.0
        ratio_count = 0
        
        for i in range(len(history) - 1, 0, -1):
            current_frame_area = self._calculate_bbox_area(history[i])
            prev_frame_area = self._calculate_bbox_area(history[i-1])
            
            if prev_frame_area > 0:
                ratio_sum += current_frame_area / prev_frame_area
                ratio_count += 1
        
        if ratio_count == 0:
            return None
        
        avg_ratio = ratio_sum / ratio_count
        
        if avg_ratio > self.approach_threshold:
            return "approaching"
        return "receding"
    
    def video_track(self, video_path: str,    
                conf_threshold=None,
                iou_threshold=None,
                return_det = False) -> list:
        
        detection_frames = self.det_model.video_detect(video_path,
                                                       conf_threshold=conf_threshold,
                                                       iou_threshold=iou_threshold
                                                       )
        tracked_frames = []
        
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
            
            # Use compensation tracker for lost objects
            if self.use_compensation and frame_idx > 0:
                # Find lost tracks
                lost_tracks = []
                lost_track_ids = self.prev_active_tracks - current_active_tracks
                
                # For simplicity, we'll use last known positions for lost tracks
                # In a more sophisticated implementation, you'd store the last bbox per track
                for lost_id in lost_track_ids:
                    # Find last known bbox for this track (from previous frames)
                    last_bbox = None
                    for prev_result in reversed(tracked_frames):
                        if prev_result["track_id"] == lost_id:
                            last_bbox = prev_result["bbox"]
                            break
                    
                    if last_bbox:
                        lost_tracks.append({'id': lost_id, 'bbox': last_bbox})
                
                # Get recovered tracks
                recovered_tracks = self.comp_tracker.step(lost_tracks, active_tracks, frame)
                print(f"Frame {frame_idx}: Active: {len(active_tracks)}, Lost: {len(lost_tracks)}, Recovered: {len(recovered_tracks)}")
                # Add recovered tracks to results
                for recovered in recovered_tracks:
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
            
            self.prev_active_tracks = current_active_tracks
            frame_idx += 1

        cap.release()
        return tracked_frames