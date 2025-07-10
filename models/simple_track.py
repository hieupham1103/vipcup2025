import cv2
import numpy as np
from collections import defaultdict, deque
from types import SimpleNamespace
import torch

class SimpleTracker:
    """
    Simple tracker optimized for videos with few objects (1-3)
    Uses IoU matching and Kalman filter for prediction
    """
    def __init__(self, 
                 max_disappeared=10,
                 iou_threshold=0.3,
                 motion_history_length=3,
                 approach_threshold=1.1):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.motion_history_length = motion_history_length
        self.approach_threshold = approach_threshold
        
        # Track storage
        self.tracks = {}
        self.next_id = 1
        self.disappeared = defaultdict(int)
        
        # Motion prediction
        self.track_history = defaultdict(lambda: deque(maxlen=motion_history_length))
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _predict_next_position(self, track_id):
        """Simple linear prediction based on last two positions"""
        if track_id not in self.tracks or len(self.tracks[track_id]) < 2:
            return None
        
        positions = list(self.tracks[track_id])
        last_pos = positions[-1]
        prev_pos = positions[-2]
        
        # Calculate velocity
        dx = last_pos[0] - prev_pos[0]
        dy = last_pos[1] - prev_pos[1]
        dw = last_pos[2] - prev_pos[2]
        dh = last_pos[3] - prev_pos[3]
        
        # Predict next position
        predicted = [
            last_pos[0] + dx,
            last_pos[1] + dy,
            last_pos[2] + dw,
            last_pos[3] + dh
        ]
        
        return predicted
    
    def _calculate_bbox_area(self, bbox):
        """Calculate area of bounding box"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def _predict_motion(self, track_id, current_bbox):
        """Predict if object is approaching or receding"""
        history = self.track_history[track_id]
        history.append(current_bbox)
        
        if len(history) < 2:
            return None
        
        ratio_sum = 0.0
        ratio_count = 0
        
        for i in range(len(history) - 1, 0, -1):
            current_area = self._calculate_bbox_area(history[i])
            prev_area = self._calculate_bbox_area(history[i-1])
            
            if prev_area > 0:
                ratio_sum += current_area / prev_area
                ratio_count += 1
        
        if ratio_count == 0:
            return None
        
        avg_ratio = ratio_sum / ratio_count
        
        if avg_ratio > self.approach_threshold:
            return "approaching"
        elif avg_ratio < (1.0 / self.approach_threshold):
            return "receding"
        else:
            return None
    
    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of [x1, y1, x2, y2, score, class_id]
        """
        if len(detections) == 0:
            # No detections, increment disappeared counter
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            return []
        
        # If no existing tracks, create new ones
        if len(self.tracks) == 0:
            results = []
            for det in detections:
                bbox = det[:4]
                score = det[4]
                class_id = det[5]
                
                self.tracks[self.next_id] = deque(maxlen=10)
                self.tracks[self.next_id].append(bbox)
                self.disappeared[self.next_id] = 0
                
                motion_type = self._predict_motion(self.next_id, bbox)
                
                result = {
                    'track_id': self.next_id,
                    'bbox': bbox,
                    'score': score,
                    'class_id': class_id,
                    'motion': motion_type
                }
                results.append(result)
                self.next_id += 1
            
            return results
        
        # Calculate IoU matrix between existing tracks and detections
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            # Use predicted position if available, otherwise use last known position
            predicted_pos = self._predict_next_position(track_id)
            if predicted_pos is not None:
                track_bbox = predicted_pos
            else:
                track_bbox = list(self.tracks[track_id])[-1]
            
            for j, det in enumerate(detections):
                det_bbox = det[:4]
                iou_matrix[i, j] = self._calculate_iou(track_bbox, det_bbox)
        
        # Hungarian algorithm alternative for small number of objects
        # Simple greedy matching for few objects
        matched_pairs = []
        unmatched_tracks = set(range(len(track_ids)))
        unmatched_detections = set(range(len(detections)))
        
        # Find best matches above threshold
        while len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            max_iou = 0
            best_track_idx = -1
            best_det_idx = -1
            
            for t_idx in unmatched_tracks:
                for d_idx in unmatched_detections:
                    if iou_matrix[t_idx, d_idx] > max_iou and iou_matrix[t_idx, d_idx] > self.iou_threshold:
                        max_iou = iou_matrix[t_idx, d_idx]
                        best_track_idx = t_idx
                        best_det_idx = d_idx
            
            if best_track_idx == -1:
                break
            
            matched_pairs.append((best_track_idx, best_det_idx))
            unmatched_tracks.remove(best_track_idx)
            unmatched_detections.remove(best_det_idx)
        
        results = []
        
        # Update matched tracks
        for track_idx, det_idx in matched_pairs:
            track_id = track_ids[track_idx]
            det = detections[det_idx]
            bbox = det[:4]
            score = det[4]
            class_id = det[5]
            
            # Update track
            self.tracks[track_id].append(bbox)
            self.disappeared[track_id] = 0
            
            motion_type = self._predict_motion(track_id, bbox)
            
            result = {
                'track_id': track_id,
                'bbox': bbox,
                'score': score,
                'class_id': class_id,
                'motion': motion_type
            }
            results.append(result)
        
        # Handle unmatched detections (create new tracks)
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            bbox = det[:4]
            score = det[4]
            class_id = det[5]
            
            self.tracks[self.next_id] = deque(maxlen=10)
            self.tracks[self.next_id].append(bbox)
            self.disappeared[self.next_id] = 0
            
            motion_type = self._predict_motion(self.next_id, bbox)
            
            result = {
                'track_id': self.next_id,
                'bbox': bbox,
                'score': score,
                'class_id': class_id,
                'motion': motion_type
            }
            results.append(result)
            self.next_id += 1
        
        # Handle unmatched tracks (increment disappeared counter)
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            self.disappeared[track_id] += 1
            
            if self.disappeared[track_id] > self.max_disappeared:
                del self.tracks[track_id]
                del self.disappeared[track_id]
        
        return results


class CustomTrackingModel:
    """
    Custom tracking model using SimpleTracker
    """
    def __init__(self, 
                 detection_model,
                 max_disappeared=30,
                 iou_threshold=0.1,
                 motion_history_length=3,
                 approach_threshold=1.1):
        self.det_model = detection_model
        self.tracker = SimpleTracker(
            max_disappeared=max_disappeared,
            iou_threshold=iou_threshold,
            motion_history_length=motion_history_length,
            approach_threshold=approach_threshold
        )
    
    def video_track(self, video_path, conf_threshold=None, iou_threshold=None, return_det=False):
        """Track objects in video"""
        detection_frames = self.det_model.video_detect(
            video_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        tracked_frames = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx >= len(detection_frames):
                break
            
            det = detection_frames[frame_idx]
            
            # Prepare detections in format [x1, y1, x2, y2, score, class_id]
            detections = []
            if det["boxes"]:
                for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
                    if isinstance(box, np.ndarray):
                        box = box.tolist()
                    detections.append([*box, float(score), int(label)])
            
            # Update tracker
            results = self.tracker.update(detections)
            
            # Format results
            for result in results:
                tracked_frame = {
                    "frame_idx": frame_idx,
                    "track_id": result['track_id'],
                    "bbox": result['bbox'],
                    "score": result['score'],
                    "label": result['class_id']
                }
                
                if result['motion']:
                    tracked_frame["motion"] = result['motion']
                
                tracked_frames.append(tracked_frame)
            
            print(f"Frame {frame_idx}: Detected {len(detections)} objects, Tracked {len(results)} objects")
            frame_idx += 1
        
        cap.release()
        return tracked_frames