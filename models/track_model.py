import cv2
import numpy as np
import ultralytics
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.cfg import get_cfg
from types import SimpleNamespace
import torch

class TrackingModel:
    def __init__(self,
                 detection_model,
                 config_path='configs/bytetrack.yml',
                 fps = 25
                 ):
        self.det_model = detection_model
        
        self.tracker = BYTETracker(
                            args=get_cfg(config_path),
                            frame_rate=fps
                            )
    
    def video_track(self, video_path: str, return_det = False) -> list:
        detection_frames = self.det_model.video_detect(video_path)
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
            
            boxes = np.array(det["boxes"]) if det["boxes"] else np.zeros((0, 4))
            scores = np.array(det["scores"]) if det["scores"] else np.zeros((0,))
            labels = np.array(det["labels"]) if det["labels"] else np.zeros((0,))
            
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

            # print(online_targets)
            # print(f"Frame {frame_idx}: {len(online_targets)} tracked objects")
            for det in online_targets:
                # print("Tracked Object:")
                # print(det)
                # x1, y1, x2, y2, track_id, cls_id = det[:6]
                x1, y1, x2, y2, track_id, score, class_id, _ = det
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                tracked_frames.append({
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "score": score,
                    "label": class_id
                })
            
            frame_idx += 1

        cap.release()
        return tracked_frames