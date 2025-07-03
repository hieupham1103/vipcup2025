import cv2
import numpy as np
import ultralytics
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.cfg import get_cfg
import torch

class DetectionModel:
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
        results = self.model.predict(image,
                                conf=self.conf_threshold,
                                iou=self.iou_threshold,
                                verbose=False,
                                stream=True,
                                device=self.device
                             )
        for result in results:
            for box in result.boxes:
                detections["boxes"].append(box.xyxy.cpu()[0])
                detections["scores"].append(box.conf.cpu()[0])
                detections["labels"].append(box.cls.cpu()[0])
            # break
            
        return detections
    
    def video_detect(self, video_path) -> list:
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            det = self.image_detect(frame)
            
            frames.append(det)
            # if len(frames) >= 30:
            #     break
            
        cap.release()
        return frames
    
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