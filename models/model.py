import cv2
import numpy as np
import ultralytics
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
    
    def image_detect(self,
                    image,
                    conf_threshold=None,
                    iou_threshold=None
                ):
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
            
        detections = {
            "boxes": [],
            "scores": [],
            "labels": []
        }
        
        # print(f"Running detection on image with conf: {conf_threshold}, iou: {iou_threshold}")
        results = self.model.predict(image,
                                conf=conf_threshold,
                                iou=iou_threshold,
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
    
    def video_detect(self,
                    video_path,
                    conf_threshold=None,
                    iou_threshold=None
                     
                     ) -> list:
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            det = self.image_detect(frame, 
                                    conf_threshold=conf_threshold, 
                                    iou_threshold=iou_threshold
                                    )
            
            frames.append(det)
            # if len(frames) >= 30:
            #     break
            
        cap.release()
        return frames