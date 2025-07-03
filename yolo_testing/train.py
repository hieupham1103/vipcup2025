import yaml
import torch.nn as nn, torch
from ultralytics import YOLO


# model = YOLO("/home/cvpr2025/yolo_testing/zoo/yolo11n.pt")


# model.train(
#     data="/home/cvpr2025/yolo_testing/config/Detection/RGB/vipcup_det_A_RGB.yml",
#     epochs=500,
#     device=1,
#     imgsz=[320, 256],    
#     batch=64,
#     patience=5,
# )

model = YOLO("/home/cvpr2025/yolo_testing/runs/detect/train/weights/last.pt")
model.train(resume=True)