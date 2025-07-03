import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import DetectionModel, TrackingModel

def visualize_detection_video(
        video_path: str,
        detection_frames: list,
        output_path: str,
        ):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_index < len(detection_frames):
            detections = detection_frames[frame_index]
            for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
                box = box[0].tolist()
                label = int(label.item())
                score = score.item()
                # print(label)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        writer.write(frame)
    cap.release()
    writer.release()

def visualize_detection_image(
        image_path: str,
        detection_results: list,
        output_path: str,
        ):
    image = cv2.imread(image_path)
    for box, score, label in zip(detection_results["boxes"], detection_results["scores"], detection_results["labels"]):
        box = box[0].tolist()
        label = int(label.item())
        score = score.item()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(output_path, image)