import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_detection_video(
        video_path: str,
        detection_frames: list,
        output_path: str,
        ):
    check_output_folder(output_path)
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
                if isinstance(box, np.ndarray):
                    box = box.tolist()
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
    check_output_folder(output_path)
    image = cv2.imread(image_path)
    for box, score, label in zip(detection_results["boxes"], detection_results["scores"], detection_results["labels"]):
        if isinstance(box, np.ndarray):
            box = box.tolist()
        label = int(label.item())
        score = score.item()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(output_path, image)

def visualize_tracking_video(
        video_path: str,
        tracking_frames: list,
        output_path: str,
        ):
    check_output_folder(output_path)
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
        for det in tracking_frames:
            if det["frame_idx"] == frame_index:
                x1, y1, x2, y2 = map(int, det["bbox"])
                track_id = int(det["track_id"])
                label = int(det["label"])
                score = det["score"]
                motion = det.get("motion", None)
                
                if motion == "approaching":
                    color = (0, 0, 255)
                elif motion == "receding":
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                text = f"ID: {track_id} {label} {score:.2f}"
                if motion:
                    text += f" [{motion}]"
                
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        writer.write(frame)
    cap.release()
    writer.release()
    

def check_output_folder(file_path: str):
    folder_path = os.path.dirname(file_path)
    os.makedirs(folder_path, exist_ok=True)
    
