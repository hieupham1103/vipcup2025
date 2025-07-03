import cv2
from ultralytics import YOLO
from compensation_tracker import CompensationTracker

model = "yolov8n"
CONFIDENCE = 0.001
INPUT_VIDEO = "data/Validation_Videos/RGB/BIRD_03897.mp4"
OUTPUT_VIDEO = f"outputs/8n_TC_{CONFIDENCE}.mp4"
MODEL_CHECKPOINT = f'checkpoints/{model}/best.pt'
TRACKER_CFG = "config/bytetrack.yml"

model = YOLO(MODEL_CHECKPOINT)

comp_tracker = CompensationTracker(
    img_size=(None, None),
    max_lost_frames=10,
    cf_thresh=0.5,
    boundary_weight=0.5,
    iou_thresh=0.7
)

cap = cv2.VideoCapture(INPUT_VIDEO)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

comp_tracker.img_size = (height, width)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        source=frame, 
        tracker=TRACKER_CFG, 
        conf=CONFIDENCE, 
        stream=False,
        verbose=False,
    )
    res = results[0]

    active = []
    lost = []
    for box in res.boxes:
        if not box.id is None:
            track_id = int(box.id)
        else:
            track_id = None
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        if track_id is not None:
            active.append({'id': track_id, 'bbox': xyxy})

    recovered = comp_tracker.step(lost, active, frame)
    print(f"Frame {frame_idx}: Active: {len(active)}, Recovered: {len(recovered)}")
    for trk in active + recovered:
        x1, y1, x2, y2 = map(int, trk['bbox'])
        tid = trk['id']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y2+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    writer.write(frame)

    frame_idx += 1

cap.release()
writer.release()
print(f"Saved tracked video to {OUTPUT_VIDEO}")

