from ultralytics import YOLO
import json
from tqdm import tqdm

# Load a model
model = YOLO("checkpoints/deyolo.pt") # trained weights
ir_path = 'dataset/images/ir_val'
vis_path = 'dataset/images/vis_val'
class_offset = 1 # offset for class IDs, if needed
preds = []
with open("splitA-gt_test.json", "r") as f:
    # Parse the JSON file
    data = json.load(f)
    images = data['images']
    print(f"Number of images: {len(images)}")

    for image_info in tqdm(images, desc="Inferencing"):
        image_name = image_info['file_name']
        image_id = image_info['id']
        # Perform object detection on RGB and IR image
        results=model.predict([f"{vis_path}/{image_name}", f"{ir_path}/IR_{image_name}"], imgsz=320, conf=0.5, save=False, save_txt=False, save_conf=False, save_crop=False, verbose=False, project=None, name=None)

        res = results[0]
        for xyxy, conf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                    res.boxes.conf.cpu().numpy(),
                                    res.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = xyxy
            preds.append({
                'image_id': image_id,
                'category_id': int(cls) + class_offset,
                'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                'score': float(conf)
            })
# Save predictions to a JSON file
with open("splitA-pred_test.json", "w") as f:
    json.dump(preds, f, indent=4)
print(f"Number of predictions: {len(preds)}")
