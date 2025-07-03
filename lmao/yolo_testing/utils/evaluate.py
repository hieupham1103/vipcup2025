import os
import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from .pycocotools_modified.coco import COCO
# from pycocotools.cocoeval import COCOeval
from .pycocotools_modified.cocoeval_modified import COCOeval
from .multiscale import YOLOMS


def coco_evaluate(gt_json: str, pred_list: list, iou_type: str = 'bbox'):
    coco_gt = COCO(gt_json)
    tmp_json = 'yolo_tmp_preds.json'
    with open(tmp_json, 'w') as f:
        json.dump(pred_list, f)
    coco_dt = coco_gt.loadRes(tmp_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print('----------------------------------------')
    print('AP_0.5-0.95', coco_eval.stats[0])
    print('AP_0.5', coco_eval.stats[1])
    print('AP_S', coco_eval.stats[3])
    # print('AP_M', coco_eval.stats[4])
    # print('AP_L', coco_eval.stats[5])
    print('f1_score: ', coco_eval.stats[20])
    print('----------------------------------------')
    os.remove(tmp_json)
    
    return coco_eval


def evaluate(model, gt_json: str, image_folder: str, class_offset: int = 1, device='cpu'):
    """
    Evaluate a detection model (YOLOv8 or YOLOMS) on a COCO-format ground truth.

    Args:
        model: YOLO instance or YOLOMS instance
        gt_json: path to COCO-format ground truth JSON
        image_folder: directory containing test images (file_name in gt_json)
        class_offset: offset to add to class indices (YOLO cls start at 0, COCO at 1)
    """
    coco_gt = COCO(gt_json)
    images = coco_gt.loadImgs(coco_gt.getImgIds())

    preds = []
    for img_info in images:
        image_id = img_info['id']
        file_name = img_info['file_name']
        img_path = os.path.join(image_folder, file_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read {img_path}")
            continue

        # Determine model type
        if hasattr(model, 'detect'):
            boxes, scores, labels = model.detect(img,
                                                 device=device)
            # boxes in [x1,y1,x2,y2]
            for (x1, y1, x2, y2), score, cls in zip(boxes, scores, labels):
                preds.append({
                    'image_id': image_id,
                    'category_id': int(cls) + class_offset,
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': float(score)
                })
        else:
            # Assume YOLO
            results = model.predict(img,
                                    imgsz=[320, 256],
                                    verbose=False,
                                    device=device)
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

    # Run COCO evaluation
    print("Running COCO evaluation...")
    return coco_evaluate(gt_json, preds, iou_type='bbox')
    
    

