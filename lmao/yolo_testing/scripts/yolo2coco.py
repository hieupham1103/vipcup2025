import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def yolo_to_coco(
    images_dir: str,
    labels_dir: str,
    file_list: str,
    output_json: str,
    class_names: list
):
    """
    Convert YOLO-format dataset to COCO-style JSON for evaluation.
    Args:
        images_dir: directory containing images
        labels_dir: directory containing YOLO .txt labels
        file_list: path to txt file listing image paths (one per line)
        output_json: filename for output COCO JSON
        class_names: list of class names
    """
    coco = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    # categories
    for idx, name in enumerate(class_names, start=1):
        coco['categories'].append({'id': idx, 'name': name, 'supercategory': ''})

    ann_id = 1
    with open(file_list, 'r') as f:
        for img_id, line in enumerate(f, start=1):
            img_path = line.strip()
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            coco['images'].append({'id': img_id, 'file_name': filename, 'width': w, 'height': h})

            # corresponding label file
            label_path = os.path.join(
                labels_dir,
                os.path.splitext(filename)[0] + '.txt'
            )
            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as lf:
                for line in lf:
                    cls, xc, yc, bw, bh = map(float, line.strip().split())
                    # convert YOLO norm to pixel
                    x_center, y_center = xc * w, yc * h
                    bw_pix, bh_pix = bw * w, bh * h
                    x1 = x_center - bw_pix / 2
                    y1 = y_center - bh_pix / 2
                    coco['annotations'].append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': int(cls) + 1,
                        'bbox': [x1, y1, bw_pix, bh_pix],
                        'area': bw_pix * bh_pix,
                        'iscrowd': 0
                    })
                    ann_id += 1

    # write to JSON
    with open(output_json, 'w') as out:
        json.dump(coco, out, indent=2)
    print(f'COCO GT JSON saved to {output_json}')

if __name__ == '__main__':
    base = '/home/cvpr2025/yolo_testing/data/vipcup_det/split_A/RGB'
    images_dir = os.path.join(base, 'images', 'test')  # or train/val
    labels_dir = os.path.join(base, 'labels', 'test')  # ensure this structure
    file_list = '/home/cvpr2025/yolo_testing/config/Detection/RGB/test.txt'
    output_json = 'gt_test.json'
    class_names = ['BIRD', 'DRONE']

    yolo_to_coco(images_dir, labels_dir, file_list, output_json, class_names)