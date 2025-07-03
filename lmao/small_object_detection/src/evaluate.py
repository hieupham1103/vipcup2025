from typing import List, Tuple

def compute_iou(box1: Tuple[int,int,int,int],
                box2: Tuple[int,int,int,int]) -> float:
    """
    Tính Intersection-over-Union (IoU) giữa hai bounding box.
    Mỗi box dưới dạng (x, y, w, h) với (x,y) là góc trên-trái.

    Trả về IoU ∈ [0,1].
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_br, y1_br = x1 + w1, y1 + h1
    x2_br, y2_br = x2 + w2, y2 + h2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_br, x2_br)
    yi2 = min(y1_br, y2_br)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_frame(pred_bboxes: List[Tuple[int,int,int,int]],
                   gt_bboxes: List[Tuple[int,int,int,int]],
                   iou_thresh: float = 0.5
                   ) -> Tuple[int, int, int]:
    """
    Args:
        pred_bboxes: list các (x,y,w,h) ở heatmap pixel
        gt_yolo_lines: list các dòng YOLO (string) cho frame đó
        orig_size: (orig_H, orig_W)
        hm_size:   (hm_H, hm_W)
        iou_thresh: ngưỡng IoU để coi là True Positive

    Returns:
        (tp, fp, fn) counts cho frame
    """

    matched_gt = [False] * len(gt_bboxes)
    tp = 0
    fp = 0

    for pred in pred_bboxes:
        best_iou = 0.0
        best_idx = -1
        for j, gt in enumerate(gt_bboxes):
            if matched_gt[j]:
                continue
            iou_ij = compute_iou(pred, gt)
            if iou_ij > best_iou:
                best_iou = iou_ij
                best_idx = j

        if best_iou >= iou_thresh and best_idx >= 0:
            tp += 1
            matched_gt[best_idx] = True
        else:
            fp += 1

    fn = matched_gt.count(False)

    return tp, fp, fn
