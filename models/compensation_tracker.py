import numpy as np
import cv2
from filterpy.kalman import KalmanFilter

def iou_bbox(boxA, boxB):
    # compute intersection over union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / (unionArea + 1e-6)

class CompensationTracker:
    """
    Compensation Tracker for re-tracking lost objects via Kalman and simple filters.
    """
    def __init__(self,
                 img_size,                      # (height, width)
                 max_lost_frames=30,
                 cf_thresh=0.5,
                 boundary_weight=0.5,
                 iou_thresh=0.7):
        self.img_h, self.img_w = img_size
        self.max_lost = max_lost_frames
        self.cf_thresh = cf_thresh
        self.boundary_weight = boundary_weight
        self.iou_thresh = iou_thresh
        # store for each track_id: dict with kf, prev_bbox, success, lost
        self.trackers = {}
        # HOG descriptor
        self.hog = cv2.HOGDescriptor()

    def _init_kf(self, bbox):
        # bbox: [x1,y1,x2,y2]
        kf = KalmanFilter(dim_x=7, dim_z=4)
        # state: x, y, scale, ratio, vx, vy, vs
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        s = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
        r = (bbox[2]-bbox[0]) / (bbox[3]-bbox[1] + 1e-6)
        kf.x = np.array([cx, cy, s, r, 0,0,0], dtype=float)
        # define matrices
        dt = 1.
        kf.F = np.eye(7)
        for i in range(4): kf.F[i, i+3] = dt
        kf.H = np.zeros((4,7)); kf.H[:4,:4] = np.eye(4)
        kf.P *= 10.
        kf.R *= 1.
        kf.Q = np.eye(7) * 0.01
        return kf

    def step(self, lost_list, active_list, frame):
        """
        lost_list: list of dict {id, bbox}
        active_list: list of dict {id, bbox}
        frame: BGR image numpy
        return: list of dict {id, bbox}
        """
        new_tracks = []
        
        # Get frame dimensions
        frame_h, frame_w = frame.shape[:2]
        
        # update existing trackers: increment lost count if not updated
        for t in lost_list:
            tid = t['id']
            bbox = t['bbox']
            if tid not in self.trackers:
                kf = self._init_kf(bbox)
                self.trackers[tid] = {
                    'kf': kf,
                    'prev': bbox,
                    'success': 1,
                    'lost': 0
                }
            else:
                self.trackers[tid]['lost'] += 1
        
        # motion predict and filter
        for tid, v in list(self.trackers.items()):
            kf = v['kf']
            prev = v['prev']
            success = v['success']
            lost = v['lost']
            
            if lost > self.max_lost:
                del self.trackers[tid]
                continue
                
            # predict
            kf.predict()
            cx, cy, s, r, vx, vy, vs = kf.x
            w = np.sqrt(s * r)
            h = s / (w + 1e-6)
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            pred = [x1, y1, x2, y2]
            
            # filters
            # 1. confidence filter
            if success / (success + lost) < self.cf_thresh:
                continue
                
            # 2. boundary filter - ensure coordinates are within frame bounds
            x1, y1, x2, y2 = pred
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(x1 + 1, min(x2, frame_w))
            y2 = max(y1 + 1, min(y2, frame_h))
            
            # Check if bbox is valid
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Check boundary weight
            bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
            boundary_penalty = 0
            if bx < 10: boundary_penalty += 1
            if by < 10: boundary_penalty += 1
            if bx + bw > frame_w - 10: boundary_penalty += 1
            if by + bh > frame_h - 10: boundary_penalty += 1
            
            if boundary_penalty * self.boundary_weight > 0.5:
                continue
                
            # 3. appearance filter using HOG (optional - can be simplified)
            try:
                # Extract patches with safe slicing
                p1, p2, p3, p4 = map(int, [x1, y1, x2, y2])
                prev_x1, prev_y1, prev_x2, prev_y2 = map(int, prev)
                
                # Ensure coordinates are within bounds
                prev_x1 = max(0, min(prev_x1, frame_w - 1))
                prev_y1 = max(0, min(prev_y1, frame_h - 1))
                prev_x2 = max(prev_x1 + 1, min(prev_x2, frame_w))
                prev_y2 = max(prev_y1 + 1, min(prev_y2, frame_h))
                
                p1 = max(0, min(p1, frame_w - 1))
                p2 = max(0, min(p2, frame_h - 1))
                p3 = max(p1 + 1, min(p3, frame_w))
                p4 = max(p2 + 1, min(p4, frame_h))
                
                # Extract patches
                patch_prev = frame[prev_y1:prev_y2, prev_x1:prev_x2]
                patch_pred = frame[p2:p4, p1:p3]
                
                # Skip if patches are too small
                if patch_prev.size == 0 or patch_pred.size == 0:
                    continue
                if patch_prev.shape[0] < 5 or patch_prev.shape[1] < 5:
                    continue
                if patch_pred.shape[0] < 5 or patch_pred.shape[1] < 5:
                    continue
                    
                # Simple appearance check (you can enhance this)
                # For now, just add the track if it passes other filters
                new_tracks.append({'id': tid, 'bbox': [x1, y1, x2, y2]})
                
            except Exception as e:
                # If appearance filtering fails, skip this track
                print(f"Appearance filtering failed for track {tid}: {e}")
                continue
        
        return new_tracks
