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
        # update existing trackers: increment lost count if not updated
        for t in lost_list:
            tid = t['id']
            bbox = t['bbox']
            if tid not in self.trackers:
                # init
                self.trackers[tid] = {
                    'kf': self._init_kf(bbox),
                    'prev': bbox,
                    'success': 1,
                    'lost': 1
                }
            else:
                self.trackers[tid]['lost'] += 1
        # motion predict and filter
        for tid, v in list(self.trackers.items()):
            kf = v['kf']
            prev = v['prev']
            success = v['success']; lost = v['lost']
            if lost > self.max_lost:
                # remove tracker
                del self.trackers[tid]
                continue
            # predict
            kf.predict()
            cx, cy, s, r, vx, vy, vs = kf.x
            w = np.sqrt(s*r)
            h = s / (w+1e-6)
            x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
            pred = [x1,y1,x2,y2]
            # filters
            # 1. confidence filter
            if success / (success+lost) < self.cf_thresh:
                continue
            # 2. boundary filter
            bx, by, bw, bh = x1, y1, w, h
            if not (bx > -bw*self.boundary_weight and bx+ bw*(1+self.boundary_weight) < self.img_w and
                    by > -bh*self.boundary_weight and by+ bh*(1+self.boundary_weight) < self.img_h):
                continue
            # 3. IoU filter vs active
            if any(iou_bbox(pred, a['bbox']) > self.iou_thresh for a in active_list):
                continue
            # 4. appearance filter: compare HOG
            x1i,y1i,x2i,y2i = map(int, prev)
            p1,p2,p3,p4 = map(int, pred)
            # extract patches
            patch_prev = frame[y1i:y2i, x1i:x2i]
            patch_pred = frame[p2:p4, p1:p3] = frame[p3:p4, p1:p3] # fix slicing
            # compute HOG
            if patch_prev.size==0 or patch_pred.size==0: continue
            h1 = self.hog.compute(patch_prev)
            h2 = self.hog.compute(patch_pred)
            dist = np.linalg.norm(h1-h2)
            if dist > 0.5:  # threshold
                continue
            # passed all => re-track
            new_tracks.append({'id': tid, 'bbox': pred})
            # update state
            kf.update(np.array([cx, cy, s, r]))
            v['prev'] = pred
            v['success'] += 1
        return new_tracks
