import cv2
import numpy as np
import ultralytics
import torch
from torchvision.ops import nms as torch_nms
from ensemble_boxes import *
from .model import DetectionModel as BaseDetectionModel
from collections import Counter


class DetectionModel(BaseDetectionModel):
    def __init__(self,
                 model_path: str,
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model(self.model_path)
        self.device = device
    
    def _load_model(self, model_path):
        if "deyolo" in model_path:
            self.model = ultralytics.YOLO(model_path)
        else:
            self.model = ultralytics.YOLO(model_path)
        return self.model
    
    def image_detect(self,
                     image,
                     scales=[1.0],
                     crop_ratio=0.65,
                     weights = [
                            0.75,  # original image
                            1.0,  # top-left
                            1.0,  # top-right
                            1.0,  # bottom-left
                            1.0,  # bottom-right
                            2.0   # center
                        ]
                    ,
                    conf_threshold=None,
                    iou_threshold=None
                     ):
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        # print(f"Running detection on image with conf: {conf_threshold}, iou: {iou_threshold}")
        detections = {
            "boxes": [],
            "scores": [],
            "labels": []
        }
        # print(type(image))
        if isinstance(image, np.ndarray):
            image = [image]
        h, w = image[0].shape[:2]
        
        batch_images = []
        batch_metadata = []
        total_weights = []
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = [cv2.resize(img, (new_w, new_h)) for img in image]
            # ảnh gốc
            batch_images.append(scaled_image)
            batch_metadata.append({
                'scale': scale,
                'crop_size': (new_w, new_h),
                'offset': (0, 0),
                'original_size': (w, h)
            })
            
            crop_w = int(new_w * crop_ratio)
            crop_h = int(new_h * crop_ratio)
                
            crop_positions = [
                (0, 0),
                (new_w - crop_w, 0),
                (0, new_h - crop_h),
                (new_w - crop_w, new_h - crop_h),
                ((new_w - crop_w) // 2, (new_h - crop_h) // 2)
            ]
            
            
            # print(f"Processing scale: {scale}, new size: ({new_w}, {new_h})")
            # print(f"Crop size: ({crop_w}, {crop_h}), positions: {crop_positions}")
            
            for idx, (x_offset, y_offset) in enumerate(crop_positions):
                crop = []
                for img in scaled_image:
                    crop.append(img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w])
                batch_images.append(crop)
                batch_metadata.append({
                    'scale': scale,
                    'crop_size': (crop_w, crop_h),
                    'offset': (x_offset, y_offset),
                    'original_size': (w, h)
                })
                
        # 3. Run batch inference
        if len(batch_images[0]) == 1:
            batch_images = [img[0] for img in batch_images]
            batch_results = self.model.predict(
                batch_images,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
                device=self.device,
                stream=True
            )
            # batch_results = []
            # for image in batch_images:
            #     result = self.model.predict(
            #         image[0],
            #         conf=conf_threshold,
            #         iou=iou_threshold,
            #         verbose=False,
            #         device=self.device,
            #         # stream=True
            #     )
            #     batch_results.append(result[0])
        else:
            batch_results = []
            for image in batch_images:
                result = self.model.predict(
                    image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                    device=self.device,
                    # stream=True
                )
                batch_results.append(result[0])
        
        
        detections_per_view = []
        
        # print(batch_images[0][1].shape, batch_images[0][1].shape)
        for idx, (result, metadata) in enumerate(zip(batch_results, batch_metadata)):
            # print(f"Processing result {idx + 1}/{len(batch_metadata)}")
            # if result.boxes is None:
            #     print("0 boxes detected, skipping...")
            # else:
            #     print(f"Detected {len(result.boxes)} boxes")
            if result.boxes is None or len(result.boxes) == 0:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            
            scale = metadata['scale']
            x_offset, y_offset = metadata['offset']
            w_orig, h_orig = metadata['original_size']
            
            # Step 1: Add offset to convert from crop coordinates to scaled image coordinates
            boxes[:, 0] = boxes[:, 0] + x_offset
            boxes[:, 1] = boxes[:, 1] + y_offset
            boxes[:, 2] = boxes[:, 2] + x_offset
            boxes[:, 3] = boxes[:, 3] + y_offset
            
            # Step 2: Convert from scaled image to original image and normalize
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] / scale) / w_orig
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] / scale) / h_orig
            
            boxes = np.clip(boxes, 0, 1)
            
            if len(boxes) > 0:
                detections_per_view.append({
                    'boxes': boxes.tolist(),
                    'scores': scores.tolist(),
                    'labels': labels.tolist()
                })
                
                total_weights.append(weights[idx])
            # print(f"View {idx + 1}/{len(batch_metadata)}: Detected {len(boxes)} boxes with weight {weights[idx]}")

        # 5. Apply Non-Maximum Weighted fusion
        if detections_per_view:
            boxes_list = [det['boxes'] for det in detections_per_view]
            scores_list = [det['scores'] for det in detections_per_view]
            labels_list = [det['labels'] for det in detections_per_view]

            # Apply Weighted Boxes Fusion
            wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=total_weights,
                iou_thr=iou_threshold,
                skip_box_thr=conf_threshold
            )
            #filter out boxes with low scores
            wbf_keep = wbf_scores > conf_threshold
            # wbf_boxes = wbf_boxes.tolist()
            # wbf_scores = wbf_scores.tolist()
            # wbf_labels = wbf_labels.tolist()
            final_boxes = wbf_boxes[wbf_keep]
            final_scores = wbf_scores[wbf_keep]
            final_labels = wbf_labels[wbf_keep]

            nms_keep_indices = torch_nms(
                torch.tensor(final_boxes),
                torch.tensor(final_scores),
                iou_threshold
            )
            final_boxes = [final_boxes[i] for i in nms_keep_indices]
            final_scores = [final_scores[i] for i in nms_keep_indices]
            final_labels = [final_labels[i] for i in nms_keep_indices]

            for box, score, label in zip(final_boxes, final_scores, final_labels):
                x1 = box[0] * w
                y1 = box[1] * h
                x2 = box[2] * w
                y2 = box[3] * h
                detections["boxes"].append([x1, y1, x2, y2])
                detections["scores"].append(score)
                detections["labels"].append(label)

        return detections
    
    def video_detect(self,
                    video_path,
                    scales=[1.0],
                    crop_ratio=0.65,
                    weights = [
                            0.75,  # original image
                            1.0,  # top-left
                            1.0,  # top-right
                            1.0,  # bottom-left
                            1.0,  # bottom-right
                            2.0   # center
                        ]
                    ,
                    conf_threshold=None,
                    iou_threshold=None,
                    video_buffer_size=2
                     ) -> list:
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
            
        frames = []
        if isinstance(video_path, str):
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                det = self.image_detect(frame,
                                        scales=scales,
                                        crop_ratio=crop_ratio,
                                        weights=weights,
                                        conf_threshold=conf_threshold,
                                        iou_threshold=iou_threshold
                                    )
                
                frames.append(det)
            cap.release()
        elif isinstance(video_path, list):
            cap_1 = cv2.VideoCapture(video_path[0])
            cap_2 = cv2.VideoCapture(video_path[1])
            while cap_1.isOpened() or cap_2.isOpened():
                ret, frame_1 = cap_1.read()
                ret2, frame_2 = cap_2.read()
                if not ret and not ret2:
                    break
                det = self.image_detect([frame_1, frame_2],
                                        scales=scales,
                                        crop_ratio=crop_ratio,
                                        weights=weights,
                                        conf_threshold=conf_threshold,
                                        iou_threshold=iou_threshold
                                    )
                frames.append(det)
            cap_1.release()
            cap_2.release()
        else:
            raise ValueError("video_path must be a string or a list of strings")
        before_frame = []
        after_frame = []
        if len(frames) > video_buffer_size:
            after_frame = frames[1:video_buffer_size + 1]
        else:
            after_frame = frames[1:]
        
        # final_frames = []
        
        # for idx, frame in enumerate(frames):
        #     print(f"Processing frame {idx + 1}/{len(frames)}")
            
        #     # Apply interpolation postprocessing
        #     interpolated_frame = self.interpolate_frame(
        #         current_frame=frame,
        #         before_frames=before_frame.copy(),
        #         after_frames=after_frame.copy(),
        #         iou_threshold=self.iou_threshold
        #     )
            
        #     final_frames.append(interpolated_frame)
                          
        #     # Update before_frame and after_frame buffers
        #     if len(before_frame) >= video_buffer_size:
        #         before_frame.pop(0)
        #     if len(after_frame) >= video_buffer_size:
        #         after_frame.pop(0)
        #     before_frame.append(frame)
        #     if idx + video_buffer_size < len(frames):
        #         after_frame.append(frames[idx + video_buffer_size])
        
        return frames

    def interpolate_frame(self, current_frame, before_frames, after_frames, iou_threshold=0.5):
        """
        Apply temporal interpolation to current frame based on before and after frames
        
        Args:
            current_frame: Current frame detections
            before_frames: List of previous frame detections
            after_frames: List of future frame detections
            iou_threshold: IoU threshold for matching objects
        
        Returns:
            Interpolated frame with corrected detections
        """
        # Create a copy of current frame to modify
        interpolated_frame = {
            "boxes": current_frame["boxes"].copy(),
            "scores": current_frame["scores"].copy(),
            "labels": current_frame["labels"].copy()
        }
        
        # 1. Remove isolated objects (appear only in current frame)
        objects_to_remove = self.find_isolated_objects(
            current_frame, before_frames, after_frames, iou_threshold
        )
        
        # Apply removal (remove in reverse order to maintain indices)
        for obj_idx in sorted(objects_to_remove, reverse=True):
            removed_label = interpolated_frame["labels"][obj_idx]
            removed_score = interpolated_frame["scores"][obj_idx]
            del interpolated_frame["boxes"][obj_idx]
            del interpolated_frame["scores"][obj_idx]
            del interpolated_frame["labels"][obj_idx]
            print(f"  -> Removed isolated object: label {removed_label}, score {removed_score:.3f}")
        
        # 2. Find missing objects (appear in before AND after frames but not in current)
        missing_objects = self.find_missing_objects(
            current_frame, before_frames, after_frames, iou_threshold
        )
        
        # 3. Find objects with wrong labels
        label_corrections = self.find_label_corrections(
            current_frame, before_frames, after_frames, iou_threshold
        )
        
        # 4. Apply missing object interpolation
        for missing_obj in missing_objects:
            interpolated_frame["boxes"].append(missing_obj["box"])
            interpolated_frame["scores"].append(missing_obj["score"])
            interpolated_frame["labels"].append(missing_obj["label"])
            print(f"  -> Added missing object: label {missing_obj['label']}, score {missing_obj['score']:.3f}")
        
        # 5. Apply label corrections (after removal, so indices might have changed)
        for correction in label_corrections:
            box_idx = correction["box_idx"]
            # Skip if this object was removed
            if box_idx < len(interpolated_frame["labels"]):
                new_label = correction["new_label"]
                old_label = interpolated_frame["labels"][box_idx]
                interpolated_frame["labels"][box_idx] = new_label
                print(f"  -> Corrected label: {old_label} → {new_label}")
        
        return interpolated_frame

    def find_isolated_objects(self, current_frame, before_frames, after_frames, iou_threshold):
        """Find objects that appear only in current frame and not in any temporal frames"""
        isolated_objects = []
        
        for curr_idx, curr_box in enumerate(current_frame["boxes"]):
            curr_label = current_frame["labels"][curr_idx]
            
            # Check if this object has any temporal support
            has_temporal_support = False
            
            # Check before frames
            for frame in before_frames:
                for i, box in enumerate(frame["boxes"]):
                    if (frame["labels"][i] == curr_label and 
                        self.bb_intersection_over_union(curr_box, box) > iou_threshold):
                        has_temporal_support = True
                        break
                if has_temporal_support:
                    break
            
            # Check after frames (only if no support found in before frames)
            if not has_temporal_support:
                for frame in after_frames:
                    for i, box in enumerate(frame["boxes"]):
                        if (frame["labels"][i] == curr_label and 
                            self.bb_intersection_over_union(curr_box, box) > iou_threshold):
                            has_temporal_support = True
                            break
                    if has_temporal_support:
                        break
            
            # If no temporal support found, mark for removal
            if not has_temporal_support:
                isolated_objects.append(curr_idx)
        
        return isolated_objects

    def find_missing_objects(self, current_frame, before_frames, after_frames, iou_threshold):
        """Find objects that appear in before AND after frames but missing in current frame"""
        missing_objects = []
        
        # Collect all objects from before and after frames
        temporal_objects = []
        
        # Add objects from before frames
        for frame in before_frames:
            for i, box in enumerate(frame["boxes"]):
                temporal_objects.append({
                    "box": box,
                    "score": frame["scores"][i],
                    "label": frame["labels"][i],
                    "frame_type": "before"
                })
        
        # Add objects from after frames
        for frame in after_frames:
            for i, box in enumerate(frame["boxes"]):
                temporal_objects.append({
                    "box": box,
                    "score": frame["scores"][i],
                    "label": frame["labels"][i],
                    "frame_type": "after"
                })
        
        # Group temporal objects by spatial proximity and label
        object_groups = self.group_temporal_objects(temporal_objects, iou_threshold)
        
        # Find groups that have both before and after objects but no current match
        for group in object_groups:
            has_before = any(obj["frame_type"] == "before" for obj in group)
            has_after = any(obj["frame_type"] == "after" for obj in group)
            
            if has_before and has_after:
                # Check if this group has a match in current frame
                group_center = self.calculate_group_center(group)
                
                has_current_match = False
                for current_box in current_frame["boxes"]:
                    if self.bb_intersection_over_union(group_center["box"], current_box) > iou_threshold:
                        has_current_match = True
                        break
                
                if not has_current_match:
                    # Interpolate the missing object
                    interpolated_obj = self.interpolate_object(group)
                    missing_objects.append(interpolated_obj)
        
        return missing_objects

    def find_label_corrections(self, current_frame, before_frames, after_frames, iou_threshold):
        """Find objects in current frame that have wrong labels compared to temporal context"""
        label_corrections = []
        
        for curr_idx, curr_box in enumerate(current_frame["boxes"]):
            curr_label = current_frame["labels"][curr_idx]
            
            # Find matching objects in temporal frames
            temporal_labels = []
            
            # Check before frames
            for frame in before_frames:
                for i, box in enumerate(frame["boxes"]):
                    if self.bb_intersection_over_union(curr_box, box) > iou_threshold:
                        temporal_labels.append(frame["labels"][i])
            
            # Check after frames
            for frame in after_frames:
                for i, box in enumerate(frame["boxes"]):
                    if self.bb_intersection_over_union(curr_box, box) > iou_threshold:
                        temporal_labels.append(frame["labels"][i])
            
            # If we have temporal context, check for label consistency
            if len(temporal_labels) >= 2:  # Need at least 2 temporal matches
                # Find the most common label in temporal context
                label_counts = Counter(temporal_labels)
                most_common_label = label_counts.most_common(1)[0][0]
                most_common_count = label_counts.most_common(1)[0][1]
                
                # If all temporal labels agree and differ from current label
                if (most_common_count == len(temporal_labels) and 
                    most_common_label != curr_label):
                    label_corrections.append({
                        "box_idx": curr_idx,
                        "old_label": curr_label,
                        "new_label": most_common_label,
                        "confidence": most_common_count / len(temporal_labels)
                    })
        
        return label_corrections

    def group_temporal_objects(self, temporal_objects, iou_threshold):
        """Group temporal objects that likely represent the same object across frames"""
        groups = []
        used_indices = set()
        
        for i, obj1 in enumerate(temporal_objects):
            if i in used_indices:
                continue
                
            # Start a new group
            current_group = [obj1]
            used_indices.add(i)
            
            # Find similar objects
            for j, obj2 in enumerate(temporal_objects):
                if j in used_indices or i == j:
                    continue
                    
                # Check if objects are similar (same label and overlapping)
                if (obj1["label"] == obj2["label"] and 
                    self.bb_intersection_over_union(obj1["box"], obj2["box"]) > iou_threshold):
                    current_group.append(obj2)
                    used_indices.add(j)
            
            if len(current_group) > 1:  # Only keep groups with multiple objects
                groups.append(current_group)
        
        return groups

    def calculate_group_center(self, group):
        """Calculate the center position and average properties of a group"""
        boxes = [obj["box"] for obj in group]
        scores = [obj["score"] for obj in group]
        labels = [obj["label"] for obj in group]
        
        # Calculate average box coordinates
        avg_box = [
            sum(box[0] for box in boxes) / len(boxes),  # x1
            sum(box[1] for box in boxes) / len(boxes),  # y1
            sum(box[2] for box in boxes) / len(boxes),  # x2
            sum(box[3] for box in boxes) / len(boxes)   # y2
        ]
        
        # Calculate average score
        avg_score = sum(scores) / len(scores)
        
        # Use most common label
        most_common_label = Counter(labels).most_common(1)[0][0]
        
        return {
            "box": avg_box,
            "score": avg_score,
            "label": most_common_label
        }

    def interpolate_object(self, group):
        """Interpolate a missing object from a group of temporal detections"""
        # before_objects = [obj for obj in group if obj["frame_type"] == "before"]
        # after_objects = [obj for obj in group if obj["frame_type"] == "after"]
        
        # Calculate interpolated position (simple average)
        all_boxes = [obj["box"] for obj in group]
        interpolated_box = [
            sum(box[0] for box in all_boxes) / len(all_boxes),  # x1
            sum(box[1] for box in all_boxes) / len(all_boxes),  # y1
            sum(box[2] for box in all_boxes) / len(all_boxes),  # x2
            sum(box[3] for box in all_boxes) / len(all_boxes)   # y2
        ]
        
        # Calculate interpolated score (conservative estimate)
        all_scores = [obj["score"] for obj in group]
        interpolated_score = min(all_scores) * 0.8  # Reduce confidence for interpolated objects
        
        # Use most common label
        all_labels = [obj["label"] for obj in group]
        interpolated_label = Counter(all_labels).most_common(1)[0][0]
        
        return {
            "box": interpolated_box,
            "score": interpolated_score,
            "label": interpolated_label
        }
    
    def bb_intersection_over_union(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou