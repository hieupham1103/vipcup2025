import torch
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import cv2
import glob

from .evaluate import evaluate_frame
from .detector import SmallDetectorModule
from .utils.video_utils import load_video_to_numpy

class DetectionEvaluator:
    """Simple evaluator for object detection model on video datasets"""
    
    def __init__(
        self,
        model: SmallDetectorModule,
        video_dir: str,
        label_dir: str,
        device: torch.device = None,
        iou_threshold: float = 0.5,
        detection_threshold: float = 0.5,
        min_area: int = 1,
    ):
        """
        Initialize the evaluator with model and dataset information.
        
        Args:
            model: The detection model to evaluate
            video_dir: Directory containing video files
            label_dir: Directory containing YOLO format labels
            device: Device to run inference on
            iou_threshold: IoU threshold for considering a detection as correct
            detection_threshold: Confidence threshold for detection
            min_area: Minimum area for bounding boxes
        """
        self.model = model
        self.video_dir = Path(video_dir)
        self.label_dir = Path(label_dir)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.iou_threshold = iou_threshold
        self.detection_threshold = detection_threshold
        self.min_area = min_area
        
        self.model.to(self.device)
        self.model.eval()
        
        # Find all video files
        self._find_videos()
        
    def _find_videos(self):
        """Find all videos in the video directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        # Find all video files in the directory
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(self.video_dir.glob(f'*{ext}')))
        
        if not video_files:
            raise ValueError(f"No video files found in {self.video_dir}")
            
        self.video_paths = sorted(video_files)
        self.videos = [v.stem for v in self.video_paths]
        
        print(f"Found {len(self.videos)} videos in {self.video_dir}")
    
    def load_video(self, video_path: str) -> torch.Tensor:
        """
        Load a video file as a tensor.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tensor of video frames [T, C, H, W]
        """
        print(f"Loading video: {video_path}")
        # Use the provided utility to load video
        output_rgb = not self.model.gray_img  # If model expects gray, load as gray
        
        try:
            video_array = load_video_to_numpy(
                video_path,
                output_rgb=output_rgb,
                normalize=True
            )
            
            # Convert to tensor
            if output_rgb:
                # Convert from [T, H, W, C] to [T, C, H, W]
                video_tensor = torch.from_numpy(np.transpose(video_array, (0, 3, 1, 2))).float()
            else:
                # Add channel dimension [T, H, W] -> [T, 1, H, W]
                video_tensor = torch.from_numpy(video_array[:, np.newaxis, :, :]).float()
                
            return video_tensor
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            raise
    
    def load_ground_truth_for_video(self, video_name: str) -> List[List[Tuple[int, int, int, int]]]:
        """
        Load ground truth bounding boxes for a video.
        
        Args:
            video_name: Name of the video file (without extension)
            
        Returns:
            List of ground truth bounding boxes for each frame
        """
        print(f"Loading ground truth for video: {video_name}")
        
        # Get video properties to convert normalized YOLO coordinates to pixel coordinates
        video_path = None
        for path in self.video_paths:
            if path.stem == video_name:
                video_path = path
                break
        
        if video_path is None:
            raise ValueError(f"Video file '{video_name}' not found in {self.video_dir}")
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Debug info
        print(f"Video dimensions: {frame_width}x{frame_height}, {frame_count} frames")
        
        # Check if we have frame-by-frame labels in a directory structure
        # Expected pattern: video_name_frame_XXXX.txt
        gt_boxes_all_frames = [[] for _ in range(frame_count)]
        label_pattern = str(self.label_dir / f"{video_name}/*.txt")
        label_files = glob.glob(label_pattern)
        # print(label_files)
        # exit()
        if label_files:
            print(f"Found {len(label_files)} label files using pattern {label_pattern}")
            
            for label_file in label_files:
                # Extract frame index from filename
                frame_file = Path(label_file).stem
                    
                try:
                    frame_idx = int(frame_file)
                    if frame_idx >= frame_count:
                        print(f"Warning: frame index {frame_idx} exceeds video length {frame_count}")
                        continue
                        
                    # Read labels
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    frame_gt_boxes = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Extract normalized coordinates (skip class_id at parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Convert normalized YOLO format to pixel coordinates
                            x_pixel = int((x_center - width/2) * frame_width)
                            y_pixel = int((y_center - height/2) * frame_height)
                            width_pixel = int(width * frame_width)
                            height_pixel = int(height * frame_height)
                            
                            # Store as (x, y, w, h) format
                            frame_gt_boxes.append((x_pixel, y_pixel, width_pixel, height_pixel))
                    
                    gt_boxes_all_frames[frame_idx] = frame_gt_boxes
                    
                except ValueError as e:
                    print(f"Error parsing frame index from {frame_file}: {e}")
                    continue
            
            return gt_boxes_all_frames
        else:
            # If no frame-by-frame labels found, check for a single label file
            print("No frame-by-frame labels found, checking for single label file")
            single_label_file = self.label_dir / f"{video_name}.txt"
            
            if single_label_file.exists():
                print(f"Found single label file: {single_label_file}")
                
                with open(single_label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 6:  # Expected format: frame_idx class_id x y w h
                        try:
                            frame_idx = int(parts[0])
                            if frame_idx < frame_count:
                                # Skip class_id (parts[1])
                                x_center, y_center, width, height = map(float, parts[2:6])
                                
                                # Convert normalized YOLO format to pixel coordinates
                                x_pixel = int((x_center - width/2) * frame_width)
                                y_pixel = int((y_center - height/2) * frame_height)
                                width_pixel = int(width * frame_width)
                                height_pixel = int(height * frame_height)
                                
                                gt_boxes_all_frames[frame_idx].append((x_pixel, y_pixel, width_pixel, height_pixel))
                        except ValueError as e:
                            print(f"Error parsing line: {line}. Error: {e}")
                            continue
                
                return gt_boxes_all_frames
            else:
                print(f"Warning: No label files found for video {video_name}")
                return []
    
    def evaluate_video(self, video_name: str) -> Dict[str, float]:
        """
        Evaluate model on a single video.
        
        Args:
            video_name: Name of the video
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating video: {video_name}")
        
        # Find video file
        video_path = None
        for v_path in self.video_paths:
            if v_path.stem == video_name:
                video_path = v_path
                break
        
        if video_path is None:
            raise ValueError(f"Video {video_name} not found in {self.video_dir}")
        
        # Load video
        video_tensor = self.load_video(str(video_path))
        
        # Load ground truth boxes
        gt_boxes_all_frames = self.load_ground_truth_for_video(video_name)
        
        # If no ground truth found, we can't evaluate this video
        if not gt_boxes_all_frames:
            print(f"Warning: No ground truth found for video {video_name}, skipping evaluation")
            return {
                'video_name': video_name,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'frame_count': len(video_tensor),
                'error': 'No ground truth found'
            }
        
        # Move video to device
        video_tensor = video_tensor.to(self.device)
        
        # Run inference
        print("Running inference...")
        with torch.no_grad():
            try:
                heatmaps, bboxes_list = self.model(video_tensor, 
                                          threshold=self.detection_threshold,
                                          min_area=self.min_area)
            except Exception as e:
                print(f"Error during inference: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'video_name': video_name,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'tp': 0,
                    'fp': 0,
                    'fn': 0,
                    'frame_count': len(video_tensor),
                    'error': str(e)
                }
        
        # Debug info
        print(f"Video has {len(video_tensor)} frames")
        print(f"Got {len(bboxes_list)} bbox predictions")
        print(f"Ground truth has {len(gt_boxes_all_frames)} frames")
        
        # Ensure prediction list matches ground truth length
        eval_length = min(len(bboxes_list), len(gt_boxes_all_frames))
        
        # Evaluate each frame
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for i in range(eval_length):
            pred_boxes = bboxes_list[i]
            gt_boxes = gt_boxes_all_frames[i]
            
            if len(pred_boxes) == 0:
                pred_frame_boxes = []
            else:
                # Flatten if nested (since model returns nested list)
                if isinstance(pred_boxes, list) and len(pred_boxes) > 0 and isinstance(pred_boxes[0], list):
                    pred_frame_boxes = pred_boxes[0]
                else:
                    pred_frame_boxes = pred_boxes
            
            # Debug info for the first few frames
            if i < 3:
                print(f"Frame {i}: {len(pred_frame_boxes)} predictions, {len(gt_boxes)} ground truths")
                if pred_frame_boxes:
                    print(f"  First prediction: {pred_frame_boxes[0]}")
                if gt_boxes:
                    print(f"  First ground truth: {gt_boxes[0]}")
            
            # Evaluate this frame
            tp, fp, fn = evaluate_frame(
                pred_bboxes=pred_frame_boxes,
                gt_bboxes=gt_boxes,
                iou_thresh=self.iou_threshold
            )
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Calculate overall metrics
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        metrics = {
            'video_name': video_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'frame_count': len(video_tensor)
        }
        
        return metrics
    
    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on all videos.
        
        Returns:
            Dictionary containing evaluation metrics for each video
            and overall average metrics
        """
        all_results = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for video_name in tqdm(self.videos, desc="Evaluating videos"):
            metrics = self.evaluate_video(video_name)
            all_results[video_name] = metrics
            
            total_tp += metrics.get('tp', 0)
            total_fp += metrics.get('fp', 0)
            total_fn += metrics.get('fn', 0)
        
        # Calculate overall average metrics
        avg_precision = total_tp / max(total_tp + total_fp, 1)
        avg_recall = total_tp / max(total_tp + total_fn, 1)
        avg_f1 = 2 * avg_precision * avg_recall / max(avg_precision + avg_recall, 1e-6)
        
        overall_metrics = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }
        
        all_results['overall'] = overall_metrics
        
        return all_results
    
    def generate_report(self, results: Dict, output_file: str = None):
        """
        Generate a text report of evaluation results.
        
        Args:
            results: Dictionary of evaluation results
            output_file: Path to save report (None for stdout only)
        """
        report = []
        report.append("=" * 50)
        report.append("DETECTION EVALUATION REPORT")
        report.append("=" * 50)
        report.append(f"IoU Threshold: {self.iou_threshold}")
        report.append(f"Detection Threshold: {self.detection_threshold}")
        report.append(f"Min Box Area: {self.min_area}")
        report.append("-" * 50)
        
        # Overall results
        overall = results['overall']
        report.append("OVERALL RESULTS:")
        report.append(f"Precision: {overall['precision']:.4f}")
        report.append(f"Recall: {overall['recall']:.4f}")
        report.append(f"F1-Score: {overall['f1']:.4f}")
        report.append(f"True Positives: {overall['tp']}")
        report.append(f"False Positives: {overall['fp']}")
        report.append(f"False Negatives: {overall['fn']}")
        report.append("-" * 50)
        
        # Per-video results
        report.append("PER-VIDEO RESULTS:")
        for video_name, metrics in results.items():
            if video_name == 'overall':
                continue
                
            report.append(f"\nVideo: {video_name}")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall: {metrics['recall']:.4f}")
            report.append(f"  F1-Score: {metrics['f1']:.4f}")
            report.append(f"  True Positives: {metrics['tp']}")
            report.append(f"  False Positives: {metrics['fp']}")
            report.append(f"  False Negatives: {metrics['fn']}")
            if 'error' in metrics:
                report.append(f"  ERROR: {metrics['error']}")
        
        report_text = "\n".join(report)
        
        # Print to console
        print(report_text)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text