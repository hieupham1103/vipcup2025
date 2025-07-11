import os
import cv2
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict

from models import multiscale_model as multiscale
from models import track_model


class SubmissionGenerator:
    def __init__(self, model_path, modality='IR', team_name='team', conf_threshold=0.3, iou_threshold=0.1):
        self.model_path = model_path
        self.modality = modality.upper()
        self.team_name = team_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize models
        self.detection_model = multiscale.DetectionModel(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device="cuda"
        )
        
        self.tracking_model = track_model.TrackingModel(
            detection_model=self.detection_model
        )
        
        # Class mapping (adjust based on your dataset)
        self.class_names = {0: 'bird', 1: 'drone'}  # Update this based on your classes
        
    def process_image(self, image_path):
        """Process a single image and return detection results"""
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        start_time = time.time()
        detections = self.detection_model.image_detect(
            image,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        results = []
        frame_name = Path(image_path).stem  # filename without extension
        
        # Group detections by class for proper track_id assignment
        class_detections = defaultdict(list)
        for i, (box, score, label) in enumerate(zip(
            detections["boxes"], 
            detections["scores"], 
            detections["labels"]
        )):
            class_detections[int(label)].append((box, score, i))
        
        # Process each class separately
        for class_id, class_dets in class_detections.items():
            for track_id, (box, score, _) in enumerate(class_dets, 1):
                # Normalize coordinates
                height, width = image.shape[:2]
                x_min_norm = box[0] / width
                y_min_norm = box[1] / height
                x_max_norm = box[2] / width
                y_max_norm = box[3] / height
                
                result = {
                    'Frame_name': frame_name,
                    'Usage': 'public',
                    'track_id': 0,  # 0 for standalone images
                    'x_min_norm': x_min_norm,
                    'y_min_norm': y_min_norm,
                    'x_max_norm': x_max_norm,
                    'y_max_norm': y_max_norm,
                    'class_label': self.class_names.get(class_id, 'unknown'),
                    'direction': 0,  # 0 for standalone images
                    'confidence_detection': score,
                    'inference_time_detection (ms)': inference_time,
                    'confidence_track': 0,  # 0 for standalone images
                    'inference_time_track (ms)': 0,  # 0 for standalone images
                    'payload_label': 0,
                    'prob_harmful': 0
                }
                results.append(result)
        
        return results
    
    def process_video(self, video_path):
        """Process a video and return tracking results"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        start_time = time.time()
        tracking_results = self.tracking_model.video_track(
            video_path=video_path,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )
        total_time = time.time() - start_time
        
        # Calculate FPS as per competition requirements
        computed_fps = total_frames / total_time if total_time > 0 else 0
        avg_detection_time = (total_time * 1000) / total_frames if total_frames > 0 else 0
        avg_tracking_time = avg_detection_time  # Same as detection for simplicity
        
        results = []
        video_name = Path(video_path).stem
        
        # Group tracking results by frame
        frame_results = defaultdict(list)
        for track_result in tracking_results:
            frame_idx = track_result['frame_idx']
            frame_results[frame_idx].append(track_result)
        
        # Process each frame
        for frame_idx in range(total_frames):
            frame_name = f"{video_name}_{frame_idx+1:03d}"
            
            if frame_idx in frame_results:
                frame_tracks = frame_results[frame_idx]
                
                # Group by class for proper track_id assignment
                class_tracks = defaultdict(list)
                for track in frame_tracks:
                    class_id = int(track['label'])
                    class_tracks[class_id].append(track)
                
                # Process each class separately
                for class_id, tracks in class_tracks.items():
                    # Sort tracks by original track_id to maintain consistency
                    tracks.sort(key=lambda x: x['track_id'])
                    
                    for new_track_id, track in enumerate(tracks, 1):
                        # Get frame dimensions for normalization
                        cap = cv2.VideoCapture(video_path)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            height, width = frame.shape[:2]
                            box = track['bbox']
                            
                            # Normalize coordinates
                            x_min_norm = box[0] / width
                            y_min_norm = box[1] / height
                            x_max_norm = box[2] / width
                            y_max_norm = box[3] / height
                            
                            # Get direction
                            direction = track.get('motion', 'unknown')
                            if direction == 'unknown':
                                direction = 0
                            
                            result = {
                                'Frame_name': frame_name,
                                'Usage': 'public',
                                'track_id': new_track_id,
                                'x_min_norm': x_min_norm,
                                'y_min_norm': y_min_norm,
                                'x_max_norm': x_max_norm,
                                'y_max_norm': y_max_norm,
                                'class_label': self.class_names.get(class_id, 'unknown'),
                                'direction': direction,
                                'confidence_detection': track['score'],
                                'inference_time_detection (ms)': avg_detection_time,
                                'confidence_track': track['score'],  # Use same confidence
                                'inference_time_track (ms)': avg_tracking_time,
                                'payload_label': 0,
                                'prob_harmful': 0
                            }
                            results.append(result)
        
        return results
    
    def process_dataset(self, data_folder, output_path):
        """Process entire dataset and generate submission CSV"""
        all_results = []
        
        # Process images
        image_extensions = ['.png', '.jpg', '.jpeg']
        for ext in image_extensions:
            for image_path in Path(data_folder).rglob(f"*{ext}"):
                if self.modality.lower() in str(image_path).lower():
                    print(f"Processing image: {image_path}")
                    results = self.process_image(str(image_path))
                    all_results.extend(results)
        
        # Process videos
        video_extensions = ['.mp4', '.avi', '.mov']
        for ext in video_extensions:
            for video_path in Path(data_folder).rglob(f"*{ext}"):
                if self.modality.lower() in str(video_path).lower():
                    print(f"Processing video: {video_path}")
                    results = self.process_video(str(video_path))
                    all_results.extend(results)
        
        # Write to CSV
        self.write_csv(all_results, output_path)
        print(f"Submission file saved to: {output_path}")
    
    def write_csv(self, results, output_path):
        """Write results to CSV file"""
        fieldnames = [
            'Frame_name', 'Usage', 'track_id', 'x_min_norm', 'y_min_norm',
            'x_max_norm', 'y_max_norm', 'class_label', 'direction',
            'confidence_detection', 'inference_time_detection (ms)',
            'confidence_track', 'inference_time_track (ms)',
            'payload_label', 'prob_harmful'
        ]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Format floating point numbers
                for key in ['x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm',
                           'confidence_detection', 'confidence_track', 'prob_harmful']:
                    if key in result and isinstance(result[key], (int, float)):
                        result[key] = f"{result[key]:.4f}"
                
                for key in ['inference_time_detection (ms)', 'inference_time_track (ms)']:
                    if key in result and isinstance(result[key], (int, float)):
                        result[key] = f"{result[key]:.1f}"
                
                writer.writerow(result)


def main():
    parser = argparse.ArgumentParser(description='Generate submission files for VIP Cup 2025')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to test data folder')
    parser.add_argument('--ir_model', type=str, required=True,
                        help='Path to IR model checkpoint')
    parser.add_argument('--rgb_model', type=str, required=True,
                        help='Path to RGB model checkpoint')
    parser.add_argument('--team_name', type=str, default='team',
                        help='Team name for submission files')
    parser.add_argument('--output_dir', type=str, default='submissions',
                        help='Output directory for submission files')
    
    # IR modality thresholds
    parser.add_argument('--ir_conf', type=float, default=0.3,
                        help='Confidence threshold for IR modality')
    parser.add_argument('--ir_iou', type=float, default=0.1,
                        help='IoU threshold for IR modality')
    
    # RGB modality thresholds
    parser.add_argument('--rgb_conf', type=float, default=0.3,
                        help='Confidence threshold for RGB modality')
    parser.add_argument('--rgb_iou', type=float, default=0.1,
                        help='IoU threshold for RGB modality')
    
    args = parser.parse_args()
    
    # Generate IR submission
    # print(f"Generating IR submission with conf={args.ir_conf}, iou={args.ir_iou}...")
    # ir_generator = SubmissionGenerator(
    #     model_path=args.ir_model,
    #     modality='IR',
    #     team_name=args.team_name,
    #     conf_threshold=args.ir_conf,
    #     iou_threshold=args.ir_iou
    # )
    # ir_output = os.path.join(args.output_dir, f"{args.team_name}_IR_submission.csv")
    # ir_generator.process_dataset(args.data_folder, ir_output)
    
    # Generate RGB submission
    print(f"Generating RGB submission with conf={args.rgb_conf}, iou={args.rgb_iou}...")
    rgb_generator = SubmissionGenerator(
        model_path=args.rgb_model,
        modality='RGB',
        team_name=args.team_name,
        conf_threshold=args.rgb_conf,
        iou_threshold=args.rgb_iou
    )
    rgb_output = os.path.join(args.output_dir, f"{args.team_name}_RGB_submission.csv")
    rgb_generator.process_dataset(args.data_folder, rgb_output)
    
    print("All submission files generated successfully!")


if __name__ == "__main__":
    main()