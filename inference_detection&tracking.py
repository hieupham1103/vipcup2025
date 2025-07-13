import os
import cv2
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict

from models import multiscale_model as multiscale
from models import track_model
from src.utils import visualize_tracking_video, visualize_detection_image


class SubmissionGenerator:
    def __init__(self, model_path, modality='FUSION', team_name='team', conf_threshold=0.3, iou_threshold=0.1, is_visualize=False):
        self.model_path = model_path
        self.modality = modality.upper()
        self.team_name = team_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.is_visualize = is_visualize
        
        # Initialize GPU models
        self.detection_model_gpu = multiscale.DetectionModel(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device="cuda"
        )
        
        self.tracking_model_gpu = track_model.TrackingModel(
            detection_model=self.detection_model_gpu
        )
        
        # Initialize CPU models
        self.detection_model_cpu = multiscale.DetectionModel(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device="cpu"
        )
        
        self.tracking_model_cpu = track_model.TrackingModel(
            detection_model=self.detection_model_cpu
        )
        
        # Class mapping (adjust based on your dataset)
        self.class_names = {0: 'bird', 1: 'drone'}  # Update this based on your classes
        
    def process_image(self, image_path_rgb, image_path_ir=None):
        """Process a single image pair (RGB + IR) and return detection results with GPU and CPU FPS"""
        image_rgb = cv2.imread(image_path_rgb)
        if image_rgb is None:
            return []
        
        if image_path_ir and os.path.exists(image_path_ir):
            image_ir = cv2.imread(image_path_ir)
            if image_ir is not None:
                images = [image_rgb, image_ir]
            else:
                images = image_rgb
        else:
            images = image_rgb

        # GPU inference
        start_time_gpu = time.time()
        detections_gpu = self.detection_model_gpu.image_detect(
            images,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )
        gpu_inference_time = (time.time() - start_time_gpu) * 1000  # Convert to ms
        gpu_fps = 1000.0 / gpu_inference_time if gpu_inference_time > 0 else 0
        
        # CPU inference (for FPS calculation)
        start_time_cpu = time.time()
        detections_cpu = self.detection_model_cpu.image_detect(
            images,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )
        cpu_inference_time = (time.time() - start_time_cpu) * 1000  # Convert to ms
        cpu_fps = 1000.0 / cpu_inference_time if cpu_inference_time > 0 else 0
        
        # Use GPU detections as primary results
        detections = detections_gpu
        
        if self.is_visualize:
            visualize_detection_image(
                image_path=image_path_rgb,
                detection_results=detections,
                output_path=f"images/visualized_{os.path.basename(image_path_rgb)}"
            )
        
        results = []
        frame_name = Path(image_path_rgb).stem  # filename without extension
        
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
                height, width = image_rgb.shape[:2]
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
                    'inference_time_detection (ms)': gpu_inference_time,
                    'confidence_track': 0,  # 0 for standalone images
                    'inference_time_track (ms)': 0,  # 0 for standalone images
                    'payload_label': 0,
                    'prob_harmful': 0,
                    'FPS (GPU)': gpu_fps,
                    'FPS (CPU)': cpu_fps
                }
                results.append(result)
        
        return results
    
    def process_video(self, video_path_rgb, video_path_ir=None):
        """Process a video pair (RGB + IR) and return tracking results with GPU and CPU FPS"""
        # Check if both videos exist for fusion
        video_paths = video_path_rgb
        if video_path_ir and os.path.exists(video_path_ir):
            video_paths = [video_path_rgb, video_path_ir]
        
        cap = cv2.VideoCapture(video_path_rgb)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # GPU tracking
        start_time_gpu = time.time()
        tracking_results_gpu = self.tracking_model_gpu.video_track(
            video_path=video_paths,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )
        total_time_gpu = time.time() - start_time_gpu
        
        # CPU tracking (for FPS calculation)
        start_time_cpu = time.time()
        tracking_results_cpu = self.tracking_model_cpu.video_track(
            video_path=video_paths,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )
        total_time_cpu = time.time() - start_time_cpu
        
        # Use GPU results as primary
        tracking_results = tracking_results_gpu
        
        # Calculate FPS for both GPU and CPU
        gpu_fps = total_frames / total_time_gpu if total_time_gpu > 0 else 0
        cpu_fps = total_frames / total_time_cpu if total_time_cpu > 0 else 0
        
        avg_detection_time_gpu = (total_time_gpu * 1000) / total_frames if total_frames > 0 else 0
        avg_tracking_time_gpu = avg_detection_time_gpu  # Same as detection for simplicity
        
        avg_detection_time_cpu = (total_time_cpu * 1000) / total_frames if total_frames > 0 else 0
        
        if self.is_visualize:
            visualize_tracking_video(
                video_path=video_path_rgb,
                tracking_frames=tracking_results,
                output_path=f"videos/visualized_rgb_{os.path.basename(video_path_rgb)}"
            )
            if video_path_ir and os.path.exists(video_path_ir):
                visualize_tracking_video(
                    video_path=video_path_ir,
                    tracking_frames=tracking_results,
                    output_path=f"videos/visualized_ir_{os.path.basename(video_path_ir)}"
                )
        
        results = []
        video_name = Path(video_path_rgb).stem
        
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
                        cap = cv2.VideoCapture(video_path_rgb)
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
                                'inference_time_detection (ms)': avg_detection_time_gpu,
                                'confidence_track': track['score'],  # Use same confidence
                                'inference_time_track (ms)': avg_tracking_time_gpu,
                                'payload_label': 0,
                                'prob_harmful': 0,
                                'FPS (GPU)': gpu_fps,
                                'FPS (CPU)': cpu_fps
                            }
                            results.append(result)
        
        return results
    
    def find_matching_ir_file(self, rgb_path, data_folder):
        """Find matching IR file for given RGB file"""
        rgb_name = Path(rgb_path).stem
        
        # Try different IR naming patterns
        ir_patterns = [
            f"IR_{rgb_name}",  # IR_BIRD_03897
            f"{rgb_name}_IR",  # BIRD_03897_IR
            rgb_name,          # Same name
        ]
        
        for pattern in ir_patterns:
            for ext in ['.mp4', '.avi', '.mov']:
                ir_path = Path(data_folder) / "IR" / f"{pattern}{ext}"
                if ir_path.exists():
                    return str(ir_path)
        
        return None

    def process_dataset(self, data_folder, output_path):
        """Process entire dataset and generate submission CSV"""
        all_results = []
        
        # Process images
        image_extensions = ['.png', '.jpg', '.jpeg']
        rgb_image_folder = Path(data_folder) / "RGB"
        ir_image_folder = Path(data_folder) / "IR"
        
        if rgb_image_folder.exists():
            for ext in image_extensions:
                for rgb_image_path in rgb_image_folder.rglob(f"*{ext}"):
                    # Find matching IR image
                    rgb_name = rgb_image_path.stem
                    ir_image_path = None
                    
                    if ir_image_folder.exists():
                        for ir_pattern in [f"IR_{rgb_name}", f"{rgb_name}_IR", rgb_name]:
                            for ir_ext in image_extensions:
                                potential_ir = ir_image_folder / f"{ir_pattern}{ir_ext}"
                                if potential_ir.exists():
                                    ir_image_path = str(potential_ir)
                                    break
                            if ir_image_path:
                                break
                    
                    results = self.process_image(str(rgb_image_path), ir_image_path)
                    all_results.extend(results)
        
        # Process videos
        video_extensions = ['.mp4', '.avi', '.mov']
        rgb_video_folder = Path(data_folder) / "RGB"
        ir_video_folder = Path(data_folder) / "IR"
        
        if rgb_video_folder.exists():
            for ext in video_extensions:
                for rgb_video_path in rgb_video_folder.rglob(f"*{ext}"):
                    print(f"Processing RGB video: {rgb_video_path}")
                    
                    # Find matching IR video
                    ir_video_path = self.find_matching_ir_file(str(rgb_video_path), data_folder)
                    if ir_video_path:
                        print(f"Found matching IR video: {ir_video_path}")
                    else:
                        print(f"No matching IR video found for {rgb_video_path}")
                    
                    results = self.process_video(str(rgb_video_path), ir_video_path)
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
            'payload_label', 'prob_harmful', 'FPS (GPU)', 'FPS (CPU)'
        ]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Format floating point numbers
                for key in ['x_min_norm', 'y_min_norm', 'x_max_norm', 'y_max_norm',
                           'confidence_detection', 'confidence_track', 'prob_harmful',
                           'FPS (GPU)', 'FPS (CPU)']:
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
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--team_name', type=str, default='team',
                        help='Team name for submission files')
    parser.add_argument('--output_dir', type=str, default='submissions',
                        help='Output directory for submission files')
    parser.add_argument('--modality', type=str, choices=['IR', 'RGB', 'FUSION'], default='FUSION')
    parser.add_argument("--visualize", action='store_true')
    # modality thresholds
    parser.add_argument('--conf', type=float, default=0.2,
                        help='Confidence threshold for model')
    parser.add_argument('--iou', type=float, default=0.1,
                        help='IoU threshold for model')
    
    args = parser.parse_args()
    
    # Generate submission
    print(f"Generating {args.modality} submission with conf={args.conf}, iou={args.iou}...")
    generator = SubmissionGenerator(
        model_path=args.model,
        modality=args.modality,
        team_name=args.team_name,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        is_visualize=args.visualize
    )
    output_file = os.path.join(args.output_dir, f"{args.team_name}_{args.modality}_submission.csv")
    generator.process_dataset(args.data_folder, output_file)
    
    print("All submission files generated successfully!")


if __name__ == "__main__":
    main()