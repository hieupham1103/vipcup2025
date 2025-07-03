"""
Evaluate object detection model on videos directly.

Usage:
    python eval.py --video_dir data/track_video/split_A/IR/test/videos \
                  --label_dir data/track_video/split_A/IR/test/labels \
                  --model_path checkpoints/rgb_model_1.pth \
                  --iou_threshold 0.1
"""
import torch
import argparse
import os
from pathlib import Path
from src.evaluator import DetectionEvaluator
from src.detector import SmallDetectorModule

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate object detection model on videos")
    parser.add_argument("--video_dir", type=str,
                        default="data/track_video/split_A/RGB/test/videos",
                        help="Directory containing video files")
    parser.add_argument("--label_dir", type=str, 
                        default="data/track_video/split_A/RGB/test/labels",
                        help="Directory containing ground truth labels")
    parser.add_argument("--model_path", type=str,
                        default="checkpoints/model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for evaluation")
    parser.add_argument("--detection_threshold", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--min_area", type=int, default=1,
                        help="Minimum area for bounding boxes")
    parser.add_argument("--window_size", type=int, default=15,
                        help="Window size for background subtraction")
    parser.add_argument("--gray_img", action="store_true", default=True,
                        help="Use grayscale images")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--video", type=str, default=None,
                        help="Evaluate specific video (filename without extension)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Print diagnostics
    print("Running evaluation with settings:")
    print(f"  Video directory: {args.video_dir}")
    print(f"  Label directory: {args.label_dir}")
    print(f"  Model path: {args.model_path}")
    print(f"  IoU threshold: {args.iou_threshold}")
    print(f"  Detection threshold: {args.detection_threshold}")
    print(f"  Grayscale input: {args.gray_img}")
    print(f"  Window size: {args.window_size}")
    
    # Check if directories exist
    if not os.path.exists(args.video_dir):
        raise ValueError(f"Video directory does not exist: {args.video_dir}")
    if not os.path.exists(args.label_dir):
        raise ValueError(f"Label directory does not exist: {args.label_dir}")
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model file does not exist: {args.model_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SmallDetectorModule(
        gray_img=args.gray_img, 
        window_size=args.window_size
    )
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.SegmentationModel.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {args.model_path}")
    
    # Create evaluator
    evaluator = DetectionEvaluator(
        model=model,
        video_dir=args.video_dir,
        label_dir=args.label_dir,
        device=device,
        iou_threshold=args.iou_threshold,
        detection_threshold=args.detection_threshold,
        min_area=args.min_area
    )
    
    # Evaluate specific video if requested
    if args.video:
        print(f"Evaluating single video: {args.video}")
        metrics = evaluator.evaluate_video(args.video)
        results = {args.video: metrics, 'overall': metrics}
    else:
        # Run evaluation on all videos
        results = evaluator.evaluate_all()
    
    # Generate report
    report_file = output_dir / f"{Path(args.model_path).stem}_evaluation_report.txt"
    evaluator.generate_report(results, str(report_file))

if __name__ == "__main__":
    main()