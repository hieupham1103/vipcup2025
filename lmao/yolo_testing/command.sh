python scripts/split_data.py \
  --base_path /home/cvpr2025/yolo_testing/data/vipcup_det/split_A/RGB \
  --train_dir images/train \
  --test_dir images/test \
  --val_ratio 0.2 \
  --nc 2 \
  --names BIRD DRONE \
  --output_dir /home/cvpr2025/yolo_testing/config/Detection/RGB
