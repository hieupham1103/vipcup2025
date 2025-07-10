python submit.py \
    --data_folder data/Test_detection_tracking \
    --ir_model checkpoints/IR/yolov8n/best.pt \
    --rgb_model checkpoints/RGB/yolov8n2/best.pt \
    --team_name your_team_name \
    --output_dir submissions \
    --ir_conf 0.1 \
    --ir_iou 0.1 \
    --rgb_conf 0.1 \
    --rgb_iou 0.1