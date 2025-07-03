import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from pathlib import Path
from src import *
from src.dataset import *


def main():
    # Configuration
    IMAGE_DIR = 'data/track_video/split_A/IR/train/train_images'
    LABEL_DIR = 'data/track_video/split_A/IR/train/train_labels'
    load_checkpoint_path = 'checkpoints/rgb_model_1.pth'
    output_checkpoint_path = 'checkpoints/rgb_model_2.pth'
    IMAGE_SIZE = (320, 256)
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    DEVICE = torch.device('cuda:1')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Creating dataset...")
    full_dataset = YOLOToHeatmapDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        image_size=IMAGE_SIZE,
        heatmap_sigma=0.5,
        transform=transform,
        gray_img=True,
    )
    print(f"Dataset created with {len(full_dataset)} samples")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    
    model = SegmentationModel(input_channels=1, out_channels=1)
    checkpoint = torch.load(load_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    trainer = SegmentationTrainer(model, train_loader, val_loader, DEVICE, LEARNING_RATE)
    
    print(f'Starting training on {DEVICE}')
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    trainer.train(NUM_EPOCHS, save_path=output_checkpoint_path)

if __name__ == '__main__':
    main()