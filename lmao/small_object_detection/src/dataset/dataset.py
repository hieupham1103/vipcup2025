import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path



class YOLOToHeatmapDataset(Dataset):
    def __init__(self, image_dir, label_dir, gray_img=True, image_size=(512, 512), heatmap_sigma=10, transform=None):
        """
        Dataset that converts YOLO format labels to heatmaps for segmentation training.
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing YOLO format label files (.txt)
            image_size: Target image size (H, W)
            heatmap_sigma: Gaussian sigma for heatmap generation
            transform: Image transforms
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma
        self.transform = transform
        self.gray_img = gray_img
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(self.image_dir.glob(ext))
        # print(image_dir)
        # Filter out images without corresponding label files
        self.valid_files = []
        for img_file in self.image_files:
            label_file = self.label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                self.valid_files.append(img_file)
        
        print(f"Found {len(self.valid_files)} image-label pairs")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        img_path = self.valid_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        if self.gray_img:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        image = image.resize(self.image_size)
        
        heatmap = self.create_heatmap_from_yolo(label_path, orig_w, orig_h)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, heatmap
    
    def create_heatmap_from_yolo(self, label_path, orig_w, orig_h):
        heatmap_w, heatmap_h = self.image_size[0] // 2, self.image_size[1] // 2
        heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                    
                    x_center_hm = x_center * heatmap_w
                    y_center_hm = y_center * heatmap_h
                    
                    heatmap = self.add_gaussian_blob(heatmap,
                                                     x_center_hm, y_center_hm,
                                                     max(2,self.heatmap_sigma * (width + height) / 5)
                                                     )
        
        return torch.FloatTensor(heatmap).unsqueeze(0)
    
    def add_gaussian_blob(self, heatmap, x, y, sigma):
        h, w = heatmap.shape
        
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        
        heatmap = np.maximum(heatmap, gaussian)
        
        return heatmap
