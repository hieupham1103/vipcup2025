import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path

class SegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, device, lr=1e-3):
        """
        Args:
            model: The model to train
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            device: Device to train on (cuda or cpu)
            lr: Learning rate
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, heatmaps) in enumerate(self.train_loader):
            images = images.to(self.device)
            heatmaps = heatmaps.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            # print(f'outputs shape: {outputs.shape}, heatmaps shape: {heatmaps.shape}')
            loss = self.criterion(outputs, heatmaps)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, heatmaps in self.val_loader:
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, heatmaps)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs, save_path='segmentation_model.pth'):
        """
        Args:
            num_epochs: Number of epochs to train for
            save_path: Path to save the best model
        """
        save_dir = os.path.dirname(save_path)
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            train_loss = self.train_epoch()
            
            val_loss = self.validate()
            
            self.scheduler.step()
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f'Model saved with val_loss: {val_loss:.4f}')