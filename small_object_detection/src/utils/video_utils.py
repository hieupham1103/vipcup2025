"""PyTorch-based background subtraction module for in-memory video processing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from typing import Union, Tuple, Optional, List
from tqdm import tqdm
from pathlib import Path


def load_video_to_numpy(video_path: str,
                        max_frames: Optional[int] = None, 
                        output_rgb: bool = True,
                        normalize: bool = True,
                        target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load a video file into a numpy array.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to load (None = all frames)
        output_rgb: Whether to return RGB (True) or grayscale (False)
        normalize: Whether to normalize pixel values to [0, 1]
        target_size: Optional (width, height) to resize frames
        
    Returns:
        Video as numpy array with shape:
          - If output_rgb=True: [T, H, W, 3] (RGB format)
          - If output_rgb=False: [T, H, W] (grayscale)
          
    Raises:
        FileNotFoundError: If the video file doesn't exist
        IOError: If the video cannot be opened
    """
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    try:
        frames = []
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if max_frames is not None and count >= max_frames:
                break
            
            # Resize if target size is specified
            if target_size is not None:
                frame = cv2.resize(frame, target_size)
            
            # Convert to RGB or grayscale
            if output_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frames.append(frame)
            count += 1
        
        if len(frames) == 0:
            raise ValueError("No frames were read from the video")
            
        video_array = np.stack(frames)
        if output_rgb:
            video_array = np.transpose(video_array, (0, 3, 1, 2))
        if normalize:
            video_array = video_array.astype(np.float32) / 255.0
            
        return video_array
        
    finally:
        cap.release()


def save_video_from_numpy(video_array,
                          output_path: str,
                          fps: float = 30.0, 
                         is_normalized: bool = True):
    """
    Save a numpy array as a video file.
    
    Args:
        video_array: Video as numpy array
          - If shape is [T, H, W, 3]: Treated as RGB
          - If shape is [T, H, W]: Treated as grayscale
          - If shape is [T, 3, H, W]: Treated as RGB in channel-first format
        output_path: Path to save the video
        fps: Frames per second
        is_normalized: Whether the input array is normalized to [0, 1]
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check input shape and convert if necessary
    if isinstance(video_array, torch.Tensor):
        video_array = video_array.cpu().numpy()
    
    if video_array.ndim == 4:
        if video_array.shape[1] == 3:
            is_color = True
            video_array = np.transpose(video_array, (0, 2, 3, 1))
        elif video_array.shape[1] == 1:
            is_color = False
            video_array = video_array.squeeze(axis=1)
        else:
            raise ValueError(f"Unsupported video shape: {video_array.shape}")
    elif video_array.ndim == 3:
        if video_array.shape[2] == 3:
            is_color = True
        elif video_array.shape[2] == 1:
            is_color = False
            video_array = video_array.squeeze(axis=2)
        else:
            raise ValueError(f"Unsupported video shape: {video_array.shape}")
    # if video_array.ndim == 4:
    #     if video_array.shape[1] == 3:  # [T, 3, H, W]
    #         if video_array.shape[1] == 1:
    #             video_array = video_array.squeeze(axis=1)
    #             is_color = False
    #         else:  # [T, 3, H, W] - RGB format
    #             is_color = True
    #             video_array = np.transpose(video_array, (0, 2, 3, 1))        
    #     else:  # [T, H, W, 3]
    #         if video_array.shape[3] == 1:  # [T, H, W, 1] - treat as grayscale
    #             video_array = video_array.squeeze(axis=-1)
    #             is_color = False
    #         else:
    #             is_color = True
    # else:  # [T, H, W]
    #     is_color = False
    print(f"Input video shape: {video_array.shape}, is_color: {is_color}")
    if is_color:
        T, H, W, C = video_array.shape
    else:
        T, H, W = video_array.shape
    
    # Convert to uint8 for video writing
    if is_normalized:
        video_array = np.clip(video_array * 255.0, 0, 255).astype(np.uint8)
    else:
        video_array = np.clip(video_array, 0, 255).astype(np.uint8)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H), isColor=is_color)
    
    for i in range(T):
        if is_color:
            frame = cv2.cvtColor(video_array[i], cv2.COLOR_RGB2BGR)
        else:
            frame = video_array[i]
        
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_path}")


def save_video_with_bboxes(video_tensor: torch.Tensor,
                            bboxes_per_frame: List[List[Tuple[int, int, int, int]]],
                            output_path: str,
                            fps: int = 30):
    """
    Tạo video từ tensor ảnh + vẽ bounding box lên từng frame.

    Args:
        video_tensor: torch.Tensor shape (T, 3, H, W) hoặc (T, H, W, 3)
        bboxes_per_frame: list length T, mỗi phần tử là list bbox (x, y, w, h)
        output_path: đường dẫn file .mp4
        fps: frame per second của video
    """
    T = len(bboxes_per_frame)

    if isinstance(video_tensor, torch.Tensor):
        if video_tensor.dim() == 4 and video_tensor.shape[1] == 3:
            # Convert từ (T, C, H, W) → (T, H, W, C)
            video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()
        elif video_tensor.dim() == 4 and video_tensor.shape[-1] == 3:
            video_np = video_tensor.cpu().numpy()
        else:
            raise ValueError("video_tensor phải có shape (T, 3, H, W) hoặc (T, H, W, 3)")
    else:
        video_np = video_tensor  # assume np.ndarray (T, H, W, 3)

    H, W = video_np.shape[1:3]

    # Chuyển về uint8 nếu chưa có
    if video_np.dtype != np.uint8:
        video_np = (np.clip(video_np, 0, 1) * 255).astype(np.uint8)

    # Ghi video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for t in tqdm(range(T), desc="Saving video"):
        frame = video_np[t]  # shape (H, W, 3), dtype uint8

        frame_copy = frame.copy()

        # Vẽ bbox lên frame
        if len(bboxes_per_frame[t][0]) > 0: 
            for (x, y, w, h) in bboxes_per_frame[t][0]:
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # red box

        writer.write(frame_copy)

    writer.release()
    print(f"[✓] Video saved to {output_path}")