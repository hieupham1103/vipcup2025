
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from . import *
import cv2


class SmallDetectorModule(nn.Module):
    def __init__(self, 
                 gray_img: bool = True,
                 window_size: int = 15,
                 ):
        super(SmallDetectorModule, self).__init__()
        
        self.window_size = window_size
        self.gray_img = gray_img
        
        self.BackgroundSubtractor = BackgroundSubtractorModule(
            window_size=window_size,
            gray_img=gray_img,
            normalize_output=True
        )
        self.SegmentationModel = SegmentationModel(
            input_channels=(1 if gray_img else 3),
            out_channels=1
        )
    def heatmap_to_image_bboxes(self,
                bboxes_hm: List[Tuple[int,int,int,int]],
                heatmap_shape: Tuple[int,int],
                orig_shape: Tuple[int,int]
            ) -> List[Tuple[int,int,int,int]]:
        """
        Chuyển list bounding box từ không gian heatmap → không gian ảnh gốc.

        Args:
            bboxes_hm: list các (x_hm, y_hm, w_hm, h_hm) tính trên heatmap.
            heatmap_shape: (h_hm, w_hm) là kích thước heatmap (chiều cao, chiều rộng).
            orig_shape:    (H, W) là kích thước ảnh gốc (height, width).

        Returns:
            List các (x_orig, y_orig, w_orig, h_orig), tọa độ trên ảnh gốc.
        """
        h_hm, w_hm = heatmap_shape
        H, W = orig_shape

        scale_x = W / w_hm
        scale_y = H / h_hm

        bboxes_orig = []
        for (x_hm, y_hm, w_hm_box, h_hm_box) in bboxes_hm:
            x_orig = int(round(x_hm * scale_x))
            y_orig = int(round(y_hm * scale_y))
            w_orig = int(round(w_hm_box * scale_x))
            h_orig = int(round(h_hm_box * scale_y))

            # Giới hạn trong ảnh gốc
            if x_orig < 0: x_orig = 0
            if y_orig < 0: y_orig = 0
            if x_orig + w_orig > W:
                w_orig = W - x_orig
            if y_orig + h_orig > H:
                h_orig = H - y_orig

            bboxes_orig.append((x_orig, y_orig, w_orig, h_orig))

        return bboxes_orig

    def get_bounding_boxes_from_heatmap(self,
                                    heatmap: np.ndarray,
                                    threshold: float = 0.5,
                                    min_area: int = 1
                                    ) -> List[Tuple[int,int,int,int]]:
        """
        Chuyển heatmap 2D thành danh sách bounding box.
        
        Args:
        heatmap (np.ndarray): mảng 2D (float hoặc int). Giá trị có thể ≥0.
        threshold (float): ngưỡng để chuyển heatmap sang nhị phân. 
                            Nếu heatmap float ∈ [0,1], thường threshold ∈ [0,1]. 
                            Nếu heatmap chứa count (>1), có thể chọn threshold ≥1.
        min_area (int): loại bỏ các contour có diện tích < min_area (pixel).
        
        Returns:
        List[Tuple[x, y, w, h]]: danh sách bounding box. 
            x,y là toạ độ góc trên-trái; w,h là chiều rộng và chiều cao.
        """
        if heatmap.ndim != 2:
            raise ValueError(f"Heatmap phải là 2D array, nhưng shape = {heatmap.shape}")
        
        hm = heatmap.astype(np.float32)
        mask = (hm >= threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area:
                bboxes.append((x, y, w, h))

        return bboxes
    
    def forward(self,
                video: torch.Tensor,
                threshold: float = 0.5,
                min_area: int = 1,
                return_option: str = None
                ) -> torch.Tensor:
        """
        video: tensor shape [T, C, H, W] (C=1 nếu gray_img=True, C=3 nếu gray_img=False)
        returns: tensor shape [T, 1, H//2, W//2] (đã qua sigmoid)
        """
        bgsub_video = self.BackgroundSubtractor(video)
        T, C_out, H, W = bgsub_video.shape
        chunks = bgsub_video.split(self.window_size, dim=0)
        
        heatmap_list = []
        
        for chunk in chunks:
            chunk = chunk

            pred = self.SegmentationModel(chunk)  
            pred = pred.detach().cpu().numpy()
            heatmap_list.append(pred)
        
        segmentation = np.concatenate(heatmap_list, axis=0)
        
        if return_option == "heatmap":
            return segmentation
        
        bboxes_per_frame = [[] for _ in range(T)]
        for i in range(T):
            mask_i = segmentation[i]
            bboxes = self.get_bounding_boxes_from_heatmap(mask_i[0], min_area=min_area)
            bboxes = self.heatmap_to_image_bboxes(
                bboxes, 
                heatmap_shape=(H//2, W//2), 
                orig_shape=(H, W)
            )
            bboxes_per_frame[i].append(bboxes)
        
        if return_option == "boxes":
            return bboxes_per_frame
        return segmentation, bboxes_per_frame
        
        