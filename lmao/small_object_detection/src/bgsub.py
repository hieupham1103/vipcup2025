import torch
import torch.nn as nn
import numpy as np
from typing import Union

class BackgroundSubtractorModule(nn.Module):
    def __init__(
        self,
        window_size: int = 15,
        gray_img: bool = True,
        eps: float = 1e-5,
        normalize_output: bool = True
    ):
        """
        Args:
            window_size: Số frame dùng để tính background model
            gray_img: Nếu True -> convert RGB->grayscale (1 kênh); nếu False -> giữ 3 kênh
            eps: Giá trị nhỏ để tránh chia cho 0
            normalize_output: Nếu True, sẽ scale mỗi frame output về [0,1]
        """
        super(BackgroundSubtractorModule, self).__init__()
        self.window_size = window_size
        self.eps = eps
        self.normalize_output = normalize_output
        self.gray_img = gray_img

    def _convert_to_tensor(self, video: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Chuyển input (numpy hoặc torch) thành torch.Tensor shape [T, C, H, W]:
          - Nếu gray_img=True: output luôn C=1 (grayscale)
          - Nếu gray_img=False: output C=3 nếu input có 3 kênh, hoặc C=1 nếu input 1 kênh.
        Hỗ trợ numpy: [T,H,W], [T,H,W,3], [T,1,H,W], [T,3,H,W]
                 tensor: [T,H,W], [T,H,W,3], [T,1,H,W], [T,3,H,W]
        """
        if isinstance(video, np.ndarray):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video
        video_tensor = video_tensor.float()

        # check if ([T,H,W,3]) then permute to CHW
        if video_tensor.dim() == 4 and video_tensor.shape[-1] == 3:
            video_tensor = video_tensor.permute(0, 3, 1, 2)

        # if [T,H,W] (grayscale) then make it [T,1,H,W]
        if video_tensor.dim() == 3:
            video_tensor = video_tensor.unsqueeze(1) 

        if video_tensor.dim() != 4 or video_tensor.shape[1] not in (1, 3):
            raise ValueError(f"Unsupported video shape {video_tensor.shape}. "
                             "Cần [T,H,W], [T,H,W,3], [T,1,H,W], hoặc [T,3,H,W].")

        # check if video_tensor is normalized to [0, 1]
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0

        if self.gray_img and video_tensor.shape[1] == 3:
            w = video_tensor.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            video_tensor = (video_tensor * w).sum(dim=1, keepdim=True)  # [T,1,H,W]

        return video_tensor  # [T, C_out, H, W], C_out = 1 or 3

    @torch.no_grad()
    def forward(self, video: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            video: [T, H, W, 3] / [T, H, W] / [T, C, H, W]
        Returns:
            background-subtracted video, shape [T, C_out, H, W],
            với C_out = 1 nếu gray_img=True, else C_out = 3
        """
        video_tensor = self._convert_to_tensor(video)  # [T, C, H, W]
        T, C, H, W = video_tensor.shape

        output = torch.zeros_like(video_tensor)

        buffer = torch.zeros((self.window_size, C, H, W),
                             dtype=video_tensor.dtype,
                             device=video_tensor.device)

        for k in range(T):
            idx = k % self.window_size
            buffer[idx] = video_tensor[k]

            if idx == self.window_size - 1:
                bg_model = buffer.mean(dim=0)
                bg_sigma = buffer.std(dim=0) + self.eps

                for i in range(self.window_size):
                    diff = (buffer[i] - bg_model).abs()
                    bgsub = diff / bg_sigma 

                    if self.normalize_output:
                        for c in range(C):
                            plane = bgsub[c]
                            minv = plane.min()
                            maxv = plane.max()
                            if (maxv - minv) > 1e-6:
                                bgsub[c] = (plane - minv) / (maxv - minv)

                    t_out = k - self.window_size + 1 + i
                    output[t_out] = bgsub

        return output