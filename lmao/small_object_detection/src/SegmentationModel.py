
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationModel(nn.Module):
    """
    PyTorch port của Keras SegmentationModel (U-Net like) từ bài báo Project Bee4Exp.
    Input: tensor shape (B, input_channels, H, W)
    Output: tensor shape (B, out_channels, H//2, W//2), đã qua sigmoid
    """

    def __init__(self, input_channels: int = 3, out_channels: int = 1):
        super(SegmentationModel, self).__init__()

        # ---- Encoder ----
        # conv1: (C_in -> 64) → BN → ReLU → MaxPool
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/2, W/2

        # conv2: (64 -> 128) → BN → ReLU → MaxPool
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/4, W/4

        # conv3: (128 -> 256) → BN → ReLU → MaxPool
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/8, W/8

        # fc-conv2: (256 -> 512) 1x1 conv → BN → ReLU
        self.conv_fc = nn.Conv2d(256, 512, kernel_size=1, padding=0, bias=False)
        self.bn_fc   = nn.BatchNorm2d(512)

        # ---- Decoder ----
        # Upsample + conv3u: (512 -> 256) conv → Add skip with conv3 output → BN → ReLU
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')  # H/4, W/4
        self.conv3u    = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.bn3u      = nn.BatchNorm2d(256)

        # Upsample + conv2u: (256 -> 128) conv → Add skip with conv2 output → BN → ReLU
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')  # H/2, W/2
        self.conv2u    = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.bn2u      = nn.BatchNorm2d(128)

        # Final detector: (128 -> out_channels) 1x1 conv → Sigmoid
        self.detector = nn.Conv2d(128, out_channels, kernel_size=1, padding=0, bias=True)
        self.act_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor shape (B, input_channels, H, W)
        returns: tensor shape (B, out_channels, H//2, W//2)
        """

        # ---- Encoder ----
        # conv1 block
        x1 = self.conv1(x)           # (B, 64, H,   W)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        p1 = self.pool1(x1)          # (B, 64, H/2, W/2)

        # conv2 block
        x2 = self.conv2(p1)          # (B,128, H/2, W/2)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        p2 = self.pool2(x2)          # (B,128, H/4, W/4)

        # conv3 block
        x3 = self.conv3(p2)          # (B,256, H/4, W/4)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        p3 = self.pool3(x3)          # (B,256, H/8, W/8)

        # fc-conv2
        xf = self.conv_fc(p3)        # (B,512, H/8, W/8)
        xf = self.bn_fc(xf)
        xf = F.relu(xf)

        # ---- Decoder ----
        # first upsample + conv3u + add skip from x3
        u1 = self.upsample1(xf)      # (B,512, H/4, W/4)
        c3u = self.conv3u(u1)        # (B,256, H/4, W/4)
        # skip connection with x3 (shape (B,256,H/4,W/4))
        s3 = c3u + x3                # (B,256, H/4, W/4)
        s3 = self.bn3u(s3)
        s3 = F.relu(s3)

        # second upsample + conv2u + add skip from x2
        u2 = self.upsample2(s3)      # (B,256, H/2, W/2)
        c2u = self.conv2u(u2)        # (B,128, H/2, W/2)
        # skip connection with x2 (shape (B,128, H/2, W/2))
        s2 = c2u + x2                # (B,128, H/2, W/2)
        s2 = self.bn2u(s2)
        s2 = F.relu(s2)

        # final 1x1 conv + sigmoid (output at H/2, W/2)
        out = self.detector(s2)      # (B,out_channels, H/2, W/2)
        out = self.act_sigmoid(out)

        return out