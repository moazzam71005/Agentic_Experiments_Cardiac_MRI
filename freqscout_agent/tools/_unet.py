"""
Lightweight U-Net for cardiac SAX segmentation (4-class: bg / RV / myo / LV pool).
Used as the model when a checkpoint is available; replaced by intensity fallback otherwise.

Architecture: 3-level encoder-decoder with skip connections, ~400 k parameters.
Input: 1×H×W float32.  Output: 4×H×W logits.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightweightUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base: int = 32):
        super().__init__()
        b = base
        self.enc1 = _ConvBlock(in_channels, b)
        self.enc2 = _ConvBlock(b, b * 2)
        self.enc3 = _ConvBlock(b * 2, b * 4)
        self.bottleneck = _ConvBlock(b * 4, b * 8)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = _ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = _ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = _ConvBlock(b * 2, b)

        self.head = nn.Conv2d(b, num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
