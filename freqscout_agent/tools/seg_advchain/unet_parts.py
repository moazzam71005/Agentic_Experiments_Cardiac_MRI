"""
U-Net sub-parts from the advchain Cardiac_Multi_view_segmentation repo.
Imports fixed to be package-relative.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .init_weight import init_weights


class double_conv(nn.Module):
    """(Conv → BN → ReLU) × 2"""

    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d, if_SN=False, activation=nn.ReLU):
        super().__init__()
        if not if_SN:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                norm(out_ch),
                activation(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                norm(out_ch),
                activation(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
                norm(out_ch),
                activation(inplace=True),
                spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
                norm(out_ch),
                activation(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d, if_SN=False, activation=nn.ReLU):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch, norm=norm, if_SN=if_SN, activation=activation)

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d, if_SN=False, activation=nn.ReLU):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, norm=norm, if_SN=if_SN, activation=activation),
        )

    def forward(self, x):
        return self.mpconv(x)


class up(nn.Module):
    def __init__(
        self, in_ch_1, in_ch_2, out_ch, type="bilinear",
        dropout=None, norm=nn.BatchNorm2d, if_SN=False, activation=nn.ReLU,
    ):
        super().__init__()
        self.type = type
        if type == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        elif type == "deconv":
            self.up = nn.ConvTranspose2d(
                (in_ch_1 + in_ch_2) // 2, (in_ch_1 + in_ch_2) // 2, 2, stride=2
            )
        elif type == "nearest":
            self.up = nn.Upsample(scale_factor=2)
        else:
            self.up = None

        self.conv = double_conv(in_ch_1 + in_ch_2, out_ch, norm=norm, if_SN=if_SN, activation=activation)
        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        if self.dropout and self.training:
            x = self.drop(x)
        return self.conv(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)
