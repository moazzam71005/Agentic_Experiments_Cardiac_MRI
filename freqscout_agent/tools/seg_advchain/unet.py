"""
UNet_64 from the advchain Cardiac_Multi_view_segmentation repo.
Imports fixed to be package-relative; model_utils dependency removed (inference only).
"""
import torch.nn as nn

from .unet_parts import inconv, down, up, outconv
from .init_weight import init_weights


class UNet(nn.Module):
    """
    Standard U-Net with 64 base filters (feature_scale=1).
    4-class output: 0=BG, 1=RV, 2=MYO, 3=LV.
    """

    def __init__(
        self,
        input_channel: int = 1,
        num_classes: int = 4,
        feature_scale: int = 1,
        dropout=None,
        norm=nn.BatchNorm2d,
    ):
        super().__init__()
        if dropout is False or dropout is None:
            dropout = None

        s = feature_scale
        self.inc  = inconv(input_channel, 64 // s, norm=norm)
        self.down1 = down(64 // s, 128 // s, norm=norm)
        self.down2 = down(128 // s, 256 // s, norm=norm)
        self.down3 = down(256 // s, 512 // s, norm=norm)
        self.down4 = down(512 // s, 512 // s, norm=norm)
        self.up1  = up(512 // s, 512 // s, 256 // s, norm=norm, dropout=dropout)
        self.up2  = up(256 // s, 256 // s, 128 // s, norm=norm, dropout=dropout)
        self.up3  = up(128 // s, 128 // s,  64 // s, norm=norm, dropout=dropout)
        self.up4  = up( 64 // s,  64 // s,  64 // s, norm=norm, dropout=dropout)
        self.outc = outconv(64 // s, num_classes)
        self.n_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        return self.outc(x)
