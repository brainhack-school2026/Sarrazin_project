"""
unet_model.py
=============
2D U-Net for MRI reconstruction.

Input:  (batch, 2, H, W) — real & imaginary channels of corrupted k-space
Output: (batch, 2, H, W) — real & imaginary channels of corrected k-space
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two conv layers with BatchNorm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

    


class UNet2D(nn.Module):
    """
    2D U-Net for k-space to image reconstruction.

    Architecture: 3 pooling levels, base_channels=32 → ~7M parameters.
    Skip connections preserve spatial detail lost in the encoder.

    Parameters
    ----------
    in_channels : int
        2 — real and imaginary parts of corrupted k-space
    out_channels : int
        1 — real and imaginary parts of corrected k-space
    base_channels : int
        Number of filters in the first encoder block.
        Doubles at each level: 32 → 64 → 128 → 256 (bottleneck)
    """
    def __init__(self, in_channels=2, out_channels=2, base_channels=32):
        super().__init__()
        b = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b,     b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(b * 4, b * 8)

        # Decoder
        self.up3   = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3  = ConvBlock(b * 8, b * 4)   # b*4 + b*4 skip

        self.up2   = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2  = ConvBlock(b * 4, b * 2)   # b*2 + b*2 skip

        self.up1   = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1  = ConvBlock(b * 2, b)        # b + b skip

        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    # def forward(self, x):
    #     # Encoder
    #     e1 = self.enc1(x)
    #     e2 = self.enc2(self.pool(e1))
    #     e3 = self.enc3(self.pool(e2))

    #     # Bottleneck
    #     b  = self.bottleneck(self.pool(e3))

    #     # Decoder with skip connections
    #     d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
    #     d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
    #     d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

    #     return self.out_conv(d1)

    def forward(self, x):
        # ── Pad to multiple of 8 (2^3 pooling levels) ────────────────
        B, C, H, W = x.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b  = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out_conv(d1)

        # ── Crop back to original size ────────────────────────────────
        return out[:, :, :H, :W]