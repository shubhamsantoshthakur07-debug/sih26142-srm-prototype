"""
DualHeadSRMNet: Deep Learning Architecture for Super Resolution Mapping (SRM).
Jointly optimizes 4-band multi-spectral super-resolution (Head 1),
thematic sub-pixel land-cover classification (Head 2),
and pixel-wise uncertainty estimation (Head 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module to model inter-band spectral dependencies."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(self.avg_pool(x))
        return x * weight


class RCAB(nn.Module):
    """Residual Channel Attention Block."""
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            ChannelAttention(channels)
        )

    def forward(self, x):
        return x + self.body(x)


class DualHeadSRMNet(nn.Module):
    """
    Complete Multi-Spectral Super-Resolution Mapping Network.
    
    Inputs:
        x: (B, 4, H, W) [Bands: Red, Green, Blue, NIR]
    Outputs:
        sr_img: (B, 4, 4H, 4W) 4x Super-Resolved multi-spectral imagery
        srm_logits: (B, num_classes, 4H, 4W) Sub-pixel land cover logits
        uncertainty: (B, 1, 4H, 4W) Estimated hallucination uncertainty [0.0 - 1.0]
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 5, num_features: int = 64, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_classes = num_classes

        # 1. Shallow Feature Extraction
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 2. Deep Residual Backbone (4 RCAB blocks)
        self.block1 = RCAB(num_features)
        self.block2 = RCAB(num_features)
        self.block3 = RCAB(num_features)
        self.block4 = RCAB(num_features)
        self.trunk_conv = nn.Conv2d(num_features, num_features, 3, padding=1)

        # 3. Head 1: Multi-Spectral Physical SR 4x
        self.sr_upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, in_channels, 3, padding=1)
        )

        # 4. Head 2: Sub-Pixel Thematic Mapping (SRM)
        self.srm_upsample = nn.Sequential(
            nn.Conv2d(num_features, 32 * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, num_classes, 3, padding=1)
        )

        # 5. Head 3: Uncertainty Quantification (Zero-Hallucination Guardrail)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(num_features, 16 * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Shallow features
        feat0 = self.head_conv(x)

        # Deep backbone with residual connection
        b1 = self.block1(feat0)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        feat_deep = feat0 + self.trunk_conv(b4)

        # Head 1: SR output (with bicubic base residual)
        bicubic_base = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        sr_residual = self.sr_upsample(feat_deep)
        sr_img = torch.clamp(bicubic_base + sr_residual, 0.0, 1.0)

        # Head 2: SRM Thematic classification logits
        srm_logits = self.srm_upsample(feat_deep)

        # Head 3: Uncertainty map
        uncertainty = self.uncertainty_head(feat_deep)

        return sr_img, srm_logits, uncertainty

    def degrade_psf(self, sr_img):
        """
        Simulates the Satellite Sensor Point Spread Function (PSF)
        by applying 4x4 spatial pooling to reconstruct the medium-resolution image.
        Used for anti-hallucination cycle consistency checks.
        """
        return F.avg_pool2d(sr_img, kernel_size=self.scale_factor, stride=self.scale_factor)
