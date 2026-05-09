"""
MobileNetV2 + TSM (Pure Motion Expert)

Input: 3-channel Frame Differences ONLY (No RGB).
Goal: Maximize parameter efficiency to learn pure physics from scratch.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models.mobilenetv2 import InvertedResidual

class TemporalShift(nn.Module):
    """Shift 1/fold_div channels forward and 1/fold_div channels backward in time."""
    def __init__(self, n_segment: int, fold_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        bt, c, h, w = x.shape
        b = bt // self.n_segment
        x = x.view(b, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = x.clone()
        out[:, 1:, :fold] = x[:, :-1, :fold]                      # past -> current
        out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]  # future -> current
        out[:, 0, :fold] = 0
        out[:, -1, fold : 2 * fold] = 0

        return out.view(bt, c, h, w)


class MobileNetV2_TSM(nn.Module):
    def __init__(
        self,
        num_classes: int = 33, 
        num_frames: int = 4,
        pretrained: bool = False,
        in_channels: int = 3,
        fold_div: int = 8,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        # 1. Load MobileNetV2 WITHOUT pretraining (Track A Rules)
        weights = None if not pretrained else tv_models.MobileNet_V2_Weights.DEFAULT
        self.backbone = tv_models.mobilenet_v2(weights=weights)

        # 2. STEM SURGERY
        if in_channels != 3:
            # MobileNetV2's stem is a Conv2dNormActivation block.
            # We need to replace the very first Conv2d layer.
            original_conv = self.backbone.features[0][0]
            # Replace it with a 6-channel version, keeping stride and padding identical
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 
                original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False # BatchNorm follows, so bias must be False
            )

        # 3. TSM INJECTION (The "Physics Engine")
        # We iterate through the MobileNet features and inject TSM right 
        # before every InvertedResidual block.
        for idx, module in enumerate(self.backbone.features):
            if isinstance(module, InvertedResidual):
                self.backbone.features[idx] = nn.Sequential(
                    TemporalShift(num_frames, fold_div),
                    module
                )

        # 4. SIMPLE CLASSIFIER HEAD
        # MobileNetV2 ends with 1280 channels
        self.head = nn.Sequential(
            nn.Dropout(p=dropout), # Slight dropout to prevent overfitting
            nn.Linear(self.backbone.last_channel, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x MUST be frame differences! Shape: (B, T, 3, H, W)
        b, t, c, h, w = x.shape

        # Fold time into batch for the 2D CNN
        x = x.view(b * t, c, h, w)

        # Extract features
        feats = self.backbone.features(x)  # (B*T, 1280, H', W')

        # Global Average Pooling (Spatial)
        feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)  # (B*T, 1280)

        # Reshape back to temporal sequence
        feats = feats.view(b, t, -1)  # (B, T, 1280)

        # Temporal Average Pooling (Consensus)
        pooled = feats.mean(dim=1)  # (B, 1280)

        return self.head(pooled)