"""
X3D Extra Small (X3D-XS) Baseline
Natively optimized for 4-frame sequences and 160x160 resolution.
Contains a built-in GPU downsampler to seamlessly integrate with 224x224 pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class X3DXSBaseline(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        
        # 1. Load the core X3D-XS model from the official PyTorchVideo Hub
        self.backbone = torch.hub.load(
            'facebookresearch/pytorchvideo', 
            'x3d_xs', 
            pretrained=pretrained
        )
        
        # 2. Modify the classification head for our specific number of classes (33)
        # PyTorchVideo's X3D stores its final Linear layer in blocks[5].proj
        in_features = self.backbone.blocks[5].proj.in_features
        self.backbone.blocks[5].proj = nn.Linear(in_features, num_classes)
        
        # Ensure it outputs raw logits (removes any default Softmax if present)
        if hasattr(self.backbone.blocks[5], 'activation'):
            self.backbone.blocks[5].activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape from your current dataloader: (Batch, Time, Channels, Height, Width)
        B, T, C, H, W = x.shape
        
        # =========================================================
        # THE 160x160 RESOLUTION PURGE (ANTI-OVERFITTER)
        # =========================================================
        # Fold Time into Batch to run 2D Spatial Interpolation
        x = x.view(B * T, C, H, W)
        
        # Force the resolution down to exactly 160x160, discarding high-freq noise
        x = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        
        # Unfold back to Video sequence
        x = x.view(B, T, C, 160, 160)
        
        # =========================================================
        # X3D 3D CONVOLUTIONAL PASS
        # =========================================================
        # PyTorchVideo natively expects (Batch, Channels, Time, Height, Width)
        x = x.permute(0, 2, 1, 3, 4)
        
        logits = self.backbone(x)
        return logits