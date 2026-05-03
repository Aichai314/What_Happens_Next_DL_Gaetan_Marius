import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18

class R2Plus1DBaseline(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        # Ensure we are strictly from scratch
        self.backbone = r2plus1d_18(weights=None) if not pretrained else r2plus1d_18(weights="DEFAULT")
        
        # Replace the default 400-class Kinetics head with your classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape from dataloader: (Batch, Time, Channels, Height, Width)
        B, T, C, H, W = x.shape
        
        # =========================================================
        # THE 112x112 RESOLUTION PURGE (ANTI-OVERFITTER)
        # =========================================================
        # Fold Time into Batch to run 2D Spatial Interpolation
        x = x.view(B * T, C, H, W)
        
        # Force the resolution down to exactly 112x112, discarding high-freq noise
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        
        # Unfold back to Video sequence
        x = x.view(B, T, C, 112, 112)
        
        # =========================================================
        # R(2+1)D 3D CONVOLUTIONAL PASS
        # =========================================================
        # R(2+1)D expects: (Batch, Channels, Time, Height, Width)
        x = x.permute(0, 2, 1, 3, 4)
        
        logits = self.backbone(x)
        return logits