import torch
import torch.nn as nn
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
        # Your dataloader provides: (Batch, Time, Channels, Height, Width)
        # R(2+1)D expects: (Batch, Channels, Time, Height, Width)
        x = x.permute(0, 2, 1, 3, 4)
        
        logits = self.backbone(x)
        return logits