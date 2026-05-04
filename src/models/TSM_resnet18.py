"""
TSM (Temporal Shift Module) Baseline Architecture.
Suitable for Track A (from scratch) or Track B (pretrained).

Forward (conceptually):
    Input:  (batch, time, C, H, W)
    Reshape: (batch * time, C, H, W)  # Merged for standard 2D CNN processing
    Backbone: ResNet18 with TSM injected into each BasicBlock's first convolution.
              -> TSM temporarily reshapes to (batch, time, C, H, W), shifts 1/4th 
                 of the channels (1/8 left, 1/8 right) along the time axis, and 
                 flattens back to (batch * time, C, H, W).
              -> This allows 2D spatial convolutions to inherently learn temporal motion.
              -> Outputs: (batch * time, 512, 1, 1)
    Flatten: (batch * time, 512)
    Reshape: (batch, time, 512)
    Temporal Pooling: Mean over the time dimension -> (batch, 512)
    Classifier: Linear layer -> (batch, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

class TemporalShift(nn.Module):
    def __init__(self, net: nn.Module, num_frames: int, n_div: int = 8) -> None:
        super().__init__()
        self.net = net
        self.num_frames = num_frames
        self.fold_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shift(x, self.num_frames, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x: torch.Tensor, num_frames: int, fold_div: int = 8) -> torch.Tensor:
        # x shape: (B*T, C, H, W)
        bt, c, h, w = x.size()
        batch_size = bt // num_frames
        
        # Reshape to explicitly expose the temporal dimension
        x = x.view(batch_size, num_frames, c, h, w)

        out = torch.zeros_like(x)
        fold = c // fold_div

        # Shift left (past frames)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        
        # Shift right (future frames)
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        
        # Keep the rest of the channels intact
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        # Flatten back to (B*T, C, H, W) for the 2D CNN
        return out.view(bt, c, h, w)

def inject_tsm_into_resnet(model: nn.Module, num_frames: int, n_div: int = 8) -> nn.Module:
    """
    Iterates through a torchvision ResNet and wraps the first convolution 
    of each BasicBlock with the TemporalShift operation.
    """
    for name, module in model.named_modules():
        if isinstance(module, models.resnet.BasicBlock):
            # Wrap conv1 in the BasicBlock
            module.conv1 = TemporalShift(module.conv1, num_frames=num_frames, n_div=n_div)
    return model

class TSMBaseline(nn.Module):
    def __init__(self, num_classes: int, num_frames: int, pretrained: bool = False, dropout: float = 0, n_div: int = 8) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Inject the Temporal Shift Module into the backbone
        backbone = inject_tsm_into_resnet(backbone, num_frames=num_frames, n_div=n_div)

        # Replace the original 1000-way ImageNet head with identity
        feature_dim = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()

        self.backbone = backbone
        
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()  # No dropout if dropout=0.0
        
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (batch_size, T, C, H, W)
        returns logits: (batch_size, num_classes)
        """
        batch_size, num_frames, channels, height, width = video_batch.shape

        # Merge batch and time so the CNN runs frame-wise: (B*T, C, H, W)
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        # (B*T, 512, 1, 1) -> (B*T, 512)
        # Note: The injected TSM layers inside the backbone will automatically 
        # unfold, shift, and refold this tensor based on the init `num_frames`.
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)

        # Restore temporal structure: (B, T, 512)
        sequence_features = frame_features.view(batch_size, num_frames, -1)

        # Simple temporal pooling: average over frames -> (B, 512)
        pooled_features = sequence_features.mean(dim=1)

        # Apply dropout before classification
        pooled_features = self.dropout(pooled_features)
        # Class scores: (B, num_classes)
        logits = self.classifier(pooled_features)
        return logits
