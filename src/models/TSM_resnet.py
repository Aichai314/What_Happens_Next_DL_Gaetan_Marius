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
from omegaconf import DictConfig
from utils import inject_tsm_into_resnet, replace_resnet_stem, inject_tdm_into_resnet

class TSMBaseline(nn.Module):
    def __init__(
        self,
        model_cfg: DictConfig,
        num_classes: int, 
        num_frames: int, 
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        size = model_cfg.get("model_size", 18)
        in_channels = model_cfg.get("in_channels", 3)
        n_div = model_cfg.get("fold_div", 8)
        dropout = model_cfg.get("dropout", 0.0)
        
        if size == 18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        elif size == 34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported model size: {size}. Choose 18 or 34.")

        backbone = replace_resnet_stem(backbone, in_channels=in_channels, keep_original=not model_cfg.get("change_stem", False))

        # Inject the Temporal Shift Module into the backbone
        backbone = inject_tsm_into_resnet(backbone, num_frames=num_frames, n_div=n_div)
        
        if model_cfg.get("inject_tdm", False):
            backbone = inject_tdm_into_resnet(backbone, num_frames=num_frames)

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
