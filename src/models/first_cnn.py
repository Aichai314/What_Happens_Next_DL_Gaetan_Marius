"""
CNN from scratch for video classification.

Architecture (frame-wise, then temporal pooling):

    Input  : (B, T, C, H, W)   — B clips, T frames each, 3-channel 224×224 images
    Reshape: (B*T, C, H, W)     — treat each frame as an independent image

    5 convolutional blocks, each: Conv2d → BatchNorm → ReLU → MaxPool(2×2)
        Block 1:   3 →  32 channels,  224×224 → 112×112
        Block 2:  32 →  64 channels,  112×112 →  56×56
        Block 3:  64 → 128 channels,   56×56  →  28×28
        Block 4: 128 → 256 channels,   28×28  →  14×14
        Block 5: 256 → 512 channels, AdaptiveAvgPool → 1×1

    Flatten   : (B*T, 512)
    Reshape   : (B, T, 512)
    Mean(dim=1): (B, 512)         — average frame features over time
    Dropout
    Linear    : (B, num_classes)

Why this design?
- BatchNorm after each Conv stabilises training and allows higher learning rates.
- MaxPool halves spatial resolution at each block: cheap and effective.
- AdaptiveAvgPool at the end makes the model resolution-agnostic.
- Temporal average pooling is simple and works well for action recognition.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv2d(3×3, pad=1) → BatchNorm → ReLU → MaxPool(2×2)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class FirstCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()

        # 5 successive conv blocks: each halves H and W, doubles channels
        self.features = nn.Sequential(
            _conv_block(3, 32),    # 224 → 112
            _conv_block(32, 64),   # 112 → 56
            _conv_block(64, 128),  #  56 → 28
            _conv_block(128, 256), #  28 → 14
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 14×14 → 1×1, any input size works
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (batch_size, T, C, H, W)
        returns logits: (batch_size, num_classes)
        """
        batch_size, num_frames, channels, height, width = video_batch.shape

        # Run the CNN on every frame independently: (B*T, C, H, W)
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)
        frame_features = self.features(frames)          # (B*T, 512, 1, 1)
        frame_features = torch.flatten(frame_features, start_dim=1)  # (B*T, 512)

        # Restore temporal axis: (B, T, 512)
        sequence = frame_features.view(batch_size, num_frames, -1)

        # Average all frame features → one vector per clip: (B, 512)
        pooled = sequence.mean(dim=1)

        logits = self.classifier(self.dropout(pooled))  # (B, num_classes)
        return logits
