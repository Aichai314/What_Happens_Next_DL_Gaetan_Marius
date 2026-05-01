"""
CNN + GRU: ResNet18 per frame, then a GRU reads the frame feature sequence.
Built specifically for short sequences (4 frames) to minimize parameter count
and combat overfitting on the Closed Track.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CNNGRU(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        gru_hidden_size: int = 128,  # Defaulted to a highly compressed size
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # The GRU (Notice we use nn.GRU instead of nn.LSTM)
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # 2. The Post-GRU Regularization Barrier
        if not pretrained:
            self.classifier_dropout = nn.Dropout(p=0.3)
        else:
            self.classifier_dropout = nn.Identity()
            
        self.classifier = nn.Linear(gru_hidden_size, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (batch_size, T, C, H, W)
        returns logits: (batch_size, num_classes)
        """
        batch_size, num_frames, channels, height, width = video_batch.shape
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        # Extract features: (B*T, 512)
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)

        # Reshape to Sequence: (B, T, 512)
        sequence = frame_features.view(batch_size, num_frames, -1)

        # GRU Forward Pass
        # PyTorch GRU returns (output, h_n). It does NOT return a cell state (c_n) like LSTM.
        gru_out, h_n = self.gru(sequence)
        
        # THE FIX: Temporal Mean Pooling
        # Instead of taking just the last frame, we average the hidden states
        # across all 4 frames. This forces the network to learn the whole sequence.
        pooled_hidden = gru_out.mean(dim=1)  # Shape: (B, hidden_size)

        # Take the output of the final timestep: (B, hidden)
        #last_hidden = gru_out[:, -1, :]
        
        # Apply final dropout before the guess
        pooled_hidden = self.classifier_dropout(pooled_hidden)

        logits = self.classifier(pooled_hidden)
        return logits