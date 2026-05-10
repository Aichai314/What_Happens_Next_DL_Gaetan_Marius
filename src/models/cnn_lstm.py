"""
CNN + LSTM: ResNet18 per frame, then an LSTM reads the frame feature sequence.

Forward:
    Input: (B, T, C, H, W)
    Frame CNN: (B*T, C, H, W) -> (B*T, 512)
    Sequence: (B, T, 512)
    LSTM: (B, T, hidden) -> take last timestep -> (B, hidden)
    Linear: (B, num_classes)
"""

from __future__ import annotations

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torchvision import models
from utils import two_stage_trainer, inject_tsm_into_resnet, replace_resnet_stem

@two_stage_trainer
class CNNLSTM(nn.Module):
    def __init__(
        self,
        model_cfg: DictConfig,
        num_classes: int,
        num_frames: int, 
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        lstm_hidden_size = int(model_cfg.get("lstm_hidden_size", 512))
        in_channels = int(model_cfg.get("in_channels", 3))
        size = int(model_cfg.get("backbone_size", 18))
        n_div = int(model_cfg.get("fold_div", 8))
        self.resnet = False

        if model_cfg.get("efficientnet", False):
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
        elif size == 18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            self.resnet = True
        elif size == 34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
            self.resnet = True
        else:
            raise ValueError(f"Unsupported model size: {size}. Choose 18 or 34.")

        if self.resnet:
            backbone = replace_resnet_stem(backbone, in_channels=in_channels)

        if model_cfg.get("pretrained_backbone_path") is not None or model_cfg.get("pursue_from") is not None:
            # Inject TSM
            backbone = inject_tsm_into_resnet(backbone, num_frames=num_frames, n_div=n_div)
        
        if self.resnet:
            feature_dim = backbone.fc.in_features 
            backbone.fc = nn.Identity()
        else:
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        self.backbone = backbone
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        if not pretrained:
            self.dropout = nn.Dropout(p=0.3)  # ADD THIS
        else:
            self.dropout = nn.Identity()  # No dropout if pretrained, to preserve learned features
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (batch_size, T, C, H, W)
        returns logits: (batch_size, num_classes)
        """
        batch_size, num_frames, channels, height, width = video_batch.shape
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        # (B*T, 512/1280)
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)

        # (B, T, 512)
        sequence = frame_features.view(batch_size, num_frames, -1)
        
        # lstm_out: (B, T, hidden), h_n: (1, B, hidden)
        lstm_out, (h_n, _) = self.lstm(sequence)

        # Last timestep output: (B, hidden)
        last_hidden = lstm_out[:, -1, :]
        
        # Apply dropout to prevent the LSTM from overfitting
        last_hidden = self.dropout(last_hidden)

        logits = self.classifier(last_hidden)
        return logits
