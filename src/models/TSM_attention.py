"""
TSM (Temporal Shift Module) Baseline Architecture with Temporal Self-Attention.
Suitable for Early Action Recognition (Track A or Track B).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from omegaconf import DictConfig
from utils import two_stage_trainer, inject_tsm_into_resnet, replace_resnet_stem

# =========================================================
# THE NEW TEMPORAL ATTENTION HEAD
# =========================================================
class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_frames: int, num_heads: int = 8, dropout: float = 0.5, attention_dropout: float = 0.3):
        super().__init__()
        # 1. Positional Embedding: Crucial so the attention knows frame order
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        # 2. Multi-Head Attention: batch_first=True expects (B, T, C)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout, batch_first=True)
        
        # 3. Standard Transformer block normalization and MLP
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C)
        
        # Add temporal position information
        x = x + self.pos_embed
        
        # Self-Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed Forward Network
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        # Instead of returning all T frames, we now pool the *attended* features
        return x.mean(dim=1)

@two_stage_trainer
class TSMAttention(nn.Module):
    def __init__(
        self,
        model_cfg: DictConfig,
        num_classes: int, 
        num_frames: int, 
        pretrained: bool = False
    ) -> None:
        super().__init__()
        dropout = float(model_cfg.get("dropout", 0))
        n_div = int(model_cfg.get("fold_div", 8))
        in_channels = int(model_cfg.get("in_channels", 3))
        attention_dropout = float(model_cfg.get("attention_dropout", 0.3))
        size = int(model_cfg.get("backbone_size", 18))

        if size == 18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        elif size == 34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported model size: {size}. Choose 18 or 34.")

        # Replace the ResNet stem
        backbone = replace_resnet_stem(backbone, in_channels=in_channels,
                                       keep_original=not model_cfg.get("change_stem", False))

        # Inject TSM
        backbone = inject_tsm_into_resnet(backbone, num_frames=num_frames, n_div=n_div)

        feature_dim = backbone.fc.in_features 
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        # =========================================================
        # INJECTING THE ATTENTION HEAD
        # =========================================================
        self.temporal_attn = TemporalSelfAttention(
            embed_dim=feature_dim, 
            num_frames=num_frames, 
            num_heads=8,       # 8 attention heads looking at the 4 frames
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = video_batch.shape

        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)

        # (B, T, 512)
        sequence_features = frame_features.view(batch_size, num_frames, -1)

        # =========================================================
        # APPLYING THE ATTENTION HEAD
        # =========================================================
        # Instead of sequence_features.mean(dim=1), we pass it through the attention block!
        pooled_features = self.temporal_attn(sequence_features)

        pooled_features = self.dropout(pooled_features)
        logits = self.classifier(pooled_features)
        
        return logits