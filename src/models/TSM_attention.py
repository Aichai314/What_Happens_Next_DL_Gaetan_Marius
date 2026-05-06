"""
TSM (Temporal Shift Module) Baseline Architecture with Temporal Self-Attention.
Suitable for Early Action Recognition (Track A or Track B).
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
        bt, c, h, w = x.size()
        batch_size = bt // num_frames
        
        x = x.view(batch_size, num_frames, c, h, w)
        out = torch.zeros_like(x)
        fold = c // fold_div

        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out.view(bt, c, h, w)

def inject_tsm_into_resnet(model: nn.Module, num_frames: int, n_div: int = 8) -> nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, models.resnet.BasicBlock):
            module.conv1 = TemporalShift(module.conv1, num_frames=num_frames, n_div=n_div)
    return model

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

class TSMAttention(nn.Module):
    def __init__(
        self, 
        num_classes: int, 
        num_frames: int, 
        pretrained: bool = False, 
        dropout: float = 0, 
        n_div: int = 8,
        in_channels: int = 3,
        attention_dropout: float = 0.3
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # 6-Channel Surgery
        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                stride=old_conv.stride, padding=old_conv.padding, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    backbone.conv1.weight[:, :3] = old_conv.weight
                    backbone.conv1.weight[:, 3:] = 0.0

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