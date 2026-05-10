"""
MobileNetV2 + TSM (Pure Motion Expert)

Input: 3-channel Frame Differences ONLY (No RGB).
Goal: Maximize parameter efficiency to learn pure physics from scratch.
"""

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models.efficientnet import MBConv

from utils import two_stage_trainer

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

class TemporalShift(nn.Module):
    """Shift 1/fold_div channels forward and 1/fold_div channels backward in time."""
    def __init__(self, n_segment: int, fold_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        bt, c, h, w = x.shape
        b = bt // self.n_segment
        x = x.view(b, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = x.clone()
        out[:, 1:, :fold] = x[:, :-1, :fold]                      # past -> current
        out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]  # future -> current
        out[:, 0, :fold] = 0
        out[:, -1, fold : 2 * fold] = 0

        return out.view(bt, c, h, w)

@two_stage_trainer
class EfficientNetAttention(nn.Module):
    def __init__(
        self,
        model_cfg: DictConfig,
        num_classes: int = 33, 
        num_frames: int = 4,
        pretrained: bool = False,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        
        fold_div = int(model_cfg.get("fold_div", 8))
        in_channels = int(model_cfg.get("in_channels", 3))

        # 1. Load EfficientNet WITHOUT pretraining (Track A Rules)
        weights = None if not pretrained else tv_models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = tv_models.efficientnet_b0(weights=weights)

        # 2. STEM SURGERY
        if in_channels != 3:
            # EfficientNet's stem is a Conv2dNormActivation block.
            # We need to replace the very first Conv2d layer.
            original_conv = self.backbone.features[0][0]
            # Replace it with a 6-channel version, keeping stride and padding identical
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, 
                original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False # BatchNorm follows, so bias must be False
            )

        # 3. TSM INJECTION (The "Physics Engine")
        for idx, module in enumerate(self.backbone.features):
            # EfficientNet groups MBConv blocks inside Sequentials
            if isinstance(module, nn.Sequential):
                for sub_idx, sub_module in enumerate(module):
                    if isinstance(sub_module, MBConv):
                        module[sub_idx] = nn.Sequential(
                            TemporalShift(num_frames, fold_div),
                            sub_module
                        )

        # =========================================================
        # INJECTING THE ATTENTION HEAD
        # =========================================================
        self.temporal_attn = TemporalSelfAttention(
            embed_dim=self.backbone.classifier[1].in_features, 
            num_frames=num_frames, 
            num_heads=8,       # 8 attention heads looking at the 4 frames
            dropout=dropout,
            attention_dropout=model_cfg.get("attention_dropout", 0.15)
        )
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x MUST be frame differences! Shape: (B, T, 3, H, W)
        b, t, c, h, w = x.shape

        # Fold time into batch for the 2D CNN
        x = x.view(b * t, c, h, w)

        frame_features = self.backbone.features(x)  # Extracts the (B*T, 1280, 4, 4) tensor
        
        # Global Spatial Average Pooling
        # (B*T, 1280)
        frame_features = nn.functional.adaptive_avg_pool2d(frame_features, (1, 1)).flatten(1) 

        # (B, T, 1280)
        sequence_features = frame_features.view(b, t, -1)

        # =========================================================
        # APPLYING THE ATTENTION HEAD
        # =========================================================
        # Instead of sequence_features.mean(dim=1), we pass it through the attention block!
        pooled_features = self.temporal_attn(sequence_features)

        pooled_features = self.dropout(pooled_features)
        logits = self.classifier(pooled_features)
        
        return logits