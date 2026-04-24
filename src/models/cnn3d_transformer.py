"""
Residual 3D-CNN + Transformer encoder for video classification (Track A).

Architecture:
  (B, T, C, H, W)
  → Stem conv (1×7×7)  +  MaxPool  →  (B, 64, T, 56, 56)
  → Layer1: 2× ResBlock3D(64→64)   →  (B,  64, T, 56, 56)
  → Layer2: 2× ResBlock3D(64→128, spatial stride 2)  →  (B, 128, T, 28, 28)
  → Layer3: 2× ResBlock3D(128→256, spatial stride 2) →  (B, 256, T, 14, 14)
  → AdaptiveAvgPool3d((None,1,1))  →  (B, T, 256) feature sequence
  → CLS token + positional embedding
  → TransformerEncoder (6 layers, pre-norm)
  → CLS → classification head → (B, num_classes)

Key improvements over naive conv blocks:
  - Residual (skip) connections  → stable gradient flow, deeper = better
  - 1×7×7 stem (not 3×3×3)      → captures wide spatial context upfront
  - Spatial-only downsampling    → temporal resolution T fully preserved
  - Hierarchical channel growth  → 64 → 128 → 256
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    """
    Basic 3D residual block.
    stride applies to spatial dims only (temporal stride always 1).
    Downsamples with 1×1×1 conv on the skip path when needed.
    """

    def __init__(self, in_c: int, out_c: int, spatial_stride: int = 1) -> None:
        super().__init__()
        stride = (1, spatial_stride, spatial_stride)

        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_c)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if spatial_stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CNN3DTransformer(nn.Module):
    """
    Args:
        num_classes : number of output classes.
        d_model     : transformer hidden dimension.
        nhead       : attention heads (must divide d_model).
        num_layers  : transformer encoder depth.
        dropout     : dropout in transformer and head.
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Stem: large spatial kernel, no temporal stride ────────────────────
        # (B, 3, T, 224, 224) → (B, 64, T, 56, 56)
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                      padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # ── Residual layers (spatial ↓, temporal preserved) ───────────────────
        self.layer1 = nn.Sequential(        # → (B,  64, T, 56, 56)
            ResBlock3D(64,  64),
            ResBlock3D(64,  64),
        )
        self.layer2 = nn.Sequential(        # → (B, 128, T, 28, 28)
            ResBlock3D(64,  128, spatial_stride=2),
            ResBlock3D(128, 128),
        )
        self.layer3 = nn.Sequential(        # → (B, 256, T, 14, 14)
            ResBlock3D(128, 256, spatial_stride=2),
            ResBlock3D(256, 256),
        )

        # ── Spatial pool → temporal token sequence ────────────────────────────
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # ── Project to d_model ────────────────────────────────────────────────
        self.input_proj = nn.Linear(256, d_model)

        # ── CLS token + learnable positional embeddings ───────────────────────
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 17, d_model))  # max 16 frames + CLS
        nn.init.trunc_normal_(self.cls_token,     std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # 3D-CNN expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = self.stem(x)       # (B,  64, T, 56, 56)
        x = self.layer1(x)     # (B,  64, T, 56, 56)
        x = self.layer2(x)     # (B, 128, T, 28, 28)
        x = self.layer3(x)     # (B, 256, T, 14, 14)

        x = self.spatial_pool(x)                          # (B, 256, T, 1, 1)
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)   # (B, T, 256)
        x = self.input_proj(x)                            # (B, T, d_model)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                    # (B, T+1, d_model)
        x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer(x)                           # (B, T+1, d_model)

        return self.head(x[:, 0, :])                      # (B, num_classes)
