"""
Lightweight 3D-CNN + Transformer encoder for video classification (Track A).

Architecture:
  (B, T, C, H, W)
  → custom 3D-CNN  (spatial 224→14, temporal preserved)
  → spatial pool   → feature sequence (B, T, 256)
  → Transformer encoder (4-6 layers, pre-norm)
  → CLS token → classification head → (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv3d_block(in_c: int, out_c: int) -> nn.Sequential:
    """3×3×3 conv with spatial stride 2 (temporal preserved), BN, ReLU."""
    return nn.Sequential(
        nn.Conv3d(
            in_c, out_c,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
            bias=False,
        ),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
    )


class CNN3DTransformer(nn.Module):
    """
    Args:
        num_classes : number of output classes.
        d_model     : transformer hidden dimension.
        nhead       : number of attention heads (must divide d_model).
        num_layers  : number of TransformerEncoder layers (4-6 recommended).
        dropout     : dropout in transformer and classifier.
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Lightweight 3D-CNN ────────────────────────────────────────────────
        # Input : (B, 3, T, 224, 224)
        # Output: (B, 256, T, 14, 14)  — spatial 224→14, temporal preserved
        self.backbone = nn.Sequential(
            _conv3d_block(3,   32),   # → (B,  32, T, 112, 112)
            _conv3d_block(32,  64),   # → (B,  64, T,  56,  56)
            _conv3d_block(64,  128),  # → (B, 128, T,  28,  28)
            _conv3d_block(128, 256),  # → (B, 256, T,  14,  14)
        )

        # ── Spatial pool → temporal token sequence ────────────────────────────
        # (B, 256, T, 14, 14) → (B, 256, T, 1, 1) → (B, T, 256)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # ── Project to d_model ────────────────────────────────────────────────
        self.input_proj = nn.Linear(256, d_model)

        # ── CLS token + learnable positional embeddings ───────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # max 16 temporal tokens + 1 CLS
        self.pos_embedding = nn.Parameter(torch.zeros(1, 17, d_model))
        nn.init.trunc_normal_(self.cls_token,     std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # pre-norm (more training-stable)
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

        x = self.backbone(x)                          # (B, 256, T, 14, 14)
        x = self.spatial_pool(x)                      # (B, 256, T,  1,  1)
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, T, 256)
        x = self.input_proj(x)                        # (B, T, d_model)

        # Prepend CLS token and add positional embeddings
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                # (B, T+1, d_model)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]

        x = self.transformer(x)                       # (B, T+1, d_model)

        return self.head(x[:, 0, :])                  # (B, num_classes)
