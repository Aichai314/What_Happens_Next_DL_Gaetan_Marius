"""
ViT + Temporal Transformer for video classification (Track B).

Architecture:
  (B, T, C, H, W)
  → ViT-B/16 frame encoder  (pretrained ImageNet, shared weights across frames)
      each frame → CLS token → (B, T, 768)
  → temporal CLS token + learnable positional embedding
  → Temporal Transformer encoder  (4-6 layers)
  → temporal CLS → classification head → (B, num_classes)

The spatial ViT captures rich per-frame semantics; the temporal Transformer
learns which frames and which interactions matter for action recognition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


_VIT_DIM = 768   # ViT-B hidden dimension


class ViTTransformer(nn.Module):
    """
    Args:
        num_classes    : number of output classes.
        freeze_vit     : freeze ViT backbone weights (faster training, less GPU).
        temporal_layers: number of temporal TransformerEncoder layers.
        temporal_heads : attention heads in the temporal transformer.
        dropout        : dropout in temporal transformer and head.
    """

    def __init__(
        self,
        num_classes: int,
        freeze_vit: bool = True,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Spatial backbone: ViT-B/16 pretrained on ImageNet ─────────────────
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # Strip the classification head — we only need the CLS token features
        vit.heads = nn.Identity()
        self.vit = vit

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad_(False)

        # ── Temporal CLS token + learnable positional embeddings ──────────────
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, _VIT_DIM))
        # max 16 frames + 1 temporal CLS
        self.temporal_pos = nn.Parameter(torch.zeros(1, 17, _VIT_DIM))
        nn.init.trunc_normal_(self.temporal_cls, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        # ── Temporal Transformer encoder ──────────────────────────────────────
        t_layer = nn.TransformerEncoderLayer(
            d_model=_VIT_DIM,
            nhead=temporal_heads,
            dim_feedforward=_VIT_DIM * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            t_layer,
            num_layers=temporal_layers,
            enable_nested_tensor=False,
        )

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(_VIT_DIM),
            nn.Dropout(dropout),
            nn.Linear(_VIT_DIM, num_classes),
        )

    # -------------------------------------------------------------------------

    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract ViT CLS-token features for each frame independently.

        Args:
            x: (B, T, C, H, W)
        Returns:
            feats: (B, T, 768)
        """
        B, T, C, H, W = x.shape
        # Merge batch and time dims for a single ViT forward pass
        x = x.reshape(B * T, C, H, W)
        feats = self.vit(x)          # (B*T, 768)  — heads=Identity
        return feats.reshape(B, T, _VIT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B = x.size(0)

        # Per-frame spatial features
        frame_feats = self._extract_frame_features(x)   # (B, T, 768)

        # Prepend temporal CLS token
        t_cls = self.temporal_cls.expand(B, -1, -1)
        seq = torch.cat([t_cls, frame_feats], dim=1)     # (B, T+1, 768)

        # Add positional embeddings
        seq_len = seq.size(1)
        seq = seq + self.temporal_pos[:, :seq_len, :]

        # Temporal self-attention
        seq = self.temporal_transformer(seq)             # (B, T+1, 768)

        # Temporal CLS → head
        return self.head(seq[:, 0, :])                   # (B, num_classes)
