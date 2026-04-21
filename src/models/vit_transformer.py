"""
ViT + Temporal Transformer for video classification (Track B).

Architecture:
  (B, T, C, H, W)
  → ViT-B/16 (ImageNet pretrained, last `unfreeze_blocks` blocks fine-tuned)
      each frame → mean of all encoder tokens → (B, T, 768)
  → temporal CLS + learnable positional embedding
  → Temporal Transformer encoder  (6 layers, pre-norm)
  → temporal CLS → classification head → (B, num_classes)

Improvements over the naive version:
  1. Mean-pool all ViT tokens (CLS + 196 patches) instead of CLS only
     → captures full spatial context per frame.
  2. Partial ViT fine-tuning (last `unfreeze_blocks` encoder blocks)
     → backbone adapts to action recognition while keeping lower layers stable.
  3. Separate learning-rate groups (backbone_lr_factor << 1)
     → fine-tuning without destroying pretrained weights.
"""

from __future__ import annotations

from typing import List, Dict, Any

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


_VIT_DIM = 768   # ViT-B hidden dimension


class ViTTransformer(nn.Module):
    """
    Args:
        num_classes      : number of output classes.
        unfreeze_blocks  : number of ViT encoder blocks (from the end) to fine-tune.
                           0 = fully frozen backbone.
        temporal_layers  : number of temporal TransformerEncoder layers.
        temporal_heads   : attention heads in the temporal transformer.
        dropout          : dropout in temporal transformer and head.
    """

    def __init__(
        self,
        num_classes: int,
        unfreeze_blocks: int = 4,
        temporal_layers: int = 6,
        temporal_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Spatial backbone: ViT-B/16 pretrained on ImageNet ─────────────────
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        vit.heads = nn.Identity()   # remove classification head
        self.vit = vit

        # Freeze everything first, then selectively unfreeze
        for p in self.vit.parameters():
            p.requires_grad_(False)

        if unfreeze_blocks > 0:
            encoder_blocks = list(self.vit.encoder.layers)  # 12 blocks
            for block in encoder_blocks[-unfreeze_blocks:]:
                for p in block.parameters():
                    p.requires_grad_(True)
            # Also unfreeze encoder layer-norm
            for p in self.vit.encoder.ln.parameters():
                p.requires_grad_(True)

        # ── Temporal CLS token + learnable positional embeddings ──────────────
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, _VIT_DIM))
        self.temporal_pos = nn.Parameter(torch.zeros(1, 17, _VIT_DIM))  # max 16 frames + 1 CLS
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

    def get_param_groups(self, base_lr: float, backbone_lr_factor: float = 0.1) -> List[Dict[str, Any]]:
        """
        Return optimizer param groups with a lower LR for fine-tuned ViT blocks.
        Use backbone_lr_factor=0.1 → ViT blocks train at 10× lower LR than the
        temporal transformer, preventing destruction of pretrained features.
        """
        vit_params   = [p for p in self.vit.parameters()       if p.requires_grad]
        other_params = [p for n, p in self.named_parameters()
                        if not n.startswith("vit.") and p.requires_grad]
        groups = [{"params": other_params, "lr": base_lr}]
        if vit_params:
            groups.append({"params": vit_params, "lr": base_lr * backbone_lr_factor})
        return groups

    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run ViT on each frame and return mean of ALL encoder tokens (CLS + patches).
        Mean-pooling captures global spatial context better than CLS alone.

        Args:
            x: (B, T, C, H, W)
        Returns:
            feats: (B, T, 768)
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        # Patch embedding + positional encoding  → (B*T, 196, 768)
        x_emb = self.vit._process_input(x)
        # Prepend CLS token                      → (B*T, 197, 768)
        batch_cls = self.vit.class_token.expand(B * T, -1, -1)
        x_emb = torch.cat([batch_cls, x_emb], dim=1)
        # ViT encoder (transformer blocks)       → (B*T, 197, 768)
        x_enc = self.vit.encoder(x_emb)

        # Mean over all 197 tokens (vs CLS-only which discards 99.5% of tokens)
        feats = x_enc.mean(dim=1)                          # (B*T, 768)
        return feats.reshape(B, T, _VIT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B = x.size(0)

        frame_feats = self._extract_frame_features(x)       # (B, T, 768)

        t_cls = self.temporal_cls.expand(B, -1, -1)
        seq = torch.cat([t_cls, frame_feats], dim=1)         # (B, T+1, 768)
        seq = seq + self.temporal_pos[:, :seq.size(1), :]

        seq = self.temporal_transformer(seq)                 # (B, T+1, 768)

        return self.head(seq[:, 0, :])                       # (B, num_classes)
