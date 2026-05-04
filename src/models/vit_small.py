"""
ViT-Small + Temporal Transformer built specifically for CLOSED TRACK (From Scratch).

Architecture:
  (B, T, C, H, W)
  → ViT-Small (Randomly initialized, ALL layers unfreezed)
      each frame → mean of all encoder tokens → (B, T, 384)
  → temporal CLS + learnable positional embedding
  → Temporal Transformer encoder  (4 layers)
  → temporal CLS → classification head → (B, num_classes)
"""

from __future__ import annotations
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

_VIT_DIM = 384   # ViT-Small hidden dimension (Half the size of ViT-Base)

class ViTSmallTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False, # Accepted for API compatibility, but bypassed
        temporal_layers: int = 4,
        temporal_heads: int = 6,
        dropout: float = 0.3,     # Increased dropout to prevent from-scratch overfitting
    ) -> None:
        super().__init__()

        if pretrained:
            print("⚠️ Warning: pretrained=True is ignored. ViTSmall is built for Closed Track (from scratch).")

        # ── Spatial backbone: ViT-Small (From Scratch) ─────────────────
        # Standard ViT-Small configuration: 12 layers, 6 heads, 384 dim
        self.vit = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=_VIT_DIM,
            mlp_dim=1536,
            num_classes=num_classes
        )
        self.vit.heads = nn.Identity()   # Remove classification head

        # Ensure EVERYTHING is unfrozen (Mandatory when training from scratch)
        for p in self.vit.parameters():
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

    def get_param_groups(self, base_lr: float, backbone_lr_factor: float = 1.0) -> List[Dict[str, Any]]:
        """
        Since we train from scratch, backbone_lr_factor should be 1.0 (or close to it)
        so the ViT learns at the same speed as the Temporal Transformer.
        """
        vit_params   = [p for p in self.vit.parameters() if p.requires_grad]
        other_params = [p for n, p in self.named_parameters()
                        if not n.startswith("vit.") and p.requires_grad]
        
        groups = [{"params": other_params, "lr": base_lr}]
        if vit_params:
            groups.append({"params": vit_params, "lr": base_lr * backbone_lr_factor})
        return groups

    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        # Patch embedding + positional encoding
        x_emb = self.vit._process_input(x)
        # Prepend CLS token
        batch_cls = self.vit.class_token.expand(B * T, -1, -1)
        x_emb = torch.cat([batch_cls, x_emb], dim=1)
        # ViT encoder
        x_enc = self.vit.encoder(x_emb)

        # Mean over all tokens
        feats = x_enc.mean(dim=1)                          
        return feats.reshape(B, T, _VIT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        frame_feats = self._extract_frame_features(x)        # (B, T, 384)

        t_cls = self.temporal_cls.expand(B, -1, -1)
        seq = torch.cat([t_cls, frame_feats], dim=1)         # (B, T+1, 384)
        seq = seq + self.temporal_pos[:, :seq.size(1), :]

        seq = self.temporal_transformer(seq)                 # (B, T+1, 384)

        return self.head(seq[:, 0, :])                       # (B, num_classes)