"""
ConvNeXt-Tiny + Temporal Shift Module (TSM) + Temporal Transformer.

TSM shifts a fraction of channels along the temporal axis before each
ConvNeXt block so the backbone sees cross-frame context without extra
parameters.  A Transformer + attention pooling then aggregates per-frame
features into a clip-level prediction.

Input shape  : (B, T, 3, H, W)
Output shape : (B, num_classes)

Architecture
------------
(B, T, 3, 224, 224)
        │
        ▼  reshape → (B*T, 3, 224, 224)
┌───────────────────────────────────────────────┐
│  ConvNeXt-Tiny backbone (pretrained ImageNet) │
│                                               │
│  Stem : Conv2d 4×4 s4 + LayerNorm            │
│  Stage 0 (3 blocks)  ← TSM before each block │
│  Downsample                                   │
│  Stage 1 (3 blocks)  ← TSM before each block │
│  Downsample                                   │
│  Stage 2 (9 blocks)  ← TSM before each block │
│  Downsample                                   │
│  Stage 3 (3 blocks)  ← TSM before each block │
│                       18 TSM blocks total     │
│  AdaptiveAvgPool2d(1) + LayerNorm             │
└───────────────────────────────────────────────┘
        │  (B*T, 768)
        ▼  reshape → (B, T, 768)
┌───────────────────────────────────────────────┐
│  Linear(768 → d_model) + positional embedding │
└───────────────────────────────────────────────┘
        │  (B, T, d_model)
        ▼
┌───────────────────────────────────────────────┐
│  Transformer Encoder  ×  num_layers           │
│  (d_model, nhead, ffn=2048, GELU, pre-norm)   │
└───────────────────────────────────────────────┘
        │  (B, T, d_model)
        ▼
┌───────────────────────────────────────────────┐
│  Attention Pooling                            │
│  score_t = softmax(Linear(enc_t))             │
│  pooled  = Σ score_t × enc_t                 │
└───────────────────────────────────────────────┘
        │  (B, d_model)
        ▼
┌───────────────────────────────────────────────┐
│  Head : LayerNorm → Dropout → Linear(33)      │
└───────────────────────────────────────────────┘
        │  (B, 33)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


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
        out[:, 1:, :fold] = x[:, :-1, :fold]                      # past → current
        out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]  # future → current
        out[:, 0, :fold] = 0
        out[:, -1, fold : 2 * fold] = 0

        return out.view(bt, c, h, w)


class _TSMBlock(nn.Module):
    """Wrap an existing block with a temporal shift."""

    def __init__(self, block: nn.Module, n_segment: int, fold_div: int = 8) -> None:
        super().__init__()
        self.shift = TemporalShift(n_segment, fold_div)
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.shift(x))


def _wrap_stage(stage: nn.Sequential, n_segment: int, fold_div: int) -> nn.Sequential:
    return nn.Sequential(*[_TSMBlock(b, n_segment, fold_div) for b in stage])


class ConvNeXtTSMTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        pretrained: bool = True,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        fold_div: int = 8,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        # ── Backbone ──────────────────────────────────────────────────────────
        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        cnxt = tv_models.convnext_tiny(weights=weights)

        # features layout: [stem, stage0, down, stage1, down, stage2, down, stage3]
        # Insert TSM before each block in the 4 stages (indices 1, 3, 5, 7)
        for stage_idx in (1, 3, 5, 7):
            cnxt.features[stage_idx] = _wrap_stage(
                cnxt.features[stage_idx],  # type: ignore[arg-type]
                num_frames,
                fold_div,
            )

        self.features = cnxt.features   # output: (B*T, 768, H', W')
        self.avgpool = cnxt.avgpool     # AdaptiveAvgPool2d(1) → (B*T, 768, 1, 1)
        self.norm = cnxt.classifier[0]  # LayerNorm2d(768)
        self.flatten = cnxt.classifier[1]  # Flatten → (B*T, 768)

        feat_dim = 768

        # ── Temporal head ─────────────────────────────────────────────────────
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = nn.Linear(d_model, 1)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)   # (B*T, 768, H', W')
        x = self.avgpool(x)    # (B*T, 768, 1, 1)
        x = self.norm(x)       # (B*T, 768, 1, 1)
        return self.flatten(x) # (B*T, 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        feats = self._extract_features(x.view(b * t, c, h, w))  # (B*T, 768)
        feats = feats.view(b, t, -1)                              # (B, T, 768)

        feats = self.input_proj(feats) + self.pos_embed  # (B, T, d_model)
        encoded = self.transformer(feats)                # (B, T, d_model)

        attn = torch.softmax(self.attn_pool(encoded), dim=1)  # (B, T, 1)
        pooled = (encoded * attn).sum(dim=1)                  # (B, d_model)

        return self.head(pooled)

    def get_param_groups(self, base_lr: float, backbone_lr_factor: float = 0.1):
        backbone_params = list(self.features.parameters()) + list(self.norm.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": other_params, "lr": base_lr},
            {"params": backbone_params, "lr": base_lr * backbone_lr_factor},
        ]
