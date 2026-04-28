"""
ResNet34 + Temporal Shift Module (TSM) + Temporal Transformer.

TSM inserts a channel-shift operation before each ResNet BasicBlock so the
backbone exchanges temporal information across frames without any extra
parameters.  A lightweight Transformer + attention pooling then aggregates
the per-frame features into a clip-level prediction.

Input shape  : (B, T, 3, H, W)
Output shape : (B, num_classes)
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
        out[:, 1:, :fold] = x[:, :-1, :fold]             # past → current
        out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]  # future → current
        out[:, 0, :fold] = 0
        out[:, -1, fold : 2 * fold] = 0

        return out.view(bt, c, h, w)


class _TSMBlock(nn.Module):
    """Wrap an existing residual block with a temporal shift."""

    def __init__(self, block: nn.Module, n_segment: int, fold_div: int = 8) -> None:
        super().__init__()
        self.shift = TemporalShift(n_segment, fold_div)
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.shift(x))


def _wrap_layer(layer: nn.Sequential, n_segment: int, fold_div: int) -> nn.Sequential:
    return nn.Sequential(*[_TSMBlock(b, n_segment, fold_div) for b in layer])


class ResNet34TSMTransformer(nn.Module):
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
        weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        rn = tv_models.resnet34(weights=weights)

        rn.layer1 = _wrap_layer(rn.layer1, num_frames, fold_div)
        rn.layer2 = _wrap_layer(rn.layer2, num_frames, fold_div)
        rn.layer3 = _wrap_layer(rn.layer3, num_frames, fold_div)
        rn.layer4 = _wrap_layer(rn.layer4, num_frames, fold_div)
        rn.fc = nn.Identity()  # output: (B*T, 512)
        self.backbone = rn

        # ── Temporal head ─────────────────────────────────────────────────────
        self.input_proj = nn.Linear(512, d_model)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        feats = self.backbone(x.view(b * t, c, h, w))  # (B*T, 512)
        feats = feats.view(b, t, -1)                   # (B, T, 512)

        feats = self.input_proj(feats) + self.pos_embed  # (B, T, d_model)
        encoded = self.transformer(feats)                # (B, T, d_model)

        attn = torch.softmax(self.attn_pool(encoded), dim=1)  # (B, T, 1)
        pooled = (encoded * attn).sum(dim=1)                  # (B, d_model)

        return self.head(pooled)

    def get_param_groups(self, base_lr: float, backbone_lr_factor: float = 0.1):
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        other_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": other_params, "lr": base_lr},
            {"params": list(self.backbone.parameters()), "lr": base_lr * backbone_lr_factor},
        ]
