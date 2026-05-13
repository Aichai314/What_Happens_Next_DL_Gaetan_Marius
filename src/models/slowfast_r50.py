"""
SlowFast R50 pretrained on Kinetics-400, adapted for 33-class prediction.

Loads pytorchvideo.models.hub.slowfast_r50 and replaces the 400-class head with
a 33-class MLP head. SlowFast is a two-pathway 3D-CNN: the slow path captures
appearance (low temporal rate), the fast path captures motion (high temporal rate).

Architecture rationale: maximally decorrelated from VideoMAE (ViT + masked
modeling). Pure 3D convolutions, supervised Kinetics pretraining, two-stream
design - the features should be orthogonal to VideoMAE's, which is exactly what
the XGBoost meta-learner needs to gain real ensemble lift.

Input shape  : (B, T, C, H, W)
Output shape : (B, num_classes)

The forward splits the input into slow (T/alpha frames) and fast (T frames)
pathways internally, so the dataset just provides a single clip of T frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50


class SlowFastR50Finetune(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        alpha: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if alpha != 4:
            raise ValueError(
                f"SlowFast R50 (pytorchvideo) requires alpha=4 to match its "
                f"pretrained fast->slow fusion conv (stride hardcoded to 4). "
                f"Got alpha={alpha}."
            )
        self.alpha = alpha

        self.backbone = slowfast_r50(pretrained=pretrained)

        # blocks[5] (PoolConcatPathway) holds the fixed AvgPool3d kernels sized for
        # canonical T_slow=8, T_fast=32. Replace with adaptive pooling so num_frames
        # can vary while still using pretrained backbone weights.
        pool_block = self.backbone.blocks[5]
        pool_block.pool = nn.ModuleList([
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # slow pathway
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # fast pathway
        ])

        # blocks[6] is the ResNetBasicHead; replace its projection with an MLP head
        head = self.backbone.blocks[-1]
        hidden_size = head.proj.in_features  # 2304 = 2048 (slow) + 256 (fast)
        head.proj = nn.Identity()
        head.activation = nn.Identity()
        # output_pool and the final view in ResNetBasicHead.forward give us [B, 2304]

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def _pack_pathways(self, x: torch.Tensor) -> list[torch.Tensor]:
        # (B, T, C, H, W) -> (B, C, T, H, W) for 3D conv layout
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        T = x.size(2)
        # Slow pathway: T // alpha frames, evenly spaced
        slow_indices = torch.linspace(0, T - 1, T // self.alpha).long().to(x.device)
        slow = torch.index_select(x, 2, slow_indices)
        # Fast pathway: all T frames
        return [slow, x]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(self._pack_pathways(x))  # [B, 2304]
        return self.head(features)

    def get_param_groups(self, base_lr: float, backbone_lr_factor: float = 0.1):
        backbone_lr = base_lr * backbone_lr_factor
        head_params = list(self.head.parameters())
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        return [
            {"params": head_params, "lr": base_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]
