"""
VideoMAE-Large fine-tuned on Kinetics-400, adapted for 33-class prediction.

Loads MCG-NJU/videomae-large-finetuned-kinetics from HuggingFace, replaces the
400-class Kinetics head with a 33-class MLP head, and uses Layer-wise LR Decay (LLRD).

Architecture: ViT-Large (307M params, hidden_size=1024, 24 transformer blocks).
Kinetics pretraining is less close to SSv2 than SSv2 pretraining, so the backbone
LR factor can be slightly higher to allow more adaptation.

LLRD: block[23] (top) → backbone_lr, block[0] (bottom) → backbone_lr × llrd^23,
      embeddings → backbone_lr × llrd^24.

Input shape  : (B, T, C, H, W)   — T=16 to match pretrained positional embeddings
Output shape : (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEConfig


class VideoMAELargeKineticsFinetune(nn.Module):

    HF_MODEL_ID = "MCG-NJU/videomae-large-finetuned-kinetics"

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        llrd: float = 0.75,
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        self.llrd = llrd

        if pretrained:
            self.model = VideoMAEForVideoClassification.from_pretrained(
                self.HF_MODEL_ID,
                ignore_mismatched_sizes=True,
            )
        else:
            config = VideoMAEConfig.from_pretrained(self.HF_MODEL_ID, local_files_only=True)
            config.num_labels = num_classes
            self.model = VideoMAEForVideoClassification(config)

        self._interpolate_position_embeddings(num_frames)
        self.model.gradient_checkpointing_enable()

        hidden_size = self.model.config.hidden_size  # 1024 for large

        # Deeper MLP head: Kinetics features need more remapping to SSv2 semantics
        self.model.classifier = nn.Identity()
        fc1 = nn.Linear(hidden_size, hidden_size)
        fc2 = nn.Linear(hidden_size, hidden_size // 2)
        fc3 = nn.Linear(hidden_size // 2, num_classes)
        for fc in (fc1, fc2, fc3):
            nn.init.trunc_normal_(fc.weight, std=0.02)
            nn.init.zeros_(fc.bias)
        self.head = nn.Sequential(
            fc1,
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.5),
            fc2,
            nn.GELU(),
            nn.Dropout(0.3),
            fc3,
        )

    def _interpolate_position_embeddings(self, num_frames: int) -> None:
        cfg = self.model.config
        tubelet_size = cfg.tubelet_size        # 2
        patch_size = cfg.patch_size            # 16
        image_size = cfg.image_size            # 224

        h = w = image_size // patch_size       # 14
        t_target = num_frames // tubelet_size  # e.g. 2 for T=4

        pe = self.model.videomae.embeddings.position_embeddings  # [1, T_pre*H*W, D]
        t_pre = pe.shape[1] // (h * w)        # e.g. 8 for T=16

        if t_pre == t_target:
            return

        d = pe.shape[-1]
        data = pe.data.reshape(1, t_pre, h, w, d).permute(0, 4, 1, 2, 3).float()  # [1, D, T, H, W]
        data = F.interpolate(data, size=(t_target, h, w), mode="trilinear", align_corners=False)
        data = data.permute(0, 2, 3, 4, 1).reshape(1, t_target * h * w, d)
        self.model.videomae.embeddings.position_embeddings = nn.Parameter(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # VideoMAE expects (B, T, C, H, W)
        return self.head(self.model(pixel_values=x).logits)

    def get_param_groups(self, base_lr: float, backbone_lr_factor: float = 0.1):
        backbone_base_lr = base_lr * backbone_lr_factor
        groups = []

        # --- Head: full base_lr ---
        head_params = list(self.head.parameters())
        if self.model.fc_norm is not None:
            head_params += list(self.model.fc_norm.parameters())
        groups.append({"params": head_params, "lr": base_lr})

        # --- Transformer blocks: LLRD from top (high LR) to bottom (low LR) ---
        encoder_layers = self.model.videomae.encoder.layer
        num_layers = len(encoder_layers)  # 24 for ViT-Large
        for depth, layer in enumerate(reversed(encoder_layers)):
            lr = backbone_base_lr * (self.llrd ** depth)
            groups.append({"params": list(layer.parameters()), "lr": lr})

        # --- Patch embeddings + backbone layernorm: lowest LR ---
        emb_lr = backbone_base_lr * (self.llrd ** num_layers)
        emb_params = list(self.model.videomae.embeddings.parameters())
        if hasattr(self.model.videomae, "layernorm") and self.model.videomae.layernorm is not None:
            emb_params += list(self.model.videomae.layernorm.parameters())
        groups.append({"params": emb_params, "lr": emb_lr})

        return groups
