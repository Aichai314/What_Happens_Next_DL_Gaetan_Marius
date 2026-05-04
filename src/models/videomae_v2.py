"""
VideoMAE fine-tuned on Something-Something v2, adapted for 33-class prediction.

Loads MCG-NJU/videomae-base-finetuned-ssv2 from HuggingFace, replaces the
174-class SSv2 head with a 33-class head, and uses Layer-wise LR Decay (LLRD)
so lower transformer blocks stay close to their pretrained weights.

LLRD: block[11] (top) → backbone_lr, block[0] (bottom) → backbone_lr × llrd^11,
      embeddings → backbone_lr × llrd^12.

Input shape  : (B, T, C, H, W)   — T=16 to match pretrained positional embeddings
Output shape : (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEConfig


class VideoMAEFinetune(nn.Module):

    HF_MODEL_ID = "MCG-NJU/videomae-base-finetuned-ssv2"

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        llrd: float = 0.75,
    ) -> None:
        super().__init__()
        self.llrd = llrd

        if pretrained:
            self.model = VideoMAEForVideoClassification.from_pretrained(
                self.HF_MODEL_ID,
                ignore_mismatched_sizes=True,
            )
        else:
            # Use local cache only — avoids any network call when we'll load a checkpoint anyway
            config = VideoMAEConfig.from_pretrained(self.HF_MODEL_ID, local_files_only=True)
            config.num_labels = num_classes
            self.model = VideoMAEForVideoClassification(config)

        hidden_size = self.model.config.hidden_size  # 768 for base

        # Swap SSv2 head (174 classes) → our 33-class head
        self.model.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.trunc_normal_(self.model.classifier.weight, std=0.02)
        nn.init.zeros_(self.model.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # VideoMAE expects (B, T, C, H, W) — same as dataset output, no permute needed
        return self.model(pixel_values=x).logits

    def get_param_groups(self, base_lr: float, backbone_lr_factor: float = 0.1):
        backbone_base_lr = base_lr * backbone_lr_factor
        groups = []

        # --- Head: full base_lr ---
        head_params = list(self.model.classifier.parameters())
        if self.model.fc_norm is not None:
            head_params += list(self.model.fc_norm.parameters())
        groups.append({"params": head_params, "lr": base_lr})

        # --- Transformer blocks: LLRD from top (high LR) to bottom (low LR) ---
        encoder_layers = self.model.videomae.encoder.layer
        num_layers = len(encoder_layers)  # 12 for ViT-Base
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
