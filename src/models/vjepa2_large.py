"""
V-JEPA 2 ViT-Large fine-tuned on SSv2, adapted for 33-class prediction.

Loads facebook/vjepa2-vitl-fpc16-256-ssv2: ViT-Large pretrained with the
predictive Joint-Embedding objective on 1M+ hours of video, then fine-tuned
on Something-Something v2 (174 classes). We replace the final classifier
with a 33-class MLP head, keeping the attentive pooler (also pretrained).

V-JEPA differs from VideoMAE: instead of reconstructing masked pixels, it
predicts the latent embeddings of masked patches. Features specialize in
temporal/relational dynamics — ideal for SSv2 manipulation tasks.

Position embeddings use 3D RoPE so T can vary without interpolation.

Architecture (ViT-L):
- 24 transformer blocks, hidden_size=1024
- Input: T=16, image_size=256, patch_size=16, tubelet_size=2 → 2048 patch tokens
- Attentive pooler (cross-attn + 3 self-attn) reduces tokens to a single 1024-dim vector

Input shape  : (B, T, C, 256, 256)
Output shape : (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VJEPA2ForVideoClassification, VJEPA2Config


class VJEPA2LargeFinetune(nn.Module):

    HF_MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        llrd: float = 0.85,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.llrd = llrd

        if pretrained:
            self.model = VJEPA2ForVideoClassification.from_pretrained(
                self.HF_MODEL_ID,
                ignore_mismatched_sizes=True,
            )
        else:
            config = VJEPA2Config.from_pretrained(self.HF_MODEL_ID)
            config.num_labels = num_classes
            self.model = VJEPA2ForVideoClassification(config)

        # Predictor is only used during the JEPA pretraining objective; remove
        # to save ~250M params and the corresponding memory at fwd/bwd time.
        if hasattr(self.model.vjepa2, "predictor"):
            self.model.vjepa2.predictor = None

        if hasattr(self.model.vjepa2.encoder, "gradient_checkpointing_enable"):
            self.model.vjepa2.encoder.gradient_checkpointing_enable()
        elif hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        hidden_size = self.model.config.hidden_size  # 1024 for ViT-L

        # Replace the 174-class linear classifier with a deeper MLP head.
        # The attentive pooler stays pretrained (it produces a 1024-dim vector).
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
            nn.Dropout(head_dropout),
            fc2,
            nn.GELU(),
            nn.Dropout(head_dropout * 0.6),
            fc3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # V-JEPA 2 expects (B, T, C, H, W) under the key `pixel_values_videos`
        return self.head(self.model(pixel_values_videos=x).logits)

    def get_param_groups(self, base_lr: float, backbone_lr_factor: float = 0.1):
        backbone_base_lr = base_lr * backbone_lr_factor
        groups = []

        # --- Head + attentive pooler: full base_lr ---
        # Pooler adapts encoder features to the task — train it like a head
        head_params = list(self.head.parameters())
        head_params += list(self.model.pooler.parameters())
        groups.append({"params": head_params, "lr": base_lr})

        # --- Encoder transformer blocks: LLRD top → bottom ---
        encoder_layers = self.model.vjepa2.encoder.layer
        num_layers = len(encoder_layers)  # 24
        for depth, layer in enumerate(reversed(encoder_layers)):
            lr = backbone_base_lr * (self.llrd ** depth)
            groups.append({"params": list(layer.parameters()), "lr": lr})

        # --- Patch embeddings + final layernorm: lowest LR ---
        emb_lr = backbone_base_lr * (self.llrd ** num_layers)
        emb_params = list(self.model.vjepa2.encoder.embeddings.parameters())
        if hasattr(self.model.vjepa2.encoder, "layernorm") and self.model.vjepa2.encoder.layernorm is not None:
            emb_params += list(self.model.vjepa2.encoder.layernorm.parameters())
        groups.append({"params": emb_params, "lr": emb_lr})

        return groups
