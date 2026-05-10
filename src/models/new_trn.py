"""
Multi-Scale Temporal Relation Network (M-TRN)
Faithfully reimplements the MIT paper:
  "Temporal Relational Reasoning in Videos" (Zhou et al., ECCV 2018)
  https://arxiv.org/abs/1711.08496

Key corrections vs. previous version:
  1. Each relation pair/triplet/quad has its OWN independent RelationModule
     (as in the paper) rather than a single shared MLP per scale.
  2. Only temporally ordered, contiguous frame combinations are used,
     consistent with the paper's intent to capture temporal progression.
  3. Scale outputs are averaged (not summed) before being combined, so
     each scale contributes equally regardless of how many combinations
     it produces.
"""

from __future__ import annotations

from itertools import combinations
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torchvision import models
from utils import two_stage_trainer, inject_tsm_into_resnet, replace_resnet_stem


class RelationModule(nn.Module):
    """
    Independent MLP for a single frame combination.
    Each pair / triplet / quad gets its own instance (paper §3.2).
    """
    def __init__(
        self,
        feature_dim: int,
        num_frames_in_relation: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        in_dim = feature_dim * num_frames_in_relation
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@two_stage_trainer
class TRNNew(nn.Module):
    """
    Multi-Scale TRN for exactly T=4 frames.

    Relation scales used (contiguous ordered subsets only):
      - 2-frame : (0,1), (1,2), (2,3)          — 3 modules
      - 3-frame : (0,1,2), (1,2,3)              — 2 modules
      - 4-frame : (0,1,2,3)                     — 1 module

    Final logits = mean over scales of (mean over combinations within scale).
    """

    # Contiguous, ordered combinations for T=4
    _PAIRS    = [(0, 1), (1, 2), (2, 3)]
    _TRIPLETS = [(0, 1, 2), (1, 2, 3)]
    _QUADS    = [(0, 1, 2, 3)]

    def __init__(
        self,
        model_cfg: DictConfig,
        num_classes: int,
        num_frames: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        relation_hidden_dim = int(model_cfg.get("relation_hidden_dim", 256))
        in_channels         = int(model_cfg.get("in_channels", 3))
        size                = int(model_cfg.get("backbone_size", 18))
        n_div               = int(model_cfg.get("fold_div", 8))
        self.resnet         = False

        # ── Backbone ────────────────────────────────────────────────────────
        if model_cfg.get("efficientnet", False):
            weights  = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
        elif size == 18:
            weights  = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            self.resnet = True
        elif size == 34:
            weights  = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
            self.resnet = True
        else:
            raise ValueError(f"Unsupported backbone size: {size}. Choose 18 or 34.")

        if self.resnet:
            backbone = replace_resnet_stem(backbone, in_channels=in_channels)

        # TSM is injected only when a pre-trained backbone path is provided
        # (intentional: allows testing TRN without TSM as an ablation)
        if (
            model_cfg.get("pretrained_backbone_path") is not None
            or model_cfg.get("pursue_from") is not None
        ):
            backbone = inject_tsm_into_resnet(backbone, num_frames=num_frames, n_div=n_div)

        if self.resnet:
            feature_dim    = backbone.fc.in_features
            backbone.fc    = nn.Identity()
        else:
            feature_dim            = backbone.classifier[1].in_features
            backbone.classifier    = nn.Identity()

        self.backbone    = backbone
        self.feature_dim = feature_dim

        # ── One independent RelationModule per combination (paper §3.2) ────
        self.relation_modules_2 = nn.ModuleList([
            RelationModule(feature_dim, 2, relation_hidden_dim, num_classes)
            for _ in self._PAIRS
        ])
        self.relation_modules_3 = nn.ModuleList([
            RelationModule(feature_dim, 3, relation_hidden_dim, num_classes)
            for _ in self._TRIPLETS
        ])
        self.relation_modules_4 = nn.ModuleList([
            RelationModule(feature_dim, 4, relation_hidden_dim, num_classes)
            for _ in self._QUADS
        ])

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _mean_over_relations(
        feats: torch.Tensor,
        combos: list[tuple[int, ...]],
        modules: nn.ModuleList,
    ) -> torch.Tensor:
        """
        Average the logits produced by each (combo, module) pair.

        Args:
            feats   : (B, T, D)
            combos  : list of frame-index tuples
            modules : one RelationModule per combo

        Returns:
            (B, num_classes) — mean logits across all combinations
        """
        total = None
        for idx_tuple, module in zip(combos, modules):
            cat = torch.cat([feats[:, i, :] for i in idx_tuple], dim=1)  # (B, D*k)
            out = module(cat)                                              # (B, C)
            total = out if total is None else total + out
        return total / len(combos)  # mean, not sum

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_batch : (B, T=4, C, H, W)

        Returns:
            logits      : (B, num_classes)
        """
        B, T, C, H, W = video_batch.shape
        assert T == 4, "This TRN is hard-coded for exactly 4 frames."

        # Extract per-frame features
        feats = self.backbone(video_batch.view(B * T, C, H, W))  # (B*T, D)
        feats = torch.flatten(feats, start_dim=1)                 # (B*T, D)
        feats = feats.view(B, T, -1)                              # (B, T, D)

        # Compute mean logits at each temporal scale
        out_2 = self._mean_over_relations(feats, self._PAIRS,    self.relation_modules_2)
        out_3 = self._mean_over_relations(feats, self._TRIPLETS, self.relation_modules_3)
        out_4 = self._mean_over_relations(feats, self._QUADS,    self.relation_modules_4)

        # Final prediction: mean across scales (each scale equally weighted)
        return (out_2 + out_3 + out_4) / 3