"""
Multi-Scale Temporal Relation Network (M-TRN)
Explicitly compares combinations of frames to recognize temporal transformations.
Highly optimized for exactly 4 frames.
"""

from __future__ import annotations

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torchvision import models
from utils import two_stage_trainer, inject_tsm_into_resnet, replace_resnet_stem


class RelationModule(nn.Module):
    """A small MLP that processes a specific combination of concatenated frames."""
    def __init__(self, feature_dim: int, num_frames_in_relation: int, hidden_dim: int, num_classes: int):
        super().__init__()
        # Input size = feature_dim (512) * how many frames we are comparing
        self.fc1 = nn.Linear(feature_dim * num_frames_in_relation, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        return self.fc2(out)

@two_stage_trainer
class TRN(nn.Module):
    def __init__(
        self,
        model_cfg: DictConfig,
        num_classes: int,
        num_frames: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        relation_hidden_dim = int(model_cfg.get("relation_hidden_dim", 256))
        in_channels = int(model_cfg.get("in_channels", 3))
        size = int(model_cfg.get("backbone_size", 18))
        n_div = int(model_cfg.get("fold_div", 8))

        if size == 18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        elif size == 34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported model size: {size}. Choose 18 or 34.")

        backbone = replace_resnet_stem(backbone, in_channels=in_channels)

        if model_cfg.get("pretrained_backbone_path") is not None or model_cfg.get("pursue_from") is not None:
            # Inject TSM
            backbone = inject_tsm_into_resnet(backbone, num_frames=num_frames, n_div=n_div)
        
        feature_dim = backbone.fc.in_features 
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # 2. The Relation Modules (Multi-Scale)
        # We create a separate "brain" for 2-frame, 3-frame, and 4-frame comparisons
        self.relation_2 = RelationModule(feature_dim, 2, relation_hidden_dim, num_classes)
        self.relation_3 = RelationModule(feature_dim, 3, relation_hidden_dim, num_classes)
        self.relation_4 = RelationModule(feature_dim, 4, relation_hidden_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (B, T=4, C, H, W)
        returns logits: (B, num_classes)
        """
        B, T, C, H, W = video_batch.shape
        assert T == 4, "This specific TRN implementation is hardcoded for exactly 4 frames."

        frames = video_batch.reshape(B * T, C, H, W)

        # Extract features -> (B*T, 512)
        feats = self.backbone(frames)
        feats = torch.flatten(feats, start_dim=1)
        
        # Reshape to -> (B, 4, 512)
        feats = feats.view(B, T, -1) 

        # ==========================================
        # MULTI-SCALE TEMPORAL RELATIONS
        # ==========================================
        
        # 1. Compute 2-Frame Relations (6 pairs)
        pairs = [(0,1), (1,2), (2,3), (0,2), (1,3), (0,3)]
        out_2 = 0
        for i, j in pairs:
            # Glue two frames together -> (B, 1024)
            cat_feats = torch.cat([feats[:, i, :], feats[:, j, :]], dim=1)
            out_2 += self.relation_2(cat_feats)
            
        # 2. Compute 3-Frame Relations (4 triplets)
        triplets = [(0,1,2), (1,2,3), (0,1,3), (0,2,3)]
        out_3 = 0
        for i, j, k in triplets:
            # Glue three frames together -> (B, 1536)
            cat_feats = torch.cat([feats[:, i, :], feats[:, j, :], feats[:, k, :]], dim=1)
            out_3 += self.relation_3(cat_feats)
            
        # 3. Compute 4-Frame Relation (1 quad)
        # Glue all four frames together -> (B, 2048)
        cat_feats_4 = torch.cat([feats[:, 0, :], feats[:, 1, :], feats[:, 2, :], feats[:, 3, :]], dim=1)
        out_4 = self.relation_4(cat_feats_4)

        # 4. Final Prediction: Sum all relational observations
        final_logits = out_2 + out_3 + out_4
        
        return final_logits