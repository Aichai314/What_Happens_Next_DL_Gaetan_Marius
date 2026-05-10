"""
EfficientNet-B0 + TSM (+ optional TDM in deep layers)

Corrections vs. previous version:
  1. TSM is injected INSIDE each MBConv block, just before the depthwise
     conv (block[1][0]), which is the correct location — not wrapping the
     entire MBConv from outside.
  2. TDM (Temporal Difference Module) is optionally injected only in deep
     layers (features[5:]) where features are abstract enough for temporal
     differences to carry meaningful motion signal.
  3. 6-channel input removed (empirically worse than 3-channel RGB).
"""

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models.efficientnet import MBConv
from utils import two_stage_trainer


# ── Temporal Shift Module ────────────────────────────────────────────────────

class TemporalShift(nn.Module):
    """
    Shift 1/fold_div channels forward and 1/fold_div channels backward in time.
    Must be placed just before a depthwise conv inside an MBConv block.
    """
    def __init__(self, n_segment: int, fold_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.fold_div  = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, c, h, w = x.shape
        b = bt // self.n_segment
        x = x.view(b, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = x.clone()
        out[:, 1:,  :fold]          = x[:, :-1, :fold]           # past  → current
        out[:, :-1, fold:2 * fold]  = x[:, 1:,  fold:2 * fold]   # future → current
        out[:, 0,   :fold]          = 0                           # no past  for frame 0
        out[:, -1,  fold:2 * fold]  = 0                           # no future for last frame

        return out.view(bt, c, h, w)


# ── Temporal Difference Module ───────────────────────────────────────────────

class TemporalDifference(nn.Module):
    """
    TDM: adds a scaled residual of (F(t) - F(t-1)) to the current feature map.
    Captures explicit motion signal at the feature level.

    alpha is learnable so the network can suppress TDM if it is not useful
    for a given layer.
    """
    def __init__(self, n_segment: int) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.alpha     = nn.Parameter(torch.zeros(1))  # init at 0 = identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, c, h, w = x.shape
        b = bt // self.n_segment
        x_r = x.view(b, self.n_segment, c, h, w)

        diff          = torch.zeros_like(x_r)
        diff[:, 1:]   = x_r[:, 1:] - x_r[:, :-1]   # F(t) - F(t-1); 0 for first frame

        return x + self.alpha * diff.view(bt, c, h, w)


# ── Injection helpers ────────────────────────────────────────────────────────

def _inject_tsm_into_mbconv(mbconv: MBConv, n_segment: int, fold_div: int) -> None:
    """
    Injects a TemporalShift in-place just before the depthwise conv inside
    an MBConv block.

    EfficientNet-B0 MBConv internal layout (block is a Sequential):
      [0] expand conv  (Conv2dNormActivation, 1x1)   — absent in stage 1 (no expansion)
      [-2] depthwise   (Conv2dNormActivation, 3x3 dw)
      [-1] project     (Conv2dNormActivation, 1x1, no activation)

    We prepend TemporalShift before the depthwise conv, which is always the
    second-to-last element of mbconv.block.
    """
    block = mbconv.block  # nn.Sequential

    # Find depthwise conv index: always second-to-last in EfficientNet-B0
    dw_idx = len(block) - 2

    # Rebuild block with TemporalShift inserted before depthwise conv
    layers = []
    for i, layer in enumerate(block):
        if i == dw_idx:
            layers.append(TemporalShift(n_segment, fold_div))
        layers.append(layer)

    mbconv.block = nn.Sequential(*layers)


def _inject_tdm_into_mbconv(mbconv: MBConv, n_segment: int) -> None:
    """
    Injects a TemporalDifference module in-place at the OUTPUT of an MBConv
    block (after the projection conv, before the residual addition).
    We wrap the whole block so TDM sees the projected features.
    """
    original_block = mbconv.block

    class BlockWithTDM(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = original_block
            self.tdm   = TemporalDifference(n_segment)

        def forward(self, x):
            return self.tdm(self.block(x))

    mbconv.block = BlockWithTDM()


# ── Main model ───────────────────────────────────────────────────────────────

@two_stage_trainer
class EfficientNet_TSM(nn.Module):
    """
    EfficientNet-B0 with:
      - TSM injected before the depthwise conv of every MBConv block
      - TDM optionally injected at the output of deep MBConv blocks (features[5:])
      - Standard 3-channel RGB input (6-channel empirically worse)
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        num_classes: int = 33,
        num_frames:  int = 4,
        pretrained:  bool = False,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        fold_div = int(model_cfg.get("fold_div", 8))
        use_tdm  = bool(model_cfg.get("use_tdm", False))

        # ── Backbone (no pretrained weights per competition rules) ──────────
        weights        = None if not pretrained else tv_models.EfficientNet_B0_Weights.DEFAULT
        self.backbone  = tv_models.efficientnet_b0(weights=weights)

        # ── TSM + optional TDM injection ────────────────────────────────────
        for feat_idx, stage in enumerate(self.backbone.features):
            if not isinstance(stage, nn.Sequential):
                continue
            deep = feat_idx >= 5   # features[5], [6], [7] are deep stages
            for sub_idx, block in enumerate(stage):
                if not isinstance(block, MBConv):
                    continue
                if model_cfg.get("new_tsm", False):
                    # TSM in every MBConv (correct position: before depthwise conv)
                    _inject_tsm_into_mbconv(block, num_frames, fold_div)
                else:
                    # Original TSM implementation: wraps entire MBConv from outside (incorrect)
                    stage[sub_idx] = nn.Sequential(
                        TemporalShift(num_frames, fold_div),
                        block,
                    )
                # TDM only in deep layers and only if enabled
                if use_tdm and deep:
                    _inject_tdm_into_mbconv(block, num_frames)

        # ── Classifier head ─────────────────────────────────────────────────
        in_features    = self.backbone.classifier[1].in_features
        if model_cfg.get("new_tsm", False):
            self.backbone.classifier = nn.Identity()   # remove original head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, C, H, W)  — standard RGB frames
        Returns:
            logits : (B, num_classes)
        """
        b, t, c, h, w = x.shape

        # Fold time into batch dimension for the 2-D CNN
        x = x.view(b * t, c, h, w)                                    # (B*T, C, H, W)

        # Feature extraction (TSM/TDM operate inside backbone.features)
        feats = self.backbone.features(x)                              # (B*T, 1280, H', W')

        # Spatial pooling
        feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
        feats = feats.flatten(1)                                       # (B*T, 1280)

        # Temporal consensus (average over frames)
        feats = feats.view(b, t, -1).mean(dim=1)                      # (B, 1280)

        return self.head(feats)