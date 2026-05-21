"""
EfficientNet-B0 + TSM + TDN (Temporal Difference Network)

Adapts the TDN paper (Chi et al. 2020, "Temporal Difference Networks for
Efficient Action Recognition") to EfficientNet-B0 as backbone.

Original paper: https://arxiv.org/abs/2012.10071

TDN key idea:
  Motion information is captured at TWO temporal scales and injected into
  the backbone EARLY (after the stem) so it influences the entire feature
  hierarchy — unlike late fusion which has been empirically shown to add
  little value on this dataset.

  - Short-range branch : frame differences F(t+1) - F(t)  [T-1 maps]
                         captures fine-grained local motion
  - Long-range branch  : clip-level difference F(T-1) - F(0)
                         captures the global action trajectory

  Both branches are processed by a lightweight TDiffModule (depthwise +
  pointwise conv) and fused into the backbone features after the stem via
  a learnable weighted residual.

EfficientNet-B0 stem output at 224px input:
  features[0] -> (B*T, 32, 112, 112)   <- TDN injection point

Adaptation from the paper (ResNet) to EfficientNet-B0:
  - out_channels = 32  (paper uses 64 for ResNet-50)
  - spatial size  = 112x112 (paper uses 56x56 for ResNet with stride-4 stem)
  - everything else is identical in spirit
"""

from __future__ import annotations

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models.efficientnet import MBConv
from utils import two_stage_trainer


# ── Temporal Shift Module ────────────────────────────────────────────────────

class TemporalShift(nn.Module):
    """
    Shifts 1/fold_div channels forward and 1/fold_div channels backward in
    time. Injected just before the depthwise conv inside each MBConv block.
    """
    def __init__(self, n_segment: int, fold_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.fold_div  = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, c, h, w = x.shape
        b    = bt // self.n_segment
        x    = x.view(b, self.n_segment, c, h, w)
        fold = c // self.fold_div

        out = x.clone()
        out[:, 1:,  :fold]         = x[:, :-1, :fold]
        out[:, :-1, fold:2 * fold] = x[:, 1:,  fold:2 * fold]
        out[:, 0,   :fold]         = 0
        out[:, -1,  fold:2 * fold] = 0

        return out.view(bt, c, h, w)


# ── TDN modules ──────────────────────────────────────────────────────────────

class TDiffModule(nn.Module):
    """
    Lightweight CNN that refines a temporal difference map into motion
    features of the same channel dimension as the backbone stem output.

    Uses depthwise + pointwise (separable) conv to stay parameter-efficient.
    BatchNorm + ReLU after each conv for training stability from scratch.

    Args:
        in_channels  : number of input channels (3*(T-1) for local,
                       3 for global)
        out_channels : must match backbone stem output channels (32 for B0)
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Depthwise spatial conv — learns local motion patterns
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3,
                padding=1, groups=in_channels, bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise — projects to backbone channel dim
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TDNFusionModule(nn.Module):
    """
    Full TDN motion branch: computes short-range and long-range temporal
    differences, refines each with a TDiffModule, and fuses them into a
    single motion feature map that is added to backbone stem features.

    Short-range (local):
      Concatenates (T-1) consecutive frame differences along channel dim.
      Input shape  : (B, T, C, H, W)  with C=3 (RGB)
      Output shape : (B, stem_channels, H', W')

    Long-range (global):
      Single difference between last and first frame of the clip.
      Input shape  : (B, T, C, H, W)
      Output shape : (B, stem_channels, H', W')

    Fusion:
      stem_out + alpha * local_motion + beta * global_motion

      alpha and beta are learnable scalars initialised at 0.5 so the
      network starts with balanced contributions and can adapt freely.

    Args:
        num_frames    : T (number of frames per clip, typically 4)
        rgb_channels  : C (typically 3)
        stem_channels : number of channels output by backbone stem (32 for B0)
        stem_stride   : spatial stride of the backbone stem (2 for B0,
                        so H' = H // 2)
    """
    def __init__(
        self,
        num_frames:    int,
        rgb_channels:  int = 3,
        stem_channels: int = 32,
        stem_stride:   int = 2,
    ) -> None:
        super().__init__()
        self.num_frames   = num_frames
        self.stem_stride  = stem_stride

        # Short-range: (T-1) difference maps concatenated -> C*(T-1) channels
        local_in = rgb_channels * (num_frames - 1)
        self.tdiff_local  = TDiffModule(local_in,    stem_channels)

        # Long-range: single difference map -> C channels
        self.tdiff_global = TDiffModule(rgb_channels, stem_channels)

        # Learnable fusion weights — init at 0.5 (balanced)
        self.alpha = nn.Parameter(torch.full((1,), 0.5))
        self.beta  = nn.Parameter(torch.full((1,), 0.5))

    def forward(
        self,
        frames: torch.Tensor,   # (B, T, C, H, W) — original RGB frames
        stem_out: torch.Tensor, # (B*T, stem_C, H', W') — after backbone stem
    ) -> torch.Tensor:
        """
        Returns stem_out enriched with motion features.
        Output shape : identical to stem_out — (B*T, stem_C, H', W')
        """
        b, t, c, h, w = frames.shape

        # ── Short-range: consecutive frame differences ────────────────────
        # diffs[i] = frames[:, i+1] - frames[:, i]  shape (B, C, H, W)
        local_diffs = torch.cat(
            [frames[:, i + 1] - frames[:, i] for i in range(t - 1)],
            dim=1,
        )  # (B, C*(T-1), H, W)

        # Downsample to match stem spatial resolution
        if self.stem_stride > 1:
            local_diffs = F.avg_pool2d(local_diffs, self.stem_stride,
                                       stride=self.stem_stride)

        local_motion = self.tdiff_local(local_diffs)   # (B, stem_C, H', W')

        # Broadcast across frames: each frame gets the same motion context
        local_motion = local_motion.unsqueeze(1).expand(-1, t, -1, -1, -1)
        local_motion = local_motion.reshape(b * t, -1,
                                            local_motion.shape[-2],
                                            local_motion.shape[-1])

        # ── Long-range: first vs last frame ──────────────────────────────
        global_diff = frames[:, -1] - frames[:, 0]    # (B, C, H, W)

        if self.stem_stride > 1:
            global_diff = F.avg_pool2d(global_diff, self.stem_stride,
                                       stride=self.stem_stride)

        global_motion = self.tdiff_global(global_diff) # (B, stem_C, H', W')

        global_motion = global_motion.unsqueeze(1).expand(-1, t, -1, -1, -1)
        global_motion = global_motion.reshape(b * t, -1,
                                              global_motion.shape[-2],
                                              global_motion.shape[-1])

        # ── Fusion: weighted residual onto stem features ──────────────────
        return stem_out + self.alpha * local_motion + self.beta * global_motion


# ── TSM injection ────────────────────────────────────────────────────────────

def _inject_tsm_into_mbconv(mbconv: MBConv, n_segment: int, fold_div: int) -> None:
    """Injects TemporalShift before depthwise conv inside an MBConv block."""
    block  = mbconv.block
    dw_idx = len(block) - 2
    layers = []
    for i, layer in enumerate(block):
        if i == dw_idx:
            layers.append(TemporalShift(n_segment, fold_div))
        layers.append(layer)
    mbconv.block = nn.Sequential(*layers)


# ── Main model ───────────────────────────────────────────────────────────────

@two_stage_trainer
class EfficientNet_TDN(nn.Module):
    """
    EfficientNet-B0 with TSM + TDN (Temporal Difference Network).

    Motion features (TDN) are injected after the backbone stem so they
    propagate through the entire feature hierarchy — this is what makes
    TDN fundamentally different from late fusion approaches.

    TSM further enriches every MBConv block with shifted temporal context,
    complementing TDN's explicit motion signal.

    Config keys (model_cfg):
      fold_div : int = 8   TSM fold divisor
    """

    # EfficientNet-B0 stem parameters (verified empirically)
    _STEM_CHANNELS = 32
    _STEM_STRIDE   = 2   # stem halves spatial resolution: 224 -> 112

    def __init__(
        self,
        model_cfg:   DictConfig,
        num_classes: int   = 33,
        num_frames:  int   = 4,
        pretrained:  bool  = False,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        fold_div = int(model_cfg.get("fold_div", 8))

        # ── Backbone ─────────────────────────────────────────────────────────
        weights       = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = tv_models.efficientnet_b0(weights=weights)

        # ── TSM injection into every MBConv ───────────────────────────────────
        for stage in self.backbone.features:
            if not isinstance(stage, nn.Sequential):
                continue
            for block in stage:
                if isinstance(block, MBConv):
                    _inject_tsm_into_mbconv(block, num_frames, fold_div)

        # ── TDN fusion module ─────────────────────────────────────────────────
        self.tdn = TDNFusionModule(
            num_frames    = num_frames,
            rgb_channels  = 3,
            stem_channels = self._STEM_CHANNELS,
            stem_stride   = self._STEM_STRIDE,
        )

        # ── Classifier head ───────────────────────────────────────────────────
        in_features              = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, C, H, W) — RGB frames
        Returns:
            logits : (B, num_classes)
        """
        b, t, c, h, w = x.shape

        # ── Stem (kept in (B, T, ...) space for TDN) ─────────────────────────
        x_flat   = x.view(b * t, c, h, w)
        stem_out = self.backbone.features[0](x_flat)   # (B*T, 32, 112, 112)

        # ── TDN: inject motion features after stem ────────────────────────────
        stem_out = self.tdn(x, stem_out)               # (B*T, 32, 112, 112)

        # ── Rest of backbone (features[1:]) with TSM ─────────────────────────
        feat = stem_out
        for layer in self.backbone.features[1:]:
            feat = layer(feat)                         # (B*T, 1280, 7, 7)

        # ── Spatial pooling + temporal consensus ──────────────────────────────
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)  # (B*T, 1280)
        feat = feat.view(b, t, -1).mean(dim=1)                 # (B, 1280)

        return self.head(feat)
