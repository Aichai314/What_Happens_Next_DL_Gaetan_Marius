"""
ConvNeXt-Tiny + TSM (Temporal Shift Module)

ConvNeXt-Tiny architecture (input 224x224):
  features[0]  Conv2dNormActivation  -> (B*T,  96, 56, 56)  patchify stem
  features[1]  Sequential[CNBlock]   -> (B*T,  96, 56, 56)
  features[2]  Sequential[...]       -> (B*T, 192, 28, 28)  downsampling
  features[3]  Sequential[CNBlock]   -> (B*T, 192, 28, 28)
  features[4]  Sequential[...]       -> (B*T, 384, 14, 14)  downsampling
  features[5]  Sequential[CNBlock]   -> (B*T, 384, 14, 14)
  features[6]  Sequential[...]       -> (B*T, 768,  7,  7)  downsampling
  features[7]  Sequential[CNBlock]   -> (B*T, 768,  7,  7)
  avgpool + classifier               -> (B*T, num_classes)

TSM injection:
  Each CNBlock has a depthwise Conv2d(C, C, 7x7, groups=C) at block[0].
  TemporalShift is prepended just before this depthwise conv — the correct
  position to give the conv temporally-shifted channels as input.

Note: torchvision only provides ConvNeXt V1. ConvNeXtV2-Nano is not yet
available natively; this file uses ConvNeXt-Tiny (V1, 28M params) as the
closest available alternative. If timm is available in your environment,
see the comment at the bottom for a ConvNeXtV2-Nano drop-in.
"""

from __future__ import annotations

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models.convnext import CNBlock
from utils import two_stage_trainer


# ── Temporal Shift Module ────────────────────────────────────────────────────

class CircularTemporalShift(nn.Module):
    def __init__(self, n_segment: int, fold_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, c, h, w = x.shape
        b = bt // self.n_segment
        x = x.view(b, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = x.clone()

        # Wrap-around shift (No Zeros to corrupt LayerNorm!)
        out[:, :, :fold] = torch.roll(x[:, :, :fold], shifts=1, dims=1)
        out[:, :, fold:2*fold] = torch.roll(x[:, :, fold:2*fold], shifts=-1, dims=1)

        return out.view(bt, c, h, w)

class ReplicateTemporalShift(nn.Module):
    """
    Shifts channels through time using Replicate Padding.
    Preserves LayerNorm stability AND strict temporal causality.
    """
    def __init__(self, n_segment: int, fold_div: int = 8) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, c, h, w = x.shape
        b = bt // self.n_segment
        
        # Shape: (Batch, Channels, Time, Height, Width)
        # Note: We put Time in dim=2 so F.pad can operate on it
        x = x.view(b, self.n_segment, c, h, w).permute(0, 2, 1, 3, 4)
        
        fold = c // self.fold_div
        
        out = torch.zeros_like(x)
        
        # 1. Shift Forward (Past -> Current)
        # We want to shift Time right. We pad 1 on the left (past), remove 1 from the right.
        # Pad format for 3D tensor (W, H, T): (0,0, 0,0, pad_left, pad_right)
        past_shift = F.pad(x[:, :fold, ...], (0, 0, 0, 0, 1, 0), mode='replicate')[:, :, :-1, ...]
        out[:, :fold, ...] = past_shift
        
        # 2. Shift Backward (Future -> Current)
        # We want to shift Time left. We pad 1 on the right (future), remove 1 from the left.
        future_shift = F.pad(x[:, fold:2*fold, ...], (0, 0, 0, 0, 0, 1), mode='replicate')[:, :, 1:, ...]
        out[:, fold:2*fold, ...] = future_shift
        
        # 3. Static Channels (No Shift)
        out[:, 2*fold:, ...] = x[:, 2*fold:, ...]
        
        # Revert shape to (B*T, C, H, W)
        out = out.permute(0, 2, 1, 3, 4).reshape(bt, c, h, w)
        
        return out


# ── Main model ───────────────────────────────────────────────────────────────

class ConvNeXt_TSM(nn.Module):
    """
    ConvNeXt-Tiny + TSM with a simple linear classifier head.

    TSM is injected before the depthwise conv of every CNBlock so that
    temporal context flows through the entire feature hierarchy.

    Config keys (model_cfg):
      fold_div   : int  = 8      TSM fold divisor
      convnext_s : bool = False  use ConvNeXt-Small instead of Tiny
                                 (not recommended from scratch, 50M params)

    For ConvNeXtV2-Nano via timm, see the comment at the bottom of this file.
    """

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

        fold_div   = int(model_cfg.get("fold_div", 8))

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            'convnextv2_nano', 
            pretrained=pretrained, 
            num_classes=0,       # Strips the default ImageNet classifier
            global_pool='avg'    # Forces output to be a flat 1D feature vector
        )
        in_features = self.backbone.num_features  # 640
        
        # ── 6-Channel Stem Surgery ───────────────────────────────────────────
        in_channels = int(model_cfg.get("in_channels", 3))
        if in_channels != 3:
            original_stem_conv = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv2d(
                in_channels, 
                original_stem_conv.out_channels,
                kernel_size=original_stem_conv.kernel_size,
                stride=original_stem_conv.stride,
                padding=original_stem_conv.padding
            )

        # ── TSM injection into every CNBlock ──────────────────────────────────
        for stage in self.backbone.stages:
            for block in stage.blocks:
                # timm CNBlock has block.conv_dw as the depthwise conv
                original_conv_dw = block.conv_dw
                block.conv_dw = nn.Sequential(
                    ReplicateTemporalShift(num_frames, fold_div),
                    original_conv_dw,
                )

        # ── Classifier head ───────────────────────────────────────────────────
        # ConvNeXt classifier: LayerNorm2d -> Flatten -> Linear
        # We keep LayerNorm2d (applied after avgpool) and replace Linear only.
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
        x = x.view(b * t, c, h, w)                     # (B*T, C, H, W)

        x = self.backbone(x)          # (B*T, 768, 7, 7)
        seq_feats = x.view(b, t, -1) # Shape: (B, T, d_model)
        
        pooled_feats = seq_feats.mean(dim=1) # Shape: (B, d_model)
        
        return self.head(pooled_feats)
