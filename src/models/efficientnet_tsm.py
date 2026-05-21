"""
EfficientNet-B0/B1/B3 & V2-S + TSM (+ optional TDM/TCN)

Updates:
  1. Added support for EfficientNetV2-S.
  2. Implemented dynamic TSM injection to handle both standard MBConv 
     and the new FusedMBConv layers found in V2 architectures.
"""

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.ops import Conv2dNormActivation
from torchvision.models.efficientnet import MBConv, FusedMBConv
from utils import two_stage_trainer

# ── Stage channel map ────────────────────────────────────────────────────────
_EFFICIENTNET_CHANNELS = {0: 32, 1: 16, 2: 24, 3: 40, 4: 80, 5: 112, 6: 192, 7: 320}

# ── Temporal Modules (TSM / TDM / TCN) ───────────────────────────────────────

class TemporalShift(nn.Module):
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
        out[:, 1:,  :fold]          = x[:, :-1, :fold]           
        out[:, :-1, fold:2 * fold]  = x[:, 1:,  fold:2 * fold]   
        out[:, 0,   :fold]          = 0                           
        out[:, -1,  fold:2 * fold]  = 0                           

        return out.view(bt, c, h, w)

class TemporalDifference(nn.Module):
    def __init__(self, n_segment: int) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.alpha     = nn.Parameter(torch.zeros(1)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, c, h, w = x.shape
        b = bt // self.n_segment
        x_r = x.view(b, self.n_segment, c, h, w)

        diff          = torch.zeros_like(x_r)
        diff[:, 1:]   = x_r[:, 1:] - x_r[:, :-1]   

        return x + self.alpha * diff.view(bt, c, h, w)

class IntermediateTCN(nn.Module):
    def __init__(self, channels: int, n_segment: int, kernel_size: int = 3, num_layers: int = 2) -> None:
        super().__init__()
        self.n_segment = n_segment
        padding        = (kernel_size - 1) // 2 
 
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers += [
                nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity(),
            ]
        self.tcn   = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.zeros(1))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, c, h, w = x.shape
        b = bt // self.n_segment
        gap = x.view(b, self.n_segment, c, h, w).mean(dim=[3, 4])  
        tcn_out = self.tcn(gap.permute(0, 2, 1)).permute(0, 2, 1)          
        tcn_out = tcn_out.reshape(bt, c, 1, 1)      
        return x + self.alpha * tcn_out


# ── Injection helpers ────────────────────────────────────────────────────────

def _inject_tsm_safely(block_module: nn.Module, n_segment: int, fold_div: int) -> None:
    """
    Dynamically scans the block to inject TSM before the spatial convolution.
    Handles both standard MBConv and V2's FusedMBConv.
    """
    block = block_module.block  # nn.Sequential
    target_idx = -1

    if isinstance(block_module, FusedMBConv):
        # In FusedMBConv, the spatial convolution (3x3) is ALWAYS the first layer
        target_idx = 0
    elif isinstance(block_module, MBConv):
        # In standard MBConv, we scan for the depthwise convolution (groups > 1)
        for i, layer in enumerate(block):
            if isinstance(layer, Conv2dNormActivation):
                conv = layer[0]
                if isinstance(conv, nn.Conv2d) and conv.groups > 1:
                    target_idx = i
                    break

    # Rebuild block with TemporalShift inserted safely
    if target_idx != -1:
        layers = []
        for i, layer in enumerate(block):
            if i == target_idx:
                layers.append(TemporalShift(n_segment, fold_div))
            layers.append(layer)
        block_module.block = nn.Sequential(*layers)


def _inject_tdm_safely(block_module: nn.Module, n_segment: int) -> None:
    """Wraps either MBConv or FusedMBConv with TDM safely."""
    original_block = block_module.block

    class BlockWithTDM(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = original_block
            self.tdm   = TemporalDifference(n_segment)

        def forward(self, x):
            return self.tdm(self.block(x))

    block_module.block = BlockWithTDM()


# ── Main model ───────────────────────────────────────────────────────────────

@two_stage_trainer
class EfficientNet_TSM(nn.Module):
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

        in_channels = int(model_cfg.get("in_channels", 3))
        fold_div = int(model_cfg.get("fold_div", 8))
        use_tdm  = bool(model_cfg.get("use_tdm", False))
        use_tcn          = bool(model_cfg.get("use_tcn", False))
        tcn_insert_after = int(model_cfg.get("tcn_insert_after", 5))
        tcn_kernel_size  = int(model_cfg.get("tcn_kernel_size", 3))
        tcn_num_layers   = int(model_cfg.get("tcn_num_layers", 2))
        stochastic_depth = float(model_cfg.get("stochastic_depth", 0.2))
        
        # Architecture toggles
        use_b1   = bool(model_cfg.get("efficientnet_b1", False))
        use_b3   = bool(model_cfg.get("efficientnet_b3", False))
        use_v2_s = bool(model_cfg.get("efficientnet_v2_s", False))

        # ── Backbone ────────────────────────────────────────────────────────
        if use_v2_s:
            weights = tv_models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.efficientnet_v2_s(weights=weights)
        elif use_b1:
            weights = tv_models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.efficientnet_b1(weights=weights)
        elif use_b3:
            weights = tv_models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.efficientnet_b3(weights=weights)
        else:
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.efficientnet_b0(weights=weights,
                                                      stochastic_depth_prob=stochastic_depth)

        # ── Stem Surgery (Dynamic Input Channels) ─────────────────────────
        if in_channels != 3:
            # The stem conv is always the first layer of the first Conv2dNormActivation block
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
        
        # ── TSM + optional TDM injection ────────────────────────────────────
        for feat_idx, stage in enumerate(self.backbone.features):
            if not isinstance(stage, nn.Sequential):
                continue
            deep = feat_idx >= 5
            apply_tsm = feat_idx >= model_cfg.get("tsm_insert_after", 4)
            for sub_idx, block in enumerate(stage):
                # V2 compatibility: Check for both MBConv and FusedMBConv
                if not isinstance(block, (MBConv, FusedMBConv)):
                    continue
                
                if model_cfg.get("new_tsm", False) and apply_tsm:
                    _inject_tsm_safely(block, num_frames, fold_div)
                else:
                    stage[sub_idx] = nn.Sequential(
                        TemporalShift(num_frames, fold_div),
                        block,
                    )
                
                if use_tdm and deep:
                    _inject_tdm_safely(block, num_frames)

        # ── Intermediate TCN ─────────────────────────────────────────────────
        if use_tcn:
            tcn_channels          = _EFFICIENTNET_CHANNELS.get(tcn_insert_after, 256) # Fallback for V2
            self.intermediate_tcn = IntermediateTCN(
                channels    = tcn_channels,
                n_segment   = num_frames,
                kernel_size = tcn_kernel_size,
                num_layers  = tcn_num_layers,
            )
            self.tcn_insert_after = tcn_insert_after
        else:
            self.intermediate_tcn = None
 
        # ── Classifier head ──────────────────────────────────────────────────
        in_features = self.backbone.classifier[1].in_features
        if model_cfg.get("new_tsm", False):
            self.backbone.classifier = nn.Identity()   
        
        if model_cfg.get("mlp_head", False):
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(in_features // 2, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)                                    

        for idx, stage in enumerate(self.backbone.features):
            x = stage(x)
            if self.intermediate_tcn is not None and idx == getattr(self, 'tcn_insert_after', -1):
                x = self.intermediate_tcn(x)    

        feats = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        feats = feats.flatten(1)                                       
        feats = feats.view(b, t, -1).mean(dim=1)                      

        return self.head(feats)