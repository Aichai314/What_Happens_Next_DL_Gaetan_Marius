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
from torchvision.models.efficientnet import MBConv, FusedMBConv
from utils import two_stage_trainer
from models.efficientnet_tsm import _inject_tsm_safely, _inject_tdm_safely

# ── Stage channel map ────────────────────────────────────────────────────────
 
# B0 and B1 share the same channel widths — B1 differs only in block depth
_EFFICIENTNET_CHANNELS = {0: 32, 1: 16, 2: 24, 3: 40, 4: 80, 5: 112, 6: 192, 7: 320}

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

class IntermediateTCN(nn.Module):
    """
    Early temporal fusion module inserted between two EfficientNet stages.
 
    Operates as a spatial residual so downstream stages and TSM are
    completely unaffected:
 
      1. Spatial GAP       : (B*T, C, H, W) -> (B, T, C)
      2. Depthwise TCN     : (B, T, C) -> (B, T, C)   [same dim, minimal params]
      3. Spatial broadcast : (B, T, C) -> (B*T, C, 1, 1)
      4. Residual add      : x + alpha * tcn_out  [preserves (B*T, C, H, W)]
 
    Depthwise Conv1d (groups=C) is used because:
      - T=4 is very short; a dense Conv1d over C channels would overfit easily.
      - Consistent with EfficientNet's own depthwise philosophy.
 
    alpha initialised at 0: module starts as identity and only activates
    if the TCN signal is genuinely useful.
    """
    def __init__(
        self,
        channels:    int,
        n_segment:   int,
        kernel_size: int = 3,
        num_layers:  int = 2,
    ) -> None:
        super().__init__()
        self.n_segment = n_segment
        padding        = (kernel_size - 1) // 2  # 'same' padding
 
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers += [
                nn.Conv1d(
                    channels, channels, kernel_size,
                    padding=padding, groups=channels,  # depthwise temporal conv
                ),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity(),
            ]
        self.tcn   = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.zeros(1))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, c, h, w = x.shape
        b = bt // self.n_segment
 
        # 1. Spatial GAP — one vector per frame
        gap = x.view(b, self.n_segment, c, h, w).mean(dim=[3, 4])  # (B, T, C)
 
        # 2. Depthwise TCN over temporal dimension
        tcn_out = self.tcn(gap.permute(0, 2, 1))   # (B, C, T)
        tcn_out = tcn_out.permute(0, 2, 1)          # (B, T, C)
 
        # 3. Broadcast to spatial resolution
        tcn_out = tcn_out.reshape(bt, c, 1, 1)      # (B*T, C, 1, 1)
 
        # 4. Learnable residual
        return x + self.alpha * tcn_out


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

# ─────────────────────────────────────────────────────────────────────────────
# Temporal SE
# ─────────────────────────────────────────────────────────────────────────────

class TemporalSE(nn.Module):
    """
    Lightweight temporal squeeze excitation.

    Input:
        x : (B, T, C)

    Learns which temporal positions are important.
    """

    def __init__(self, num_frames: int, reduction: int = 2):
        super().__init__()

        hidden_dim = max(1, num_frames // reduction)

        self.fc = nn.Sequential(
            nn.Linear(num_frames, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_frames),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, C)

        # Temporal squeeze
        w = x.mean(dim=-1)   # (B, T)

        # Temporal excitation
        w = self.fc(w)       # (B, T)
        w = w.unsqueeze(-1)  # (B, T, 1)

        return x * w


# ─────────────────────────────────────────────────────────────────────────────
# Temporal head
# ─────────────────────────────────────────────────────────────────────────────

class TemporalHead(nn.Module):
    """
    Lightweight temporal head:

        RGB features
        +
        difference features
        -> temporal Conv1D
        -> temporal SE
        -> classifier
    """

    def __init__(
        self,
        in_channels: int,
        num_frames: int,
        num_classes: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_frames = num_frames

        # RGB + diff concatenation
        fusion_dim = in_channels * 2

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                fusion_dim,
                fusion_dim,
                kernel_size=3,
                padding=1,
                groups=fusion_dim,  # depthwise temporal conv
            ),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
        )

        self.temporal_se = TemporalSE(num_frames=num_frames)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats : (B, T, C)

        Returns:
            logits : (B, num_classes)
        """

        b, t, c = feats.shape

        # ── Difference branch ────────────────────────────────────────────────

        diff = feats[:, 1:] - feats[:, :-1]   # (B, T-1, C)

        # Pad first timestep with zeros to recover T frames
        zero = torch.zeros_like(diff[:, :1])
        diff = torch.cat([zero, diff], dim=1)  # (B, T, C)

        # ── RGB + diff fusion ───────────────────────────────────────────────

        x = torch.cat([feats, diff], dim=-1)   # (B, T, 2C)

        # ── Temporal Conv1D ─────────────────────────────────────────────────

        x = x.permute(0, 2, 1)                 # (B, 2C, T)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)                 # (B, T, 2C)

        # ── Temporal SE ─────────────────────────────────────────────────────

        x = self.temporal_se(x)

        # ── Temporal pooling ────────────────────────────────────────────────

        x = x.mean(dim=1)                      # (B, 2C)

        return self.classifier(x)

# =========================================================
# THE NEW TEMPORAL ATTENTION HEAD
# =========================================================
class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_frames: int, 
                 num_heads: int = 8, dropout: float = 0.3, 
                 attention_dropout: float = 0.1, num_classes: int = 33):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_frames, embed_dim) * 0.02
        )
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, 
            dropout=attention_dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),  # norme finale avant classification
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out
        
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        
        return self.head(x.mean(dim=1))

# ── Main model ───────────────────────────────────────────────────────────────

class EfficientNetTemporalHead(nn.Module):
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

        in_channels = int(model_cfg.get("in_channels", 3))
        fold_div = int(model_cfg.get("fold_div", 8))
        use_tdm  = bool(model_cfg.get("use_tdm", False))
        use_tcn          = bool(model_cfg.get("use_tcn", False))
        tcn_insert_after = int(model_cfg.get("tcn_insert_after", 5))
        tcn_kernel_size  = int(model_cfg.get("tcn_kernel_size", 3))
        tcn_num_layers   = int(model_cfg.get("tcn_num_layers", 2))
        stochastic_depth = float(model_cfg.get("stochastic_depth", 0.2))
        backbone_path    = model_cfg.get("backbone_path", None)
        
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
        
        if backbone_path:
            print(f"--> Loading strictly backbone from: {backbone_path}")
            checkpoint = torch.load(backbone_path, map_location="cpu")
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Extract ONLY the backbone weights
            backbone_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'backbone' in k}
            self.backbone.load_state_dict(backbone_dict, strict=False)
 
        # ── Classifier head ──────────────────────────────────────────────────
        in_features = self.backbone.classifier[1].in_features
        if model_cfg.get("new_tsm", False):
            self.backbone.classifier = nn.Identity() 
        
        # ── Temporal head ───────────────────────────────────────────────────

        if model_cfg.get("attention_head", False):
            print("--> Using Temporal Self-Attention head")
            self.temporal_head = TemporalSelfAttention(
                embed_dim=in_features,
                num_frames=num_frames,
                num_heads=model_cfg.get("attention_heads", 8),
                dropout=dropout,
                attention_dropout=model_cfg.get("attention_dropout", 0.15)
            )
        else:
            self.temporal_head = TemporalHead(
                in_channels=in_features,
                num_frames=num_frames,
                num_classes=num_classes,
                dropout=dropout,
            )
    
    # ────────────────────────────────────────────────────────────────────────
    # Parameter groups
    # ────────────────────────────────────────────────────────────────────────

    def get_param_groups(
        self,
        base_lr: float,
        backbone_lr_factor: float = 0.1,
    ):
        """
        Returns optimizer parameter groups.

        Example:
            backbone lr = base_lr * 0.1
            head lr      = base_lr
        """

        return [
            {
                "params": self.backbone.parameters(),
                "lr": base_lr * backbone_lr_factor,
            },
            {
                "params": self.temporal_head.parameters(),
                "lr": base_lr,
            },
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, C, H, W)  — standard RGB frames
        Returns:
            logits : (B, num_classes)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)                                    

        for idx, stage in enumerate(self.backbone.features):
            x = stage(x)
            if self.intermediate_tcn is not None and idx == getattr(self, 'tcn_insert_after', -1):
                x = self.intermediate_tcn(x)    

        feats = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        feats = feats.flatten(1)                                       # (B*T, 1280)

        feats = feats.view(b, t, -1)                 # (B, T, 1280)

        return self.temporal_head(feats)