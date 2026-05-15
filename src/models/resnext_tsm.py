import torch
import torch.nn as nn
import torchvision.models as tv_models

class TemporalShift(nn.Module):
    """Shift 1/fold_div channels forward and 1/fold_div channels backward in time."""
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
        out[:, 1:, :fold] = x[:, :-1, :fold]                      # past -> current
        out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]  # future -> current
        out[:, 0, :fold] = 0
        out[:, -1, fold : 2 * fold] = 0

        return out.view(bt, c, h, w)

class ResNeXt50_TSM(nn.Module):
    """
    ResNeXt-50 32x4d + TSM.
    Optimized for 6-channel input (RGB + Difference) and from-scratch training.
    """
    def __init__(
        self, 
        num_classes: int = 33, 
        num_frames: int = 4, 
        in_channels: int = 6, 
        fold_div: int = 8,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        
        # 1. Load Backbone (Track A: weights=None)
        self.backbone = tv_models.resnext50_32x4d(weights=None)
        
        # 2. Stem Surgery for 6-Channel Early Fusion
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # 3. TSM Injection into Bottleneck blocks
        # We inject right before the 3x3 grouped convolution (conv2)
        for stage in [self.backbone.layer1, self.backbone.layer2, 
                      self.backbone.layer3, self.backbone.layer4]:
            for block in stage:
                block.conv2 = nn.Sequential(
                    TemporalShift(n_segment=num_frames, fold_div=fold_div),
                    block.conv2
                )
        
        # 4. Head Replacement
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        
        # Fold time into batch for 2D processing
        x = x.view(b * t, c, h, w)
        
        # Backbone processing (includes global avg pooling)
        x = self.backbone(x) # (B*T, 2048)
        
        # Temporal Consensus (Average pooling across time)
        x = x.view(b, t, -1).mean(dim=1) # (B, 2048)
        
        return self.head(x)