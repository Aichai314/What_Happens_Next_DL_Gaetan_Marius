import torch
import torch.nn as nn
import timm
from omegaconf import DictConfig

from utils import two_stage_trainer

class TemporalShift(nn.Module):
    # Temporal Shift Module (TSM)
    # Shifts part of the channels along the temporal dimension to exchange 
    # information with neighboring frames without extra parameters.
    def __init__(self, net: nn.Module, num_frames: int, n_div: int = 8) -> None:
        super().__init__()
        self.net = net
        self.num_frames = num_frames
        self.fold_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shift(x, self.num_frames, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x: torch.Tensor, num_frames: int, fold_div: int = 8) -> torch.Tensor:
        # x shape: (B*T, C, H, W)
        bt, c, h, w = x.size()
        batch_size = bt // num_frames
        
        # Reshape to explicitly expose the temporal dimension
        x = x.view(batch_size, num_frames, c, h, w)

        out = torch.zeros_like(x)
        fold = c // fold_div

        # Shift left (past frames)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        
        # Shift right (future frames)
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        
        # Keep the rest of the channels intact
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        # Flatten back to (B*T, C, H, W) for the 2D CNN
        return out.view(bt, c, h, w)


def _replace_first_conv(module: nn.Module, num_frames: int, n_div: int) -> bool:
    # Recursively searches for the first nn.Conv2d in a block and wraps it in TemporalShift.
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Wrap the first convolution found
            setattr(module, name, TemporalShift(child, num_frames=num_frames, n_div=n_div))
            return True
        elif len(list(child.children())) > 0:
            # Continue searching down the tree
            if _replace_first_conv(child, num_frames, n_div):
                return True
    return False


def inject_tsm_into_efficientformer(model: nn.Module, num_frames: int, n_div: int = 8) -> nn.Module:
    # Injects TSM into the early convolutional stages of a timm EfficientFormer.
    # EfficientFormer uses Meta3D blocks (Transformer) deep down, and Conv blocks early.
    print(f"--> Injecting TSM into EfficientFormer early convolutional stages...")
    
    # EfficientFormer in timm is usually structured with a `stages` module list
    if hasattr(model, 'stages'):
        # We only want to inject TSM into the early purely convolutional stages.
        # EfficientFormer typically has 4 stages. Stage 1, 2, and 3 are Conv. Stage 4 is Attention.
        for i in range(min(3, len(model.stages))): 
            stage = model.stages[i]
            
            if hasattr(stage, 'blocks'):
                for block_idx, block in enumerate(stage.blocks):
                    success = _replace_first_conv(block, num_frames, n_div)
                    if success:
                        print(f"    - TSM injected into stage {i}, block {block_idx}")
    return model


class TemporalAttentionHead(nn.Module):
    # Multi-Head Attention orchestrator for temporal sequences.
    def __init__(self, in_features: int, num_classes: int, num_heads: int = 8, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        # Max 32 frames fallback for positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 32, in_features)) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_features, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(in_features, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, T, F = x.shape
        x = x + self.pos_embed[:, :T, :]
        x = self.transformer(x)      # Interact frames temporally
        x = x.mean(dim=1)            # Global average pooling over time
        x = self.dropout(x)
        return self.fc(x)

@two_stage_trainer
class EfficientFormer_Hybrid(nn.Module):
    # EfficientFormer (Conv+Attention Spatial) + TSM (Early Temporal) + Custom Head (Late Temporal)
    def __init__(self,
        model_cfg: DictConfig,
        num_frames: int = 4,
        num_classes: int = 33,
        pretrained: bool = False
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.head_type = model_cfg.get('head_type', 'attention').lower()
        self.dropout_rate = model_cfg.get('dropout', 0.3)
        
        # 1. Load EfficientFormer Backbone
        # Note: 'efficientformer_l1' is a common lightweight variant. 
        # Update your yaml to pass 'efficientformer_l3' if you want a larger model.
        backbone_name = model_cfg.get('backbone', 'efficientformer_l1')
        
        print(f"--> Building Backbone: {backbone_name}")
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained,
            drop_path_rate=model_cfg.get('stochastic_depth', 0.1),
            num_classes=0
        )
        
        # Determine feature dimension 
        self.in_features = self.backbone.num_features
        
        # 2. Inject TSM into the Convolutional Stages
        if model_cfg.get('use_tsm', True):
            self.backbone = inject_tsm_into_efficientformer(
                self.backbone, 
                num_frames=self.num_frames, 
                n_div=model_cfg.get('fold_div', 8)
            )
            
        # 3. Construct the specified Classification Head
        print(f"--> Building Temporal Head: {self.head_type.upper()}")
        if self.head_type == 'linear':
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.in_features, self.num_classes)
            )
            
        elif self.head_type == 'mlp':
            hidden_dim = self.in_features // 2
            self.head = nn.Sequential(
                nn.Linear(self.in_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, self.num_classes)
            )
            
        elif self.head_type == 'lstm':
            hidden_dim = self.in_features // 2
            self.lstm = nn.LSTM(
                input_size=self.in_features, 
                hidden_size=hidden_dim, 
                num_layers=1, 
                batch_first=True,
                bidirectional=True  
            )
            self.head = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim * 2, self.num_classes)
            )
            
        elif self.head_type == 'attention':
            self.head = TemporalAttentionHead(
                in_features=self.in_features,
                num_classes=self.num_classes,
                dropout=self.dropout_rate
            )
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}. Choose from 'linear', 'mlp', 'lstm', 'attention'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        
        # Fold temporal dimension into batch dimension for spatial backbone
        x = x.view(B * T, C, H, W)
        
        # Extract features: returns (B*T, F)
        features = self.backbone(x)
        
        # Unfold temporal dimension: (B, T, F)
        features = features.view(B, T, self.in_features)
        
        # Route to the appropriate head
        if self.head_type in ['linear', 'mlp']:
            # Global Average Pooling over the temporal dimension
            x = features.mean(dim=1)
            out = self.head(x)
            
        elif self.head_type == 'lstm':
            lstm_out, (hn, cn) = self.lstm(features)
            x = lstm_out[:, -1, :]
            out = self.head(x)
            
        elif self.head_type == 'attention':
            out = self.head(features)
            
        return out
