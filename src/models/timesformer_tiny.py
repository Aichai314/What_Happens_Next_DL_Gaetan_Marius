"""
TimeSformer-Tiny (Divided Space-Time Attention)
Optimized for "From Scratch" training on small datasets.

Instead of computing full 3D attention (which scales quadratically and instantly out-of-memories),
this computes:
1. Temporal Attention: Each patch looks at itself across the 4 frames.
2. Spatial Attention: Each patch looks at all other patches in the SAME frame.

Input shape: (B, T, C, H, W)
Output shape: (B, num_classes)
"""

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PatchEmbed(nn.Module):
    """Splits images into patches and embeds them to d_model."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_model=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # A single Conv2D layer with stride=patch_size is the exact mathematical 
        # equivalent of non-overlapping patch extraction + linear projection!
        self.proj = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B*T, C, H, W)
        x = self.proj(x)  # (B*T, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B*T, num_patches, d_model)
        return x

class DividedAttentionBlock(nn.Module):
    """The core TimeSformer Block: Temporal Attention -> Spatial Attention -> MLP"""
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        # 1. Temporal Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # 2. Spatial Attention
        self.norm2 = nn.LayerNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # 3. MLP
        self.norm3 = nn.LayerNorm(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, B, T, N):
        # x shape: (B, T, N, d_model)
        
        # ----------------------------------------------------
        # 1. TEMPORAL ATTENTION
        # ----------------------------------------------------
        res = x
        x = self.norm1(x)
        # Reshape so time is the sequence: (B*N, T, d_model)
        # Every spatial patch forms its own batch to look across time
        x = rearrange(x, 'b t n d -> (b n) t d')
        
        x, _ = self.temporal_attn(x, x, x)
        
        # Return to 4D and add residual
        x = rearrange(x, '(b n) t d -> b t n d', b=B, n=N)
        x = res + x
        
        # ----------------------------------------------------
        # 2. SPATIAL ATTENTION
        # ----------------------------------------------------
        res = x
        x = self.norm2(x)
        # Reshape so space is the sequence: (B*T, N, d_model)
        # Every frame forms its own batch to look across space
        x = rearrange(x, 'b t n d -> (b t) n d')
        
        x, _ = self.spatial_attn(x, x, x)
        
        # Return to 4D and add residual
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)
        x = res + x
        
        # ----------------------------------------------------
        # 3. MLP
        # ----------------------------------------------------
        # The MLP operates on each token independently, so we can flatten batch & time
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = x + self.mlp(self.norm3(x))
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)
        
        return x

class TimeSformerTiny(nn.Module):
    def __init__(
        self,
        model_cfg: DictConfig,
        num_classes=33, 
        num_frames=4,
        dropout=0.1
    ):
        super().__init__()
        self.num_frames = num_frames
        self.d_model = int(model_cfg.get("d_model", 384))
        self.img_size = int(model_cfg.get("img_size", 112))
        
        # 1. Patch Extraction
        self.patch_embed = PatchEmbed(self.img_size, int(model_cfg.get("patch_size", 16)),
                                      int(model_cfg.get("in_chans", 3)), self.d_model)
        num_patches = self.patch_embed.num_patches
        
        # 2. Tokens & Positional Embeddings
        # We need a Spatial Pos Embedding AND a Temporal Pos Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, self.d_model))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches + 1, self.d_model))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, 1, self.d_model))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            DividedAttentionBlock(self.d_model, int(model_cfg.get("num_heads", 6)), dropout=dropout)
            for _ in range(int(model_cfg.get("depth", 8)))
        ])
        
        # 4. Classification Head
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, num_classes)
        
        # Initialization
        nn.init.trunc_normal_(self.spatial_pos_embed, std=.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Collapse batch and time for patch embedding
        x = x.view(B * T, C, H, W)

        # Force the resolution down to exactly 112x112, discarding high-freq noise
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear',
                          align_corners=False)

        x = self.patch_embed(x)  # (B*T, num_patches, d_model)
        
        # Reshape back to add time dimension: (B, T, num_patches, d_model)
        x = rearrange(x, '(b t) p d -> b t p d', b=B, t=T)
        
        # Add CLS token to each frame
        cls_tokens = self.cls_token.expand(B, T, -1, -1)  # (B, T, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=2)             # (B, T, num_patches+1, d_model)
        
        # Add Positional Embeddings
        # 1. Spatial pos embed broadcasts across time
        x = x + self.spatial_pos_embed
        # 2. Temporal pos embed broadcasts across space (patches)
        x = x + self.temporal_pos_embed
        x = self.pos_drop(x)
        
        N = x.shape[2] # N = num_patches + 1 (the CLS token)
        
        # Pass through Divided Space-Time blocks
        for block in self.blocks:
            x = block(x, B, T, N)
            
        # We only care about the CLS token representation for classification
        # Shape: (B, T, N, d_model) -> extract N=0
        cls_output = x[:, :, 0, :]  # (B, T, d_model)
        
        # Average the CLS token across the 4 frames to get video-level representation
        cls_output = cls_output.mean(dim=1)  # (B, d_model)
        
        # Final Norm and Classifier
        cls_output = self.norm(cls_output)
        return self.head(cls_output)