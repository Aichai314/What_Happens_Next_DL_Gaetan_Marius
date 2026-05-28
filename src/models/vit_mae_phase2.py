import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from omegaconf import DictConfig

class SpaceTimeBlock(nn.Module):
    """
    SpaceTimeBlock (K=6) from the exp4a architecture diagram.
    Applies a Temporal Attention layer over the sequence of frames for each spatial patch,
    followed by the heavily pretrained Spatial Attention block from the MAE.
    """
    def __init__(self, spatial_block: Block, num_frames: int):
        super().__init__()
        self.spatial_block = spatial_block
        self.num_frames = num_frames
        
        # Extract dimensions from the pretrained spatial block
        dim = spatial_block.norm1.normalized_shape[0]
        num_heads = spatial_block.attn.num_heads
        
        # Added temporal attention
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # ZERO-INIT PROJECTION: This is a crucial trick.
        # By initializing the output projection of the temporal attention to 0, 
        # this block initially acts as a perfect identity mapping (doing nothing).
        # This prevents the random weights of the temporal attention from destroying 
        # the carefully pre-trained spatial representations during the first few epochs.
        nn.init.constant_(self.temporal_attn.out_proj.weight, 0)
        nn.init.constant_(self.temporal_attn.out_proj.bias, 0)

    def forward(self, x):
        # x shape from ViT: (B * T, N, D)  where N is num_patches + 1 (CLS)
        BT, N, D = x.shape
        T = self.num_frames
        B = BT // T
        
        # --- 1. Added Temporal Attention ---
        # Isolate the temporal sequence for each individual spatial patch
        # Reshape to (B, T, N, D), then transpose to (B, N, T, D), then merge B and N
        xt = x.view(B, T, N, D).transpose(1, 2).reshape(B * N, T, D)
        
        # Apply Temporal Attention with a residual connection
        res = xt
        xt = self.temporal_norm(xt)
        xt_out, _ = self.temporal_attn(xt, xt, xt, need_weights=False)
        xt = res + xt_out
        
        # Reshape back to (B * T, N, D) to feed into the spatial block
        x_temporal = xt.view(B, N, T, D).transpose(1, 2).reshape(B * T, N, D)
        
        # --- 2. Reused Spatial Attention ---
        # Feed the temporally-enriched tokens into the original pretrained spatial block
        out = self.spatial_block(x_temporal)
        
        return out


class PerceiverHead(nn.Module):
    """
    Perceiver head processing all video tokens via Cross-Attention with learned queries.
    """
    def __init__(self, dim: int, num_queries: int = 16, num_heads: int = 8, num_classes: int = 33, dropout: float = 0.3):
        super().__init__()
        # 16 learned queries ask all the tokens of the video
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        # Cross attention between queries and video tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # Standard Perceiver usually includes an MLP after cross attention
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Linear -> 33 classes
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x contains ALL tokens from the video: (B, T * N, D)
        B = x.shape[0]
        
        # Expand the 16 learned queries for the whole batch -> (B, 16, D)
        q = self.queries.expand(B, -1, -1)
        
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(x)
        
        # Queries (Q) cross-attend to all video tokens (K, V)
        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm, need_weights=False)
        
        # Residual and MLP
        out = q + attn_out
        out = out + self.mlp(self.norm_mlp(out)) # (B, 16, D)
        out = self.dropout(out)
        
        # mean over queries -> (B, D)
        out = out.mean(dim=1)
        
        # Linear -> 33 classes
        return self.classifier(out)


class ViT_MAE_Phase2(nn.Module):
    """
    Phase 2 Fine-Tuning Architecture based on the top competitor's diagram.
    Integrates a MAE-pretrained ViT, SpaceTime Blocks, and a Perceiver Head.
    """
    def __init__(
        self,
        model_cfg: DictConfig,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_frames: int = 4,
        num_classes: int = 33,
        space_time_k: int = 6 
    ):
        super().__init__()
        self.num_frames = num_frames
        self.space_time_k = space_time_k
        
        embed_dim = model_cfg.get("embed_dim", 384)
        self.depth = model_cfg.get("depth", 12)
        num_heads = model_cfg.get("num_heads", 6) 
        
        # --- NEW: Extract Dropout Config ---
        drop_path_rate = model_cfg.get("drop_path", 0.1)
        attn_drop = model_cfg.get("attn_drop", 0.0)
        proj_drop = model_cfg.get("proj_drop", 0.0)
        head_drop = model_cfg.get("head_drop", 0.0)
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        
        # --- NEW: Calculate Stochastic Depth Rates ---
        # Stochastic depth increases linearly deeper into the network
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        
        self.blocks = nn.ModuleList()
        for i in range(self.depth):
            # --- NEW: Pass dropout & drop_path to the spatial block ---
            spatial_block = Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=4, 
                qkv_bias=True, 
                norm_layer=nn.LayerNorm,
                proj_drop=proj_drop,      # MLP dropout
                attn_drop=attn_drop,      # Attention dropout
                drop_path=dpr[i]          # Stochastic depth
            )
            
            if i >= self.depth - space_time_k:
                self.blocks.append(SpaceTimeBlock(spatial_block, num_frames=num_frames))
            else:
                self.blocks.append(spatial_block)
                
        self.norm = nn.LayerNorm(embed_dim)
        
        # --- NEW: Pass head_drop to Perceiver Head ---
        self.perceiver_head = PerceiverHead(
            dim=embed_dim,
            num_queries=16,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=head_drop             # Add this parameter to PerceiverHead
        )

        self._init_weights()
        
        pretrained_path = model_cfg.get("pretrained_encoder", None)
        if pretrained_path:
            print(f"--> Initializing from pretrained MAE: {pretrained_path}")
            # Load to CPU first to prevent VRAM spikes on GPU 0
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            
            # Handle both raw PyTorch state_dicts and your custom saved payloads
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.load_pretrained_mae(state_dict)

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.pos_embed, std=.02)

    def load_pretrained_mae(self, state_dict):
        """
        Custom weight loading logic to cleanly load the MAE encoder weights into this Phase 2 model.
        Because we wrapped the last K blocks inside `SpaceTimeBlock`, the parameter keys
        like `blocks.11.norm1.weight` need to be mapped to `blocks.11.spatial_block.norm1.weight`.
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            # Only keep the encoder weights from MAE (drop decoder stuff)
            if 'decoder' in k or 'mask_token' in k:
                continue
                
            if k.startswith('blocks.'):
                parts = k.split('.')
                block_idx = int(parts[1])
                
                # If this block was converted to a SpaceTimeBlock, remap the key
                if block_idx >= len(self.blocks) - self.space_time_k:
                    # e.g., blocks.8.norm1.weight -> blocks.8.spatial_block.norm1.weight
                    new_k = f"blocks.{block_idx}.spatial_block." + ".".join(parts[2:])
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            else:
                new_state_dict[k] = v
                
        # Load weights with strict=False because the Temporal Attention parameters 
        # and the Perceiver Head parameters will not exist in the MAE checkpoint
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"--> Loaded MAE pre-trained weights.")
        print(f"    Missing keys (expected for Temporal/Perceiver): {len(msg.missing_keys)}")
        print(f"    Unexpected keys: {len(msg.unexpected_keys)}")
    
    def get_param_groups(self, base_lr: float, llrd: float = 0.75):
        """
        Generates parameter groups with Layer-wise Learning Rate Decay (LLRD).
        The classifier/perceiver gets the base_lr.
        Each spatial block going backward gets its LR multiplied by `llrd`.
        The patch_embed gets the lowest LR.
        """
        param_groups = []
        
        # 1. The Head / New components (gets full base_lr)
        # This includes the PerceiverHead and the newly initialized Temporal layers inside SpaceTimeBlocks
        head_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'perceiver_head' in name or 'temporal_attn' in name or 'temporal_norm' in name:
                head_params.append(p)
                
        if head_params:
            param_groups.append({"params": head_params, "lr": base_lr})

        # 2. The Spatial Backbone (decays by layer depth)
        # depth=12 means 12 blocks. We want block 11 to have base_lr * llrd
        # block 10 to have base_lr * llrd^2, etc.
        for idx in range(self.depth):
            # Calculate the decaying learning rate for this specific block
            # The deepest block (idx == depth - 1) gets base_lr * llrd^1
            layer_lr = base_lr * (llrd ** (self.depth - idx))
            
            layer_params = []
            for name, p in self.blocks[idx].named_parameters():
                if not p.requires_grad:
                    continue
                # Exclude the temporal layers we already caught in group 1
                if 'temporal' not in name: 
                    layer_params.append(p)
                    
            if layer_params:
                param_groups.append({"params": layer_params, "lr": layer_lr})

        # 3. The Stem (patch_embed, cls_token, pos_embed)
        # These get the lowest possible learning rate: base_lr * llrd^(depth + 1)
        stem_lr = base_lr * (llrd ** (self.depth + 1))
        stem_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name:
                stem_params.append(p)
                
        if stem_params:
            param_groups.append({"params": stem_params, "lr": stem_lr})
            
        print(f"--> Built {len(param_groups)} parameter groups for LLRD (factor: {llrd})")
        return param_groups

    def forward(self, x):
        # input clip: (B, 4, 3, 224, 224)
        B, T, C, H, W = x.shape
        
        # Fold temporal dimension into batch for spatial processing
        x = x.view(B * T, C, H, W)
        
        # per-frame patch-embed
        x = self.patch_embed(x) # (B*T, D, 14, 14)
        x = x.flatten(2).transpose(1, 2) # (B*T, 196, D)
        
        # + CLS + sincos pos
        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B*T, 197, D)
        x = x + self.pos_embed
        
        # Pass through all blocks (First blocks: Spatial, Other blocks: SpaceTime)
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x) # (B*T, 197, D)
        
        # return_all_tokens (B, 4, 197, 768)
        _, N, D = x.shape
        x = x.view(B, T, N, D)
        
        # Flatten time and spatial patches into a single massive sequence for the Perceiver
        x = x.view(B, T * N, D) # (B, 788, 768)
        
        # Perceiver head -> Linear -> 33 classes
        out = self.perceiver_head(x)
        
        return out

# --- Quick verification script ---
if __name__ == '__main__':
    from omegaconf import OmegaConf
    
    # Simulate a ViT-Small config
    cfg = OmegaConf.create({
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6
    })
    
    model = ViT_MAE_Phase2(cfg, num_frames=4, space_time_k=6)
    
    dummy_video = torch.randn(2, 4, 3, 224, 224)
    out = model(dummy_video)
    
    print(f"Input shape:  {dummy_video.shape}")
    print(f"Output shape: {out.shape}")
