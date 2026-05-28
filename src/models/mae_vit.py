from omegaconf import DictConfig
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

class MAE_ViT(nn.Module):
    """
    Masked Autoencoder (MAE) based on ViT-Small (patch size 16).
    """
    def __init__(
        self,
        model_cfg: DictConfig,
        img_size=224,
        patch_size=16,
        in_chans=3,
        mask_ratio=0.75
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        embed_dim = model_cfg.get("embed_dim", 384)
        depth = model_cfg.get("depth", 12)
        num_heads = model_cfg.get("num_heads", 6)
        decoder_embed_dim = model_cfg.get("decoder_embed_dim", 192)
        decoder_depth = model_cfg.get("decoder_depth", 4)
        decoder_num_heads = model_cfg.get("decoder_num_heads", 3)
        
        # --------------------------------------------------------------------------
        # MAE ENCODER (The part you keep for Stage 2)
        # --------------------------------------------------------------------------
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------
        # MAE DECODER (The part you throw away later)
        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Output maps back to the raw RGB pixels of the 16x16 patch (16 * 16 * 3 = 768)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self._init_weights()

    def _init_weights(self):
        # Setup position embeddings (usually done with sine-cosine, but zero-init works for Kaggle constraints)
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)
        
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward_encoder(self, x):
        # 1. Patchify and Embed
        x = self.patch_embed(x) # (N, dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2) # (N, L, dim)
        
        # Add position embeddings BEFORE masking
        x = x + self.pos_embed[:, 1:, :]
        
        # 2. Masking logic (75%)
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # 3. Append CLS token and pass through blocks
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(N, -1, -1)
        x_kept = torch.cat((cls_tokens, x_kept), dim=1)
        
        for blk in self.blocks:
            x_kept = blk(x_kept)
        x_kept = self.norm(x_kept)
        
        return x_kept, ids_restore

    def forward_decoder(self, x_kept, ids_restore):
        # 1. Map to decoder dimension
        x = self.decoder_embed(x_kept)
        
        # 2. Re-insert the mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) # No cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Re-attach cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add decoder position embeddings
        x = x + self.decoder_pos_embed
        
        # 3. Pass through decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predict the raw pixels
        x = self.decoder_pred(x)
        
        # Remove cls token from predictions
        x = x[:, 1:, :]
        return x

    def forward(self, imgs):
        latent, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        target = self.patchify(imgs)
        return pred, target
