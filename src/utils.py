"""
Small helpers: reproducibility, image transforms, and metric computation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models
from PIL import Image
from functools import wraps

def two_stage_trainer(cls):
    """
    A class decorator that wraps the __init__ of any model.
    Handles Stage 1 (Backbone Grafting & Freezing) and Stage 2 (Full Resume & Fine-Tuning).
    """
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, model_cfg, *args, **kwargs):
        # 1. Run standard __init__ FIRST to build the complete architecture
        original_init(self, model_cfg, *args, **kwargs)

        # Retrieve paths from Hydra config
        pursue_path = model_cfg.get("pursue_from", None)
        backbone_path = model_cfg.get("pretrained_backbone_path", None)

        # =================================================================
        # STAGE 2: FULL MODEL RESUME (Backbone + Head)
        # =================================================================
        if pursue_path:
            print(f"--> [Wrapper] STAGE 2: Loading FULL model weights from: {pursue_path}")
            checkpoint = torch.load(pursue_path, map_location="cpu")
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Load the entire network (self) rather than just self.backbone
            self.load_state_dict(state_dict, strict=False)

        # =================================================================
        # STAGE 1: BACKBONE GRAFTING (Ignore Head)
        # =================================================================
        elif backbone_path:
            print(f"--> [Wrapper] STAGE 1: Loading strictly backbone from: {backbone_path}")
            checkpoint = torch.load(backbone_path, map_location="cpu")
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Extract ONLY the backbone weights
            backbone_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'backbone' in k}
            self.backbone.load_state_dict(backbone_dict, strict=False)

        # =================================================================
        # FREEZING LOGIC (Independent of Loading)
        # =================================================================
        if model_cfg.get("freeze_backbone", False):
            print("--> [Wrapper] LOCKING Backbone parameters (Gradients OFF).")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            print("--> [Wrapper] Backbone is UNFROZEN (Gradients ON). Training all layers.")
            for param in self.backbone.parameters():
                param.requires_grad = True

    # Replace the __init__
    cls.__init__ = new_init
    return cls


def set_seed(seed: int) -> None:
    """Make runs reproducible (as far as CUDA allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VideoTransform:
    """
    Augmentation cohérente pour un clip vidéo (liste de PIL Images).

    Les transforms SPATIALES (crop, grayscale) utilisent les mêmes
    paramètres aléatoires pour toutes les frames du clip — indispensable pour
    ne pas corrompre l'information temporelle.
    
    ATTENTION: Le Horizontal Flip a été retiré car il corrompt les labels 
    directionnels du dataset Something-Something (ex: left-to-right).

    Le ColorJitter est appliqué par frame.
    Le Temporal Jittering (drop frame) est appliqué au niveau de la liste.
    """

    def __init__(
        self,
        cfg: DictConfig,
        is_training: bool = True,
        use_imagenet_norm: bool = True,
        image_size: int = 224,
    ) -> None:
        self.is_training = is_training
        self.image_size = image_size
        self.cfg = cfg

        if use_imagenet_norm:
            self.mean = [0.485, 0.456, 0.406]
            self.std  = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std  = [0.5, 0.5, 0.5]

        if is_training:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
            )
        
        self.use_frame_differencing = cfg.get("dataset", {}).get("use_frame_differencing", False)

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        # Temporal Jittering: Simule une frame "droppée" en la dupliquant
        # Force le réseau à ne pas dépendre d'un timing parfait.
        if self.is_training and len(frames) > 2:
            if random.random() < float(self.cfg.augmentation.temporal_drop_prob): # 50% de chance d'appliquer le drop temporel
                drop_idx = random.randint(1, len(frames) - 1)
                frames[drop_idx] = frames[drop_idx - 1]

        crop_i, crop_j, crop_h, crop_w = 0, 0, frames[0].height, frames[0].width
        do_gray = False

        if self.is_training:
            # ── Paramètres spatiaux tirés UNE SEULE FOIS pour tout le clip ──
            crop_i, crop_j, crop_h, crop_w = transforms.RandomResizedCrop.get_params(  
                frames[0], scale=[0.7, 1.0], ratio=[3 / 4, 4 / 3]
            )
            
            # CRITICAL: Increase grayscale probability massively (e.g., 80%)
            # This blinds the model to specific colored objects.
            do_gray = random.random() < self.cfg.augmentation.grayscale_prob  # 80% de chance de convertir en gris (mais toujours 3 canaux pour la compatibilité) 
            
            # CRITICAL: Add Gaussian Blur to destroy sharp background textures
            do_blur = random.random() < self.cfg.augmentation.blur_prob  # 50% de chance d'appliquer le flou
            blurer = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))

        result: List[torch.Tensor] = []
        for img in frames:
            if self.is_training:
                # Spatial (identique pour toutes les frames)
                img = TF.resized_crop(  
                    img, crop_i, crop_j, crop_h, crop_w,
                    [self.image_size, self.image_size],
                )
                if do_gray:
                    img = TF.rgb_to_grayscale(img, num_output_channels=3)  
                if do_blur:
                    img = blurer(img)
                
                # Couleur (par frame)
                img = self.color_jitter(img)
            else:
                img = TF.resize(img, [self.image_size, self.image_size])  

            tensor = TF.to_tensor(img)  
            tensor = TF.normalize(tensor, self.mean, self.std)
            result.append(tensor)
        
        stacked_frames = torch.stack(result)  # (T, 3, H, W)
        
        # =========================================================
        # EXPLICIT FRAME DIFFERENCING
        # =========================================================
        if getattr(self, 'use_frame_differencing', False):
            # Create a zero tensor for the differences
            diffs = torch.zeros_like(stacked_frames)
            
            # Diff[t] = Frame[t] - Frame[t-1]
            diffs[1:] = stacked_frames[1:] - stacked_frames[:-1]
            
            # Concatenate along the channel dimension (dim=1)
            # Resulting shape: (T, 6, H, W)
            stacked_frames = torch.cat([stacked_frames, diffs], dim=1)

        return stacked_frames


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
) -> transforms.Compose:
    """
    Standard torchvision pipeline for single RGB frames.

    use_imagenet_norm:
        True  -> mean/std from ImageNet (usual when pretrained=True)
        False -> still scale to [0,1]; you can swap norms if you prefer
    """
    if use_imagenet_norm:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[torch.Tensor, ...]:
    """
    Compute top-k correctness for each k in topk.

    logits: (batch_size, num_classes)
    targets: (batch_size,) integer class indices
    Returns a tuple of tensors, each shape (1,) with accuracy in [0, 1].
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    # (batch_size, max_k) indices of top predictions
    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()  # (max_k, batch_size)
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    accuracies = []
    for k in topk:
        # Any hit in the top-k row slice counts
        accuracies.append(correct[:k].reshape(-1).float().sum() / batch_size)
    return tuple(accuracies)


def split_train_val(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Stratified split of (video_path, label) into train and validation portions.
    Ensures that every class has the exact same ratio in both sets.
    """
    rng = random.Random(seed)
    
    # 1. Group samples by class
    by_class: Dict[int, List[Tuple[Path, int]]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []

    # 2. Split each class individually (Stratification)
    for cls, cls_samples in by_class.items():
        rng.shuffle(cls_samples)
        
        if val_ratio <= 0.0:
            train_samples.extend(cls_samples)
            continue
            
        n_val = int(round(len(cls_samples) * val_ratio))
        # Ensure at least 1 val sample if the class has more than 1 total sample
        n_val = max(1, n_val) if len(cls_samples) > 1 else 0

        val_samples.extend(cls_samples[:n_val])
        train_samples.extend(cls_samples[n_val:])

    # 3. Shuffle the final aggregated lists
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples

class TemporalShift(nn.Module):
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

def inject_tsm_into_resnet(model: nn.Module, num_frames: int, n_div: int = 8) -> nn.Module:
    """
    Iterates through a torchvision ResNet and wraps the first convolution 
    of each BasicBlock with the TemporalShift operation.
    """
    for name, module in model.named_modules():
        if isinstance(module, models.resnet.BasicBlock):
            # Wrap conv1 in the BasicBlock
            module.conv1 = TemporalShift(module.conv1, num_frames=num_frames, n_div=n_div)
    return model

def replace_resnet_stem(backbone: nn.Module, in_channels: int = 3,
                        keep_original: bool = True) -> nn.Module:
    """
    Replaces the standard ResNet 7x7 stride-2 convolution with three 3x3 convolutions if keep_original is False.
    Otherwise, it replaces it simply changes the number of input channels.
    This preserves spatial data much better early in the network.
    Automatically handles custom input channels (e.g., 6 for Early Fusion).
    """
    out_channels = backbone.conv1.out_channels # Usually 64
    mid_channels = out_channels // 2           # Usually 32

    if keep_original:
        # Just replace the original conv1 with a new one that has the desired in_channels
        backbone.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        return backbone

    # Create the new high-resolution stem
    new_stem = nn.Sequential(
        # 1st 3x3 Conv: Stride 2 (Handles the initial downsampling)
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        
        # 2nd 3x3 Conv: Stride 1 (Processes spatial patterns without shrinking)
        nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        
        # 3rd 3x3 Conv: Stride 1 (Projects to 64 channels to match original ResNet design)
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    )

    # Replace the massive 7x7 convolution
    backbone.conv1 = new_stem
    
    # NOTE: We do NOT replace backbone.bn1, backbone.relu, or backbone.maxpool.
    # The output of new_stem flows perfectly into the original bn1.
    return backbone

class TemporalDifference(nn.Module):
    """
    Calculates explicit semantic velocity between frames at the feature level.
    Adds the difference (t - (t-1)) as a residual to the current frame.
    """
    def __init__(self, net: nn.Module, num_frames: int) -> None:
        super().__init__()
        self.net = net
        self.num_frames = num_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.differencing(x, self.num_frames)
        return self.net(x)

    @staticmethod
    def differencing(x: torch.Tensor, num_frames: int) -> torch.Tensor:
        # x shape: (B*T, C, H, W)
        bt, c, h, w = x.size()
        batch_size = bt // num_frames
        
        # Reshape to explicitly expose the temporal dimension
        x_view = x.view(batch_size, num_frames, c, h, w)
        
        # Calculate the semantic velocity: V(t) = F(t) - F(t-1)
        diffs = torch.zeros_like(x_view)
        diffs[:, 1:] = x_view[:, 1:] - x_view[:, :-1]
        
        # Inject velocity back into the spatial representation
        # Mathematically this acts as a first-order Taylor expansion predicting the next frame
        out = x_view + diffs
        
        # Flatten back to (B*T, C, H, W) for the 2D CNN
        return out.view(bt, c, h, w)

class SplitTDM(nn.Module):
    """
    Channel-Split Temporal Difference Module.
    Divides features into 3 groups: Spatial, Past Velocity, and Future Velocity.
    Perfect for training from scratch as it prevents variance explosion.
    """
    def __init__(self, net: nn.Module, num_frames: int) -> None:
        super().__init__()
        self.net = net
        self.num_frames = num_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.split_differencing(x, self.num_frames)
        return self.net(x)

    @staticmethod
    def split_differencing(x: torch.Tensor, num_frames: int) -> torch.Tensor:
        # x shape: (B*T, C, H, W)
        bt, c, h, w = x.size()
        batch_size = bt // num_frames
        
        # Reshape to explicitly expose the temporal dimension
        x_view = x.view(batch_size, num_frames, c, h, w)
        out = torch.zeros_like(x_view)
        
        # Calculate split sizes
        c1 = c // 2          # 50% for standard spatial (Safe Zone)
        c2 = c // 4          # 25% for Past Velocity
        c3 = c - (c1 + c2)   # Remaining ~25% for Future Velocity
        
        # =========================================================
        # Group 1: Standard Spatial (The "Safe Zone" for learning)
        # =========================================================
        out[:, :, :c1] = x_view[:, :, :c1]
        
        # =========================================================
        # Group 2: Past Motion [ F(t) - F(t-1) ]
        # =========================================================
        # Note: Frame 0 has no past, so it remains 0 in the `out` tensor.
        out[:, 1:, c1:c1+c2] = x_view[:, 1:, c1:c1+c2] - x_view[:, :-1, c1:c1+c2]
        
        # =========================================================
        # Group 3: Future Motion [ F(t+1) - F(t) ]
        # =========================================================
        # Note: The last frame has no future, so it remains 0.
        out[:, :-1, c1+c2:] = x_view[:, 1:, c1+c2:] - x_view[:, :-1, c1+c2:]
        
        # Flatten back to (B*T, C, H, W) for the 2D CNN
        return out.view(bt, c, h, w)


def inject_tdm_into_resnet(model: nn.Module, num_frames: int, full: bool = False,
                           split: bool = True) -> nn.Module:
    """
    Iterates through a torchvision ResNet and wraps the SECOND convolution 
    of each BasicBlock with the SplitTDM (default) or TemporalDifference operation.
    This works in perfect synergy with TSM (which wraps conv1).
    """
    for name, module in model.named_modules():
        if isinstance(module, models.resnet.BasicBlock) and (full or'layer4' in name):
            print(f"--> Injecting {"SplitTDM" if split else "TemporalDifference"} into {name}.conv2")
            # Wrap conv2 in the BasicBlock (TSM is on conv1)
            if split:
                module.conv2 = SplitTDM(module.conv2, num_frames=num_frames)
            else:
                module.conv2 = TemporalDifference(module.conv2, num_frames=num_frames)
    return model