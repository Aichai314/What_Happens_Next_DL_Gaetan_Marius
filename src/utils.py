"""
Small helpers: reproducibility, image transforms, and metric computation.
"""

from __future__ import annotations

import random
import math
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        tta_mode: str = None,
    ) -> None:
        self.is_training = is_training
        self.image_size = image_size
        self.cfg = cfg
        self.tta_mode = tta_mode

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
            self.max_rotation = float(cfg.augmentation.get("rotation_degrees", 10.0))
        
        self.use_frame_differencing = cfg.get("dataset", {}).get("use_frame_differencing", False)
        self.only_motion = cfg.get("dataset", {}).get("only_motion", False)

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
            angle = random.uniform(-self.max_rotation, self.max_rotation) if random.random() < self.cfg.augmentation.get("rotation_prob", 0.0) else 0.0
            
            # =========================================================
            # NEW: EXPLICIT COLOR JITTER PARAMS (Consistent across clip)
            # =========================================================
            # brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
            b_factor = random.uniform(0.6, 1.4)
            c_factor = random.uniform(0.6, 1.4)
            s_factor = random.uniform(0.7, 1.3)
            h_factor = random.uniform(-0.1, 0.1)
            
            # PyTorch's ColorJitter randomly shuffles the order of application.
            # We mimic that here to prevent bias.
            jitter_order = [0, 1, 2, 3]
            random.shuffle(jitter_order)

        result: List[torch.Tensor] = []

        # 1. TTA MODE (Validation with deterministic augmentation)
        if not self.is_training and self.tta_mode is not None:
            for img in frames:
                # Rotate
                if self.tta_mode == 'rotate_cw':
                    img = TF.rotate(img, 5.0)
                elif self.tta_mode == 'rotate_ccw':
                    img = TF.rotate(img, -5.0)
                
                # Resize / Crop
                if self.tta_mode == 'zoom':
                    # Scale 224 -> 256 for the zoom out effect
                    zoom_size = int(self.image_size * (256/224))
                    img = TF.resize(img, [zoom_size, zoom_size])
                    top = (zoom_size - self.image_size) // 2
                    left = (zoom_size - self.image_size) // 2
                    img = TF.crop(img, top, left, self.image_size, self.image_size)
                else:
                    img = TF.resize(img, [self.image_size, self.image_size])
                
                # Filters
                if self.tta_mode == 'gray':
                    img = TF.rgb_to_grayscale(img, num_output_channels=3)
                if self.tta_mode == 'blur':
                    blurer = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.0))
                    img = blurer(img)

                tensor = TF.to_tensor(img)
                tensor = TF.normalize(tensor, self.mean, self.std)
                result.append(tensor)

        # 2. STANDARD TRAINING MODE
        elif self.is_training:
            for img in frames:
                # Spatial (identique pour toutes les frames)
                img = TF.rotate(img, angle, fill=[128, 128, 128])
                img = TF.resized_crop(  
                    img, crop_i, crop_j, crop_h, crop_w,
                    [self.image_size, self.image_size],
                )
                if do_gray:
                    img = TF.rgb_to_grayscale(img, num_output_channels=3)  
                if do_blur:
                    img = blurer(img)
                
                # =========================================================
                # NEW: APPLY CONSISTENT COLOR JITTER
                # =========================================================
                if not do_gray: # Don't jitter color if we just made it grayscale
                    for idx in jitter_order:
                        if idx == 0: img = TF.adjust_brightness(img, b_factor)
                        elif idx == 1: img = TF.adjust_contrast(img, c_factor)
                        elif idx == 2: img = TF.adjust_saturation(img, s_factor)
                        elif idx == 3: img = TF.adjust_hue(img, h_factor)

                tensor = TF.to_tensor(img)
                tensor = TF.normalize(tensor, self.mean, self.std)
                result.append(tensor)
        else:
            for img in frames:
                img = TF.resize(img, [self.image_size, self.image_size])  
                tensor = TF.to_tensor(img)
                tensor = TF.normalize(tensor, self.mean, self.std)
                result.append(tensor)
        
        stacked_frames = torch.stack(result)  # (T, 3, H, W)
        
        if self.is_training and random.random() < self.cfg.augmentation.get("erasing_prob", 0.0):
            # Calculate the box size (e.g., between 2% and 10% of image area)
            area = self.image_size * self.image_size
            target_area = random.uniform(0.02, 0.1) * area
            aspect_ratio = random.uniform(0.3, 3.3)
            
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if h < self.image_size and w < self.image_size:
                i = random.randint(0, self.image_size - h)
                j = random.randint(0, self.image_size - w)
                
                # Apply the EXACT same black box across all time steps
                stacked_frames[:, :, i:i+h, j:j+w] = 0.0 # or use the mean values
        
        # =========================================================
        # EXPLICIT FRAME DIFFERENCING
        # =========================================================
        if self.use_frame_differencing:
            # Create a zero tensor for the differences
            diffs = torch.zeros_like(stacked_frames)
            
            # Diff[t] = Frame[t] - Frame[t-1]
            diffs[1:] = stacked_frames[1:] - stacked_frames[:-1]
            
            if self.only_motion:
                # If only_motion is True, we discard the original frames and keep only the differences
                return diffs
            
            # Concatenate along the channel dimension (dim=1)
            # Resulting shape: (T, 6, H, W)
            stacked_frames = torch.cat([stacked_frames, diffs], dim=1) 
            if self.is_training and self.cfg.augmentation.get("remove_background", False):
                # Randomly zero out the original spatial channels to force reliance on motion
                if random.random() < self.cfg.augmentation.get("remove_rgb_prob", 0.15):
                    stacked_frames[:, 0:3, :, :] = 0.0 # Black out the RGB
                
                if random.random() < self.cfg.augmentation.get("heavy_spatial_blur", 0.2):
                    # Apply a strong Gaussian blur to the spatial channels to destroy fine details
                    blurer = transforms.GaussianBlur(kernel_size=(15, 15), sigma=(5.0, 10.0))
                    
                    # Apply blur directly to the 4D tensor slice (Time, Channels, Height, Width)
                    # This operates instantly and preserves your mean/std normalization!
                    stacked_frames[:, 0:3, :, :] = blurer(stacked_frames[:, 0:3, :, :])

        return stacked_frames

class TTATransform:
    """
    Factory class that returns 6 VideoTransform instances configured for TTA.
    Since they are instances of VideoTransform, they natively handle
    (T, 6, H, W) frame differencing and are compatible with VideoFrameDataset.
    """
    
    def __init__(self, cfg: DictConfig, use_imagenet_norm: bool = False, image_size: int = 224):
        self.cfg = cfg
        self.use_imagenet_norm = use_imagenet_norm
        self.image_size = image_size
    
    def get_transforms(self):
        """Returns the list of 6 configured TTA transforms"""
        modes = [
            None,          # 1. Original (No TTA flag)
            'zoom',        # 2. Slight Zoom out
            'gray',        # 3. Grayscale
            'blur',        # 4. Slight Blur
            'rotate_cw',   # 5. Rotation +5°
            'rotate_ccw',  # 6. Rotation -5°
        ]
        
        transforms_list = []
        for mode in modes:
            transforms_list.append(
                VideoTransform(
                    cfg=self.cfg,
                    is_training=False,
                    use_imagenet_norm=self.use_imagenet_norm,
                    image_size=self.image_size,
                    tta_mode=mode
                )
            )
            
        return transforms_list

class TTAWeightedMean(nn.Module):
    def __init__(self, n_transforms: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_transforms))
    
    def forward(self, logits_list):
        # logits_list : liste de (B, num_classes)
        w = F.softmax(self.weights, dim=0)
        stacked = torch.stack(logits_list, dim=0)  # (N_tta, B, num_classes)
        return (w[:, None, None] * stacked).sum(0)  # (B, num_classes)


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

# Format: {True_Class (Majority): [(Sibling_Class (Minority), leak_weight)]}
ASYMMETRIC_SIMILARITY_MAP = {
    
    # ── THE "PRETENDING" CLUSTER ─────────────────────────────────────────────
    # Pretending actions are visually identical for the first 80% of the video.
    # The network must be forgiven for confusing them, but the Minority must be protected.

    # True: 22 (Putting into, ~2200 samples) -> Leak to 16 (Pretending, ~300 samples)
    22: [(16, 0.15)], 
    
    # True: 11 (Picking up, ~1250 samples) -> Leak to 14 (Pretending, ~300 samples)
    11: [(14, 0.15)],

    # True: 13 (Pouring out of, ~900 samples) -> Leak to 15 (Pretending to pour out, ~300 samples)
    13: [(15, 0.15)],

    # True: 29 (Throwing, ~1700 samples) -> Leak to 17 (Pretending to throw, ~1500 samples)
    # Note: Less imbalanced, but highly semantically ambiguous. A smaller leak is safer.
    29: [(17, 0.10)],


    # ── THE "SPILLING" CRISIS ────────────────────────────────────────────────
    # Class 26 (Spilling next to) is your rarest class (~150 samples). 
    # Spilling is physically just a "failed" put or pour.
    
    # True: 23 (Putting next to, ~2000 samples) -> Leak heavily to 26
    23: [(26, 0.15)],
    
    # True: 12 (Pouring into, ~1000 samples) -> Leak to 26
    12: [(26, 0.10)],


    # ── THE "DIRECTIONAL" BLUR ───────────────────────────────────────────────
    # Left-to-Right (18) and Right-to-Left (19) are massive classes (~1500+ samples).
    # Moving Away (06) and Moving Closer (07) are identical motions but on the Z-axis.
    # Sometimes camera angles make X-axis and Z-axis motion ambiguous.
    # If your confusion matrix shows bleeding here, add a tiny cross-leak:
    18: [(6, 0.05), (7, 0.05)],
    19: [(6, 0.05), (7, 0.05)],
}

class AsymmetricSmoothedCrossEntropy(nn.Module):
    """
    Custom Cross Entropy Loss that applies Asymmetric Semantic Smoothing.
    Automatically handles both hard integer labels and soft MixUp probabilities.
    """
    def __init__(self, similarity_map: dict = ASYMMETRIC_SIMILARITY_MAP, base_smooth: float = 0.2, num_classes: int = 33):
        super().__init__()
        self.similarity_map = similarity_map
        self.base_smooth = base_smooth
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ── 1. Format Targets ──────────────────────────────────────────────
        # If targets are 1D (hard integer labels from standard training)
        if targets.dim() == 1:
            targets_probs = F.one_hot(targets, num_classes=self.num_classes).float()
        # If targets are 2D (soft probabilities from MixUp)
        else:
            targets_probs = targets.float()

        # ── 2. Standard Uniform Smoothing ──────────────────────────────────
        # Creates a new tensor, so we don't accidentally modify dataloader references in-place
        smoothed = targets_probs * (1.0 - self.base_smooth) + (self.base_smooth / self.num_classes)
        
        # ── 3. Asymmetric Bleed ────────────────────────────────────────────
        for majority_class, leaks in self.similarity_map.items():
            for minority_sibling, leak_weight in leaks:
                # Calculate transfer based on the ORIGINAL target weight (handles MixUp splits perfectly)
                transfer = targets_probs[:, majority_class] * leak_weight
                
                smoothed[:, minority_sibling] += transfer
                smoothed[:, majority_class] -= transfer
                
        # ── 4. Final Cross Entropy ─────────────────────────────────────────
        # PyTorch F.cross_entropy natively computes cross entropy for probability targets
        return F.cross_entropy(logits, smoothed)