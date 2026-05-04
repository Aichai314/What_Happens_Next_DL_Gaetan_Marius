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
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


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

        return torch.stack(result)  # (T, C, H, W)


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
