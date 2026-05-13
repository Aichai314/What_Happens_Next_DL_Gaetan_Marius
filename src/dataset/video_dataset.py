"""
VideoFrameDataset: loads a fixed number of RGB frames per video folder.

Expected layout under root_dir::

    root_dir/
      000_SomeClassName/
        video_12345/
          frame_000.jpg
          frame_001.jpg
          ...
      001_AnotherClass/
        ...

Class index is parsed from the leading number in the class folder name (000, 001, ...).
Each __getitem__ returns:
    video_tensor: float tensor of shape (T, C, H, W)
    label: int64 scalar class index
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

# The "Arrow of Time" Dictionary
# Maps a class ID to its perfect temporal opposite
REVERSIBLE_CLASSES = {
    0: 10, 10: 0,   # Closing <-> Opening
    1: 31, 31: 1,   # Covering <-> Uncovering
    3: 32, 32: 3,   # Folding <-> Unfolding
    6: 7, 7: 6,     # Moving away <-> Moving closer
    8: 9, 9: 8,     # Moving down <-> Moving up
    18: 19, 19: 18, # Left-to-right <-> Right-to-left
    22: 28, 28: 22  # Putting into <-> Taking out of
}

def _list_frame_paths(video_dir: Path) -> List[Path]:
    """All image files in a video folder, sorted by name."""
    paths: List[Path] = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(sorted(video_dir.glob(extension)))
    return sorted(paths, key=lambda p: p.name)


def _parse_class_index(class_dir_name: str) -> Optional[int]:
    """
    Expect folder names like '017_Class_name'. Returns 17, or None if no prefix.
    """
    match = re.match(r"^(\d+)_", class_dir_name)
    if match is None:
        return None
    return int(match.group(1))


def collect_video_samples(root_dir: Path) -> List[Tuple[Path, int]]:
    """
    Walk root_dir: each class folder contains video subfolders with frames.

    Returns list of (video_folder_path, class_index).
    """
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    samples: List[Tuple[Path, int]] = []
    class_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]

    # If folders lack numeric prefix, assign indices by sorted order (0..C-1).
    fallback_index = {p.name: i for i, p in enumerate(class_dirs)}

    for class_dir in class_dirs:
        parsed = _parse_class_index(class_dir.name)
        class_index = parsed if parsed is not None else fallback_index[class_dir.name]

        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            frame_paths = _list_frame_paths(video_dir)
            if len(frame_paths) == 0:
                continue
            samples.append((video_dir, class_index))

    if len(samples) == 0:
        raise RuntimeError(f"No video folders with frames under {root_dir}")

    return samples


def _pick_frame_indices(num_available: int, num_frames: int) -> List[int]:
    """
    Evenly spaced indices in [0, num_available - 1], inclusive.
    If fewer frames than requested, indices may repeat (last frame duplicated).
    """
    if num_available <= 0:
        raise ValueError("Video has no frames.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    if num_available == 1:
        return [0] * num_frames

    # linspace in index space
    positions = torch.linspace(0, num_available - 1, steps=num_frames)
    indices = [int(round(float(x))) for x in positions]
    return indices


def _pick_frame_indices_random(num_available: int, num_frames: int) -> List[int]:
    """
    Random temporal sampling: pick num_frames distinct indices uniformly at random
    from [0, num_available - 1], then sort to preserve temporal order.

    Used as a training-time augmentation when extra frames are available
    (e.g. 13 SSv2 frames per video vs the challenge's 4). Each epoch sees a different
    subset of frames, regularising temporal feature learning.

    Falls back to deterministic uniform sampling when num_available <= num_frames,
    so enabling the flag is safe even on machines with only the 4-frame challenge data.
    """
    if num_available <= 0:
        raise ValueError("Video has no frames.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if num_available <= num_frames:
        return _pick_frame_indices(num_available, num_frames)
    return sorted(random.sample(range(num_available), num_frames))


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int,
        transform: Callable[[Image.Image], torch.Tensor],
        sample_list: Optional[List[Tuple[Path, int]]] = None,
        random_temporal_sampling: bool = False,
    ) -> None:
        """
        Args:
            root_dir: Split root (contains class folders).
            num_frames: T in the returned tensor (T, C, H, W).
            transform: Applied independently to each PIL image (typically Resize + ToTensor + Normalize).
            sample_list: Optional pre-built list of (video_dir, label). Use for train/val splits.
            random_temporal_sampling: If True, randomly subsample num_frames indices when
                more are available (training-time augmentation). Always False for val/test
                to keep evaluation deterministic. No-op when num_available <= num_frames.
        """
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.random_temporal_sampling = random_temporal_sampling

        if sample_list is None:
            self.samples = collect_video_samples(self.root_dir)
        else:
            self.samples = list(sample_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_dir, label = self.samples[index]
        frame_paths = _list_frame_paths(video_dir)
        if self.random_temporal_sampling:
            indices = _pick_frame_indices_random(len(frame_paths), self.num_frames)
        else:
            indices = _pick_frame_indices(len(frame_paths), self.num_frames)

        pil_frames: List[Image.Image] = []
        for frame_index in indices:
            path = frame_paths[frame_index]
            with Image.open(path) as image:
                pil_frames.append(image.convert("RGB"))

        # VideoTransform: List[PIL] -> (T, C, H, W) avec augmentation cohérente
        # Fallback: per-frame transform puis stack (compatibilité ascendante)
        if callable(self.transform) and hasattr(self.transform, "is_training"):
            video_tensor = self.transform(pil_frames)
        else:
            video_tensor = torch.stack(
                [self.transform(img) for img in pil_frames], dim=0
            )

        label_tensor = torch.tensor(label, dtype=torch.long)
        return video_tensor, label_tensor
