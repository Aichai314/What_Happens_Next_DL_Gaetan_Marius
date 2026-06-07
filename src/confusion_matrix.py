#!/usr/bin/env python3
"""
Compute and display a per-class confusion matrix from a trained checkpoint.

Runs inference on ``dataset.val_dir`` (labelled data) and prints a recall
table sorted from worst to best class, then saves a heatmap PNG next to
the checkpoint.

Example (from ``src/``)::

    python confusion_matrix.py training.checkpoint_path=outputs/best_model.pt
    python confusion_matrix.py training.checkpoint_path=/path/to/custom.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import VideoTransform, set_seed


def load_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device) -> nn.Module:
    if "config" not in checkpoint or checkpoint["config"] is None:
        raise ValueError(
            "Checkpoint has no 'config' entry. Train with the current train.py so the "
            "full Hydra config is saved with the weights."
        )
    cfg = OmegaConf.create(checkpoint["config"])
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_class_names(val_dir: Path, num_classes: int) -> List[str]:
    """Parse class names from folder names like '000_Closing_something'."""
    names = [f"class_{i}" for i in range(num_classes)]
    for d in sorted(val_dir.iterdir()):
        if not d.is_dir():
            continue
        parts = d.name.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            idx = int(parts[0])
            if idx < num_classes:
                names[idx] = parts[1].replace("_", " ")
    return names


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for video_batch, labels in tqdm(data_loader, desc="Inference", leave=False):
        video_batch = video_batch.to(device)
        preds = model(video_batch).argmax(dim=1).cpu()
        labels = labels.cpu()
        indices = labels * num_classes + preds
        cm += torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def print_per_class_table(cm: torch.Tensor, class_names: List[str]) -> None:
    per_class_total = cm.sum(dim=1).float()       # true positives + false negatives
    per_class_predicted = cm.sum(dim=0).float()   # true positives + false positives
    per_class_correct = cm.diagonal().float()
    recall    = per_class_correct / per_class_total.clamp(min=1)
    precision = per_class_correct / per_class_predicted.clamp(min=1)

    order = recall.argsort().tolist()
    print("\n--- Per-class metrics (sorted by recall, worst → best) ---")
    print(f"{'Action':<50} {'Precision':>10} {'Recall':>7}  {'Correct':>7} / {'Total':>5}")
    print("-" * 88)
    for i in order:
        if per_class_total[i].item() == 0:
            continue
        print(
            f"{class_names[i]:<50} {precision[i].item():>10.1%} {recall[i].item():>7.1%}"
            f"  {int(per_class_correct[i]):>7} / {int(per_class_total[i]):>5}"
        )
    overall = per_class_correct.sum() / per_class_total.sum()
    print("-" * 88)
    print(f"{'Overall accuracy':<50} {'':>10} {overall.item():>7.1%}\n")


def print_top_confusions(cm: torch.Tensor, class_names: List[str], top_k: int = 30) -> None:
    """Print the largest off-diagonal (true -> predicted) error cells, plus the
    dominant error sink per class. Used to define specialist confusion groups."""
    per_class_total = cm.sum(dim=1).float().clamp(min=1)
    confusions = []
    n = cm.shape[0]
    for t in range(n):
        for p in range(n):
            if t == p or cm[t, p].item() == 0:
                continue
            cnt = int(cm[t, p].item())
            confusions.append((cnt, cnt / per_class_total[t].item(), t, p))
    confusions.sort(reverse=True)

    print(f"\n--- Top {top_k} confusions (TRUE → PREDICTED), by absolute count ---")
    print(f"{'True':<38} {'→ Predicted':<38} {'N':>5} {'% of true':>9}")
    print("-" * 94)
    for cnt, frac, t, p in confusions[:top_k]:
        print(f"{class_names[t]:<38} → {class_names[p]:<36} {cnt:>5} {frac:>8.1%}")

    print("\n--- Dominant error sink per class (where each class leaks most) ---")
    print(f"{'True':<38} {'leaks most to':<38} {'N':>5} {'% of true':>9}")
    print("-" * 94)
    rows = []
    for t in range(n):
        off = cm[t].clone()
        off[t] = 0
        if off.sum().item() == 0:
            continue
        p = int(off.argmax().item())
        cnt = int(off[p].item())
        rows.append((cnt / per_class_total[t].item(), cnt, t, p))
    rows.sort(reverse=True)
    for frac, cnt, t, p in rows:
        print(f"{class_names[t]:<38} → {class_names[p]:<36} {cnt:>5} {frac:>8.1%}")
    print()


def save_heatmap(cm: torch.Tensor, class_names: List[str], save_path: Path) -> None:
    per_class_total = cm.sum(dim=1).float()
    cm_norm = (cm.float() / per_class_total.unsqueeze(1).clamp(min=1)).numpy()
    short = [n[:28] for n in class_names]
    num_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {save_path.stem}\n(normalized by row = recall per class)")
    for i in range(num_classes):
        if cm_norm[i, i] > 0:
            color = "white" if cm_norm[i, i] > 0.5 else "black"
            ax.text(i, i, f"{cm_norm[i, i]:.0%}", ha="center", va="center",
                    fontsize=6, color=color)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {save_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = load_model_from_checkpoint(ckpt, device)

    pretrained_used = bool(ckpt.get("pretrained", cfg.model.pretrained))
    num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    num_classes = int(ckpt.get("num_classes", cfg.model.num_classes))

    # Resolve image_size in a model-agnostic way:
    #   1. checkpoint's own saved config (VJEPA stores 256)
    #   2. current Hydra cfg
    #   3. 224 (universal default — correct for VideoMAE / SlowFast / EfficientNet)
    # Models that never set image_size keep the previous 224 behaviour unchanged.
    ckpt_cfg = ckpt.get("config") or {}
    if not isinstance(ckpt_cfg, dict):  # OmegaConf → plain dict
        ckpt_cfg = OmegaConf.to_container(OmegaConf.create(ckpt_cfg), resolve=True)
    ckpt_dataset = ckpt_cfg.get("dataset", {}) if isinstance(ckpt_cfg, dict) else {}
    image_size = int(
        ckpt_dataset.get("image_size", cfg.dataset.get("image_size", 224))
    )
    print(f"Using image_size={image_size}, num_frames={num_frames}")
    eval_transform = VideoTransform(
        cfg, is_training=False, use_imagenet_norm=pretrained_used, image_size=image_size
    )

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        val_samples = val_samples[: int(max_samples)]

    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    print(f"Running on {len(val_dataset)} validation clips…")
    cm = compute_confusion_matrix(model, val_loader, device, num_classes)

    class_names = get_class_names(val_dir, num_classes)
    print_per_class_table(cm, class_names)
    print_top_confusions(cm, class_names, top_k=30)

    heatmap_path = checkpoint_path.with_name(checkpoint_path.stem + "_confusion.png")
    save_heatmap(cm, class_names, heatmap_path)


if __name__ == "__main__":
    main()
