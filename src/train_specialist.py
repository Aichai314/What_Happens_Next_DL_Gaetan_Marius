#!/usr/bin/env python3
"""
Standalone trainer for the 9-class confusion SPECIALIST.

This script does NOT modify train.py or any shared file. It imports the generic
pieces (build_model, train_one_epoch, evaluate_epoch, dataset, transforms) and
adds only the specialist-specific logic:

  * filter the 33-class data down to the 9 confusable target classes
  * remap their challenge indices -> contiguous 0..8 for a 9-way head
  * class-balanced WeightedRandomSampler (Moving-up has 3170 train vids vs
    Dropping-into 903 — the imbalance is why "Picking up" collapses to 12%)

The resulting checkpoint stores the full resolved Hydra config (model.name=
vjepa2_large, num_classes=9), so it loads via build_model() exactly like any
other expert. Its 9-way softmax is meant as EXTRA FEATURES for the XGBoost
stack, not a standalone Kaggle submission.

Run (from repo root)::

    python src/train_specialist.py experiment=vjepa2_specialist \
      dataset.train_dir=$(pwd)/processed_data/val2/train \
      dataset.val_dir=$(pwd)/processed_data/val2/val \
      training.checkpoint_path=/Data/marius.truquin/Model_checkpoints/best_model_vjepa2_specialist.pt \
      +run_name=vjepa2_specialist_9c
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model, train_one_epoch, evaluate_epoch
from utils import VideoTransform, set_seed

# Challenge class index -> specialist index (0..8). Order = ascending challenge id.
# G1 vertical/lift + G2 real-vs-pretend, from the VJEPA confusion analysis.
TARGET_CLASSES: List[int] = [2, 9, 11, 14, 16, 17, 22, 29, 30]
CH2SPEC: Dict[int, int] = {ch: i for i, ch in enumerate(TARGET_CLASSES)}
SPEC2CH: Dict[int, int] = {i: ch for ch, i in CH2SPEC.items()}
CLASS_LABELS = {
    2: "Dropping into", 9: "Moving up", 11: "Picking up",
    14: "Pretending pick up", 16: "Pretending put into",
    17: "Pretending throw", 22: "Putting into", 29: "Throwing",
    30: "Turning upside down",
}


def filter_and_remap(samples: List[Tuple[Path, int]]) -> List[Tuple[Path, int]]:
    """Keep only the 9 target classes; remap their label to 0..8."""
    return [(p, CH2SPEC[c]) for (p, c) in samples if c in CH2SPEC]


def make_balanced_sampler(samples: List[Tuple[Path, int]]) -> WeightedRandomSampler:
    """Inverse-frequency weights so every one of the 9 classes is seen equally."""
    counts = Counter(label for _, label in samples)
    class_w = {c: 1.0 / n for c, n in counts.items()}
    weights = [class_w[label] for _, label in samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def per_class_report(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Print per-specialist-class recall on val; return overall accuracy."""
    model.eval()
    n = len(TARGET_CLASSES)
    correct = [0] * n
    total = [0] * n
    for video_batch, labels in loader:
        video_batch = video_batch.to(device)
        preds = model(video_batch).argmax(dim=1).cpu()
        for t, pr in zip(labels.tolist(), preds.tolist()):
            total[t] += 1
            if t == pr:
                correct[t] += 1
    print("\n--- Specialist per-class recall (val) ---")
    print(f"{'Class':<24} {'Recall':>8} {'Correct':>9} / {'Total':>5}")
    print("-" * 52)
    order = sorted(range(n), key=lambda i: (correct[i] / max(total[i], 1)))
    for i in order:
        ch = SPEC2CH[i]
        rec = correct[i] / max(total[i], 1)
        print(f"{CLASS_LABELS[ch]:<24} {rec:>8.1%} {correct[i]:>9} / {total[i]:>5}")
    overall = sum(correct) / max(sum(total), 1)
    print("-" * 52)
    print(f"{'OVERALL (9-class)':<24} {overall:>8.1%}  (VJEPA 33-way baseline on these = 61.9%)\n")
    return overall


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    assert int(cfg.model.num_classes) == len(TARGET_CLASSES), (
        f"model.num_classes must be {len(TARGET_CLASSES)} for the specialist; "
        f"use experiment=vjepa2_specialist"
    )
    set_seed(int(cfg.dataset.seed))

    wandb.init(
        project="what-happens-next-video",
        name=cfg.get("run_name", "vjepa2_specialist"),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()
    val_dir = Path(cfg.dataset.val_dir).resolve()
    train_samples = filter_and_remap(collect_video_samples(train_dir))
    val_samples = filter_and_remap(collect_video_samples(val_dir))
    print(f"Specialist data: {len(train_samples)} train / {len(val_samples)} val "
          f"(filtered to {len(TARGET_CLASSES)} classes)")
    train_dist = Counter(l for _, l in train_samples)
    print("Train class distribution (specialist idx -> n):",
          {k: train_dist[k] for k in sorted(train_dist)})

    use_imagenet_norm = bool(cfg.model.get("pretrained", True))
    image_size = int(cfg.dataset.get("image_size", 224))
    train_tf = VideoTransform(cfg, is_training=True, use_imagenet_norm=use_imagenet_norm, image_size=image_size)
    eval_tf = VideoTransform(cfg, is_training=False, use_imagenet_norm=use_imagenet_norm, image_size=image_size)

    num_frames = int(cfg.dataset.num_frames)
    train_ds = VideoFrameDataset(train_dir, num_frames, train_tf, sample_list=train_samples)
    val_ds = VideoFrameDataset(val_dir, num_frames, eval_tf, sample_list=val_samples)

    use_balanced = bool(cfg.training.get("balanced_sampling", True))
    sampler = make_balanced_sampler(train_samples) if use_balanced else None
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.training.batch_size),
        sampler=sampler, shuffle=(sampler is None),
        num_workers=int(cfg.training.num_workers), pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=int(cfg.training.num_workers), pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    wandb.watch(model, log="all", log_freq=200)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(cfg.training.get("label_smoothing", 0.0)))
    base_lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.get("weight_decay", 1e-4))
    if hasattr(model, "get_param_groups"):
        params = model.get_param_groups(base_lr, float(cfg.training.get("backbone_lr_factor", 0.1)))
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    total_epochs = int(cfg.training.epochs)
    accum = int(cfg.training.get("gradient_accumulation_steps", 1))
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = total_epochs * steps_per_epoch
    sched_type = cfg.training.get("scheduler", "none").lower()
    if sched_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    elif sched_type == "warmup_cosine":
        warmup_steps = int(cfg.training.get("warmup_epochs", 1)) * steps_per_epoch
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = None

    ckpt_path = Path(cfg.training.checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path = (Path(cfg.training.latest_checkpoint_path).resolve()
                   if cfg.training.get("latest_checkpoint_path") else None)

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(total_epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler,
            mixup_fn=None, accumulation_steps=accum, scheduler=scheduler,
        )
        va_loss, va_acc = evaluate_epoch(model, val_loader, loss_fn, device)
        wandb.log({"val/loss": va_loss, "val/accuracy_top1": va_acc, "epoch": epoch + 1})
        print(f"Epoch {epoch+1}/{total_epochs} | train {tr_acc:.4f} | val {va_acc:.4f} "
              f"| gap {tr_acc - va_acc:+.3f}")

        payload: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "num_classes": len(TARGET_CLASSES),
            "pretrained": bool(cfg.model.get("pretrained", True)),
            "num_frames": num_frames,
            "val_accuracy": va_acc,
            "config": OmegaConf.to_container(cfg, resolve=True),
            # Specialist-only metadata for ensemble integration / decoding:
            "specialist_target_classes": TARGET_CLASSES,
            "specialist_idx_to_challenge": SPEC2CH,
        }
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(payload, ckpt_path)
            print(f"  saved best -> {ckpt_path} (val={va_acc:.4f})")
        if latest_path is not None:
            torch.save(payload, latest_path)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Best 9-class val acc: {best_acc:.4f}")
    # Reload best and print the per-class breakdown vs the 61.9% baseline.
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    overall = per_class_report(model, val_loader, device)
    sidecar = ckpt_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({
        "target_classes_challenge_idx": TARGET_CLASSES,
        "specialist_idx_to_challenge": SPEC2CH,
        "best_val_accuracy_9class": best_acc,
        "final_val_accuracy_9class": overall,
        "vjepa_33way_baseline_on_these": 0.619,
    }, indent=2))
    print(f"Wrote {sidecar}")


if __name__ == "__main__":
    main()
