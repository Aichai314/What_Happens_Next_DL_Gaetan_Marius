"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.cnn3d_transformer import CNN3DTransformer
from models.first_cnn import FirstCNN
from models.vit_transformer import ViTTransformer
from models.TSM_resnet18 import TSMBaseline
from models.r2plus1d_baseline import R2Plus1DBaseline
from utils import VideoTransform, build_transforms, set_seed, split_train_val


def log_run(cfg: DictConfig, best_val_accuracy: float, duration_s: float, results_path: Path) -> None:
    """Append one row per completed training run to a shared CSV file."""
    row = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_min":    f"{duration_s / 60:.1f}",
        # model
        "model":           cfg.model.name,
        "pretrained":      cfg.model.get("pretrained", False),
        "num_classes":     cfg.model.num_classes,
        "d_model":         cfg.model.get("d_model", ""),
        "nhead":           cfg.model.get("nhead", ""),
        "num_layers":      cfg.model.get("num_layers", ""),
        "lstm_hidden":     cfg.model.get("lstm_hidden_size", ""),
        "dropout":         cfg.model.get("dropout", ""),
        # training
        "epochs":          cfg.training.epochs,
        "batch_size":      cfg.training.batch_size,
        "lr":              cfg.training.lr,
        "optimizer":       cfg.training.get("optimizer", "adam"),
        "scheduler":       cfg.training.get("scheduler", "none"),
        # data
        "num_frames":      cfg.dataset.num_frames,
        "val_ratio":       cfg.dataset.val_ratio,
        "seed":            cfg.dataset.seed,
        # result
        "best_val_acc":    f"{best_val_accuracy:.4f}",
        "checkpoint":      cfg.training.checkpoint_path,
    }

    write_header = not results_path.exists()
    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Run logged to {results_path}")


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained
    num_frames = cfg.dataset.num_frames

    if name == "tsm_baseline":
        dropout = cfg.model.get("dropout", 0.5)
        print("Building TSM with dropout, p =", dropout)
        return TSMBaseline(num_classes=num_classes, num_frames=num_frames, pretrained=pretrained, dropout=float(dropout))
    if name == "r2plus1d":
        return R2Plus1DBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )
    if name == "first_cnn":
        dropout = cfg.model.get("dropout", 0.5)
        return FirstCNN(num_classes=num_classes, dropout=float(dropout))
    if name == "cnn3d_transformer":
        return CNN3DTransformer(
            num_classes=num_classes,
            d_model=int(cfg.model.get("d_model", 256)),
            nhead=int(cfg.model.get("nhead", 8)),
            num_layers=int(cfg.model.get("num_layers", 4)),
            dropout=float(cfg.model.get("dropout", 0.1)),
        )
    if name == "vit_transformer":
        return ViTTransformer(
            num_classes=num_classes,
            unfreeze_blocks=int(cfg.model.get("unfreeze_blocks", 4)),
            temporal_layers=int(cfg.model.get("temporal_layers", 6)),
            temporal_heads=int(cfg.model.get("temporal_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.1)),
        )

    raise ValueError(f"Unknown model.name: {name}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    mixup_fn: Optional[v2.MixUp] = None,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == "cuda"

    pbar = tqdm(data_loader, desc="Train", leave=False)
    for video_batch, labels in pbar:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        # Apply Mixup on the GPU if configured
        if mixup_fn is not None:
            # 1. Save original shape
            B, T, C, H, W = video_batch.shape
            
            # 2. Collapse Batch and Time: (B, T, C, H, W) -> (B*T, C, H, W)
            # BUT: Mixup needs to mix videos, not individual frames independently.
            # So we treat the video as a "thick" image by stacking channels 
            # OR we apply the same lambda to all frames in a video.
            
            # The cleaner way for v2.MixUp with 5D:
            # Collapse Time into Channels temporarily: (B, T*C, H, W)
            video_batch = video_batch.view(B, T * C, H, W)
            
            # 3. Apply Mixup
            video_batch, mixup_labels = mixup_fn(video_batch, labels)
            
            # 4. Restore original 5D shape: (B, T, C, H, W)
            video_batch = video_batch.view(B, T, C, H, W)
        else:
            mixup_labels = labels

        optimizer.zero_grad()
        with torch.amp.autocast("cuda",enabled=use_amp):
            logits = model(video_batch)
            loss = loss_fn(logits, mixup_labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        
        # If Mixup is used, labels are probabilities (B, C). 
        # For training accuracy estimation, we compare against the dominant class.
        if mixup_fn is not None:
            target_labels = mixup_labels.argmax(dim=1)
        else:
            target_labels = labels
            
        correct += int((predictions == target_labels).sum().item())
        total += labels.size(0)
        pbar.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{correct / max(total, 1):.4f}")

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(data_loader, desc="Val  ", leave=False)
    for video_batch, labels in pbar:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        logits = model(video_batch)
        loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)
        pbar.set_postfix(loss=f"{running_loss / max(total, 1):.4f}", acc=f"{correct / max(total, 1):.4f}")

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()
    val_dir = Path(cfg.dataset.val_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]
        
    train_samples = all_samples
    val_samples = collect_video_samples(val_dir)

    # train_samples, val_samples = split_train_val(
    #     all_samples,
    #     val_ratio=float(cfg.dataset.val_ratio),
    #     seed=int(cfg.dataset.seed),
    # )

    use_imagenet_norm = bool(cfg.model.pretrained)
    use_augmentation = bool(cfg.training.get("augmentation", False))
    if use_augmentation:
        train_transform = VideoTransform(cfg, is_training=True,  use_imagenet_norm=use_imagenet_norm)
    else:
        train_transform = build_transforms(is_training=True, use_imagenet_norm=use_imagenet_norm)
    eval_transform = VideoTransform(cfg, is_training=False, use_imagenet_norm=use_imagenet_norm)

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_transform,
        sample_list=train_samples,
    )
    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    
    # --- Mixup Configuration ---
    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.0))
    mixup_fn = None
    if mixup_alpha > 0.0:
        mixup_fn = v2.MixUp(alpha=mixup_alpha, num_classes=int(cfg.model.num_classes))

    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # --- Optimizer Configuration (Adam vs AdamW) ---
    base_lr = float(cfg.training.lr)
    weight_decay = float(cfg.training.get("weight_decay", 1e-4))
    opt_type = cfg.training.get("optimizer", "adam").lower()

    if hasattr(model, "get_param_groups"):
        backbone_lr_factor = float(cfg.training.get("backbone_lr_factor", 0.1))
        params = model.get_param_groups(base_lr, backbone_lr_factor)
    else:
        params = model.parameters()

    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
        
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # --- Scheduler Configuration (Cosine vs Warmup+Cosine) ---
    sched_type = cfg.training.get("scheduler", "none").lower()
    total_epochs = int(cfg.training.epochs)
    
    if sched_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    elif sched_type == "warmup_cosine":
        warmup_epochs = int(cfg.training.get("warmup_epochs", 5))
        # Linear warmup from 1% of base_lr up to 100%
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        # Cosine decay for the remaining epochs
        cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = None

    best_val_accuracy = 0.0
    start_epoch = 0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    if cfg.training.get("latest_checkpoint_path"):
        latest_checkpoint_path = Path(cfg.training.latest_checkpoint_path).resolve()
    else:
        latest_checkpoint_path = None
        
    if cfg.training.get("resume_from", None):
        print(f"Resuming training from {cfg.training.resume_from}...")
        checkpoint = torch.load(cfg.training.resume_from, map_location=device)
        
        # 1. Restore Weights
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # 2. Restore Optimizer momentum
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 3. Restore Scheduler state (SAFE CHECK)
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore Scaler state (SAFE CHECK)
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # 4. Fast-forward the timeline
        start_epoch = checkpoint["epoch"] + 1
        best_val_accuracy = checkpoint.get("val_accuracy", 0.0)
        print(f"Successfully restored! Resuming at Epoch {start_epoch} with Best Acc {best_val_accuracy:.4f}")

    t_start = time.time()

    epoch_bar = tqdm(range(start_epoch, total_epochs), desc="Epochs", unit="ep")
    for epoch in epoch_bar:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, mixup_fn
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_bar.set_postfix(
            train_acc=f"{train_acc:.3f}",
            val_acc=f"{val_acc:.3f}",
            gap=f"{train_acc - val_acc:+.3f}",
            lr=f"{current_lr:.2e}",
        )
        print(
            f"Epoch {epoch + 1}/{total_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {current_lr:.2e}"
        )

        payload: Dict[str, Any] = {
            "epoch": epoch,                                      # Critical: Where did we stop?
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),      # Critical: AdamW momentum
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict(),
            "num_classes": int(cfg.model.num_classes),
            "pretrained": bool(cfg.model.pretrained),
            "num_frames": int(cfg.dataset.num_frames),
            "val_accuracy": val_acc,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if cfg.model.name == "cnn_lstm":
            payload["lstm_hidden_size"] = int(
                cfg.model.get("lstm_hidden_size", 512)
            )
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )
        if latest_checkpoint_path is not None:
            torch.save(payload, latest_checkpoint_path)
            print(
                f"  Saved latest model to {latest_checkpoint_path} (val acc={val_acc:.4f})"
            )
        

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")

    results_path = checkpoint_path.parent / "training_results.csv"
    log_run(cfg, best_val_accuracy, time.time() - t_start, results_path)


if __name__ == "__main__":
    main()