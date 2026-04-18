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
from typing import Any, Dict, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.cnn3d_transformer import CNN3DTransformer
from models.first_cnn import FirstCNN
from utils import build_transforms, set_seed, split_train_val


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

    raise ValueError(f"Unknown model.name: {name}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(data_loader, desc="Train", leave=False)
    for video_batch, labels in pbar:
        # video_batch: (B, T, C, H, W), labels: (B,)
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(video_batch)  # (B, num_classes)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
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
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
    )

    # Match normalization to pretrained flag (ImageNet stats when using pretrained weights).
    use_imagenet_norm = bool(cfg.model.pretrained)
    train_transform = build_transforms(
        is_training=True, use_imagenet_norm=use_imagenet_norm
    )
    eval_transform = build_transforms(
        is_training=False, use_imagenet_norm=use_imagenet_norm
    )

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_transform,
        sample_list=train_samples,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.lr))

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    t_start = time.time()

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model.name,
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

            torch.save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")

    results_path = checkpoint_path.parent / "training_results.csv"
    log_run(cfg, best_val_accuracy, time.time() - t_start, results_path)


if __name__ == "__main__":
    main()
