import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import wandb
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import VideoTransform, set_seed
# Import the MAE model you generated earlier
from models.mae_vit import MAE_ViT


def train_one_epoch_mae(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    accumulation_steps: int = 1,
) -> float:
    """Returns average MSE loss on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    total_images = 0
    use_amp = device.type == "cuda"

    pbar = tqdm(data_loader, desc="Train MAE", leave=False)
    optimizer.zero_grad()
    
    for step, (video_batch, _) in enumerate(pbar):
        # video_batch is (B, T, C, H, W). We don't care about time or labels.
        # Flatten into a massive batch of independent images: (B*T, C, H, W)
        B, T, C, H, W = video_batch.shape
        images = video_batch.view(B * T, C, H, W).to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            preds, targets = model(images)
            # Standard MSE loss between predicted patches and actual patches
            loss = torch.mean((preds - targets) ** 2) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            wandb.log({
                "train/loss": loss.item() * accumulation_steps,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            })
            scaler.update()
            optimizer.zero_grad()

        # Multiply by batch size for accurate average tracking
        batch_size = images.size(0)
        running_loss += float(loss.item()) * accumulation_steps * batch_size
        total_images += batch_size
        
        pbar.set_postfix(mse=f"{running_loss / max(total_images, 1):.4f}")

    return running_loss / max(total_images, 1)


@torch.no_grad()
def evaluate_epoch_mae(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Returns average MSE loss on the validation loader."""
    model.eval()
    running_loss = 0.0
    total_images = 0

    pbar = tqdm(data_loader, desc="Val MAE  ", leave=False)
    for video_batch, _ in pbar:
        B, T, C, H, W = video_batch.shape
        images = video_batch.view(B * T, C, H, W).to(device)

        preds, targets = model(images)
        loss = torch.mean((preds - targets) ** 2)

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        total_images += batch_size
        
        pbar.set_postfix(mse=f"{running_loss / max(total_images, 1):.4f}")

    return running_loss / max(total_images, 1)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("="*50)
    print("🚀 STARTING MASKED AUTOENCODER (MAE) PRE-TRAINING")
    print("="*50)
    
    set_seed(int(cfg.dataset.seed))
    wandb.init(project="what-happens-next-video", config=OmegaConf.to_container(cfg, resolve=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = Path(cfg.dataset.train_dir).resolve()
    val_dir = Path(cfg.dataset.val_dir).resolve()
    
    train_samples = collect_video_samples(train_dir)
    val_samples = collect_video_samples(val_dir)

    # Standard spatial transforms (No MixUp, no TSM differences!)
    # We use ImageNet norm because later fine-tuning expects standard normalized inputs.
    image_size = int(cfg.dataset.get("image_size", 224))
    train_transform = VideoTransform(cfg, is_training=True, use_imagenet_norm=True, image_size=image_size)
    eval_transform = VideoTransform(cfg, is_training=False, use_imagenet_norm=True, image_size=image_size)

    # Datasets
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
        train_dataset, batch_size=int(cfg.training.batch_size), shuffle=True,
        num_workers=int(cfg.training.num_workers), pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=int(cfg.training.num_workers), pin_memory=True
    )

    # 1. Instantiate the Unsupervised MAE Model
    model = MAE_ViT(model_cfg=cfg.model,
                    in_chans=cfg.model.get("in_channels", 3),
                    mask_ratio=cfg.model.get("mask_ratio", 0.75)).to(device)
    
    # 2. MAE Specific Learning Rate Rule
    # lr = base_lr * (effective_batch_size / 256)
    accumulation_steps = int(cfg.training.get("gradient_accumulation_steps", 1))
    effective_batch_size = int(cfg.training.batch_size) * int(cfg.dataset.num_frames) * accumulation_steps
    
    base_lr = cfg.training.get("base_lr", 1.5e-4)
    actual_lr = base_lr * (effective_batch_size / 256.0)
    print(f"\n📈 MAE Learning Rate Scaling:")
    print(f"   Effective Batch Size (Videos x Frames x Accum): {effective_batch_size}")
    print(f"   Actual Scaled LR: {actual_lr:.2e}\n")

    # 3. Optimizer (Weight decay 0.05 is standard for MAE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=actual_lr, betas=(0.9, 0.95), weight_decay=cfg.training.get("weight_decay", 0.05))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # 4. Long Scheduler (400 epochs!)
    total_epochs = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 40)) # Longer warmup for MAE
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    # 5. Tracking BEST LOSS, not accuracy
    best_val_loss = float("inf")
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    
    start_epoch = 0
    if cfg.training.get("resume_from", None):
        print(f"Resuming training from {cfg.training.resume_from}...")
        checkpoint = torch.load(cfg.training.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if cfg.training.get("resume_scheduler", True):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_val_loss = checkpoint.get("val_loss", float("inf"))

    # =========================================================
    # TRAINING LOOP
    # =========================================================
    epoch_bar = tqdm(range(start_epoch, total_epochs), desc="Epochs", unit="ep")
    for epoch in epoch_bar:
        train_loss = train_one_epoch_mae(model, train_loader, optimizer, device, scaler, accumulation_steps)
        val_loss = evaluate_epoch_mae(model, val_loader, device)
        
        wandb.log({"val/loss": val_loss, "epoch": epoch + 1})
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_bar.set_postfix(train_mse=f"{train_loss:.4f}", val_mse=f"{val_loss:.4f}", lr=f"{current_lr:.2e}")
        print(f"Epoch {epoch + 1}/{total_epochs} | Train MSE {train_loss:.4f} | Val MSE {val_loss:.4f} | lr {current_lr:.2e}")

        # Save Checkpoint based on MINIMUM MSE LOSS
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": val_loss,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(payload, checkpoint_path)
            print(f"  🌟 Saved new BEST MAE model to {checkpoint_path} (Val MSE={val_loss:.4f})")
            
        # Always save latest
        if cfg.training.get("latest_checkpoint_path"):
            torch.save(payload, Path(cfg.training.latest_checkpoint_path).resolve())

    print(f"Done. Best MAE Validation MSE: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
