import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Import your existing utilities
from train import build_model
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import VideoTransform

@torch.no_grad()
def evaluate_and_plot(
    ckpt_path: str, 
    val_dir: str, 
    model_name: str, 
    num_classes: int = 33, 
    batch_size: int = 64,
    cmap: str = "Blues"
):
    """
    Evaluates a single model and generates a normalized confusion matrix heatmap.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # PART 1: LOAD MODEL & DATA
    # ---------------------------------------------------------
    print(f"\n[{model_name}] Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = OmegaConf.create(ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_samples = collect_video_samples(Path(val_dir))
    use_imagenet_norm = cfg.model.get("pretrained", False)

    dataset = VideoFrameDataset(
        root_dir=val_dir, 
        num_frames=4, 
        transform=VideoTransform(cfg, is_training=False, use_imagenet_norm=use_imagenet_norm), 
        sample_list=val_samples
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ---------------------------------------------------------
    # PART 2: RUN INFERENCE
    # ---------------------------------------------------------
    print(f"[{model_name}] Running Validation Set...")
    all_preds, all_labels = [], []

    for batch, labels in tqdm(loader):
        batch = batch.to(device)
        
        # Get predictions and move to CPU
        preds = model(batch).argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Calculate overall accuracy just to print it
    acc = (y_true == y_pred).mean()
    print(f"[{model_name}] Overall Accuracy: {acc:.4f}")

    # ---------------------------------------------------------
    # PART 3: GENERATE & SAVE HEATMAP
    # ---------------------------------------------------------
    print(f"[{model_name}] Generating Heatmap...")
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # Normalize by row (true class)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    sns.set_theme(style="white")
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(cm_norm, cmap=cmap, annot=False, vmin=0, vmax=1)
    
    plt.title(f"{model_name} - Normalized Accuracy", fontsize=16, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    
    # Format filename cleanly (e.g., "r(2+1)d_temporal_expert_cm.png")
    safe_filename = "".join(x if x.isalnum() else "_" for x in model_name).lower()
    save_path = f"heatmap_{safe_filename}.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved to '{save_path}'")


if __name__ == "__main__":
    # You can now cleanly process any number of models sequentially!
    
    evaluate_and_plot(
        ckpt_path="best_model_tsm.pt",
        val_dir="processed_data/val2/val",
        model_name="TSM Spatial Expert",
        cmap="Blues"
    )

    evaluate_and_plot(
        ckpt_path="best_model_r2plus1d.pt",
        val_dir="processed_data/val2/val",
        model_name="R(2+1)D Temporal Expert",
        cmap="Oranges"
    )