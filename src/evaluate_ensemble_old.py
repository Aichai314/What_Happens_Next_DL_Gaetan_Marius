import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# Import your existing utilities
from train import build_model
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import VideoTransform

@torch.no_grad()
def evaluate_ensemble(
    tsm_ckpt_path: str, 
    r2plus1d_ckpt_path: str, 
    val_dir: str, 
    weight_tsm: float = 0.5, 
    weight_r2d: float = 0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Checkpoint A (TSM Spatial Expert)...")
    ckpt_a = torch.load(tsm_ckpt_path, map_location=device)
    cfga = OmegaConf.create(ckpt_a["config"])
    model_a = build_model(cfga).to(device)
    model_a.load_state_dict(ckpt_a["model_state_dict"])
    model_a.eval()

    print("Loading Checkpoint B (R(2+1)D Temporal Expert)...")
    ckpt_b = torch.load(r2plus1d_ckpt_path, map_location=device)
    cfgb = OmegaConf.create(ckpt_b["config"])
    model_b = build_model(cfgb).to(device)
    model_b.load_state_dict(ckpt_b["model_state_dict"])
    model_b.eval()

    # Setup standard validation dataloader
    val_samples = collect_video_samples(Path(val_dir))
    
    # Ensure both models were trained with the same normalization assumptions
    # (Defaulting to the TSM config's assumptions)
    use_imagenet_norm = ckpt_a["config"]["model"].get("pretrained", False)
    eval_transform_a = VideoTransform(cfga, is_training=False, use_imagenet_norm=use_imagenet_norm)
    eval_transform_b = VideoTransform(cfgb, is_training=False, use_imagenet_norm=use_imagenet_norm)

    val_dataset_a = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=4, # Both were trained on 4 frames
        transform=eval_transform_a,
        sample_list=val_samples,
    )
    val_dataset_b = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=4,
        transform=eval_transform_b,
        sample_list=val_samples,
    )

    val_loader_a = torch.utils.data.DataLoader(
        val_dataset_a, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    val_loader_b = torch.utils.data.DataLoader(
        val_dataset_b, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    correct_top1 = 0
    total = 0

    pbar = tqdm(val_loader_a, desc="Ensemble Eval")
    for (video_batch_a, labels_a), (video_batch_b, _) in zip(pbar, val_loader_b):
        video_batch_a = video_batch_a.to(device)
        video_batch_b = video_batch_b.to(device)
        labels = labels_a.to(device) # labels are the same for both

        # Get raw probability scores (logits) from both models
        logits_a = model_a(video_batch_a)
        logits_b = model_b(video_batch_b)

        # --- LATE FUSION: Weighted Average of Logits ---
        fused_logits = (logits_a * weight_tsm) + (logits_b * weight_r2d)

        # Calculate accuracy on the fused predictions
        predictions = fused_logits.argmax(dim=1)
        correct_top1 += int((predictions == labels).sum().item())
        total += labels.size(0)
        
        pbar.set_postfix(acc=f"{correct_top1 / total:.4f}")

    final_acc = correct_top1 / total
    print(f"\n✅ Final Ensemble Accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    # Adjust paths accordingly!
    evaluate_ensemble(
        tsm_ckpt_path="best_model_tsm.pt",
        r2plus1d_ckpt_path="best_model_r2plus1d.pt",
        val_dir="processed_data/val2/val",
        weight_tsm=0.7,  # Because of the much higher accuracy of the TSM model, we give it more weight in the ensemble
        weight_r2d=0.3
    )