#!/usr/bin/env python3
"""
Ensemble Submission Generator (TTA + Meta-Learner Aware)

This script:
1. Trains the Meta-Learner (and TTA weights) on the Val set.
2. Loops through experts to extract Softmax probabilities on the Test set (applying TTA).
3. Feeds Test probabilities into the router to generate final predictions.
4. Writes the kaggle submission CSV.
"""

import csv
import gc
import hashlib
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from train import build_model
from dataset.video_dataset import VideoFrameDataset
from utils import VideoTransform

# 1. Import directory parsing tools
from create_submission import (
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
)

# 2. Import Meta-Learner AND TTA utilities
from evaluate_ensemble import evaluate_and_stack_n_models, precompute_tta_logits


def _get_cache_dir() -> Path:
    """Get or create the cache directory for storing computed logits."""
    cache_dir = Path(".ensemble_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _get_test_cache_path(ckpt_path: str, test_root: str, TTA: bool = False) -> Path:
    """
    Compute a unique cache file path for a checkpoint's test set probabilities.
    Includes TTA status in the hash.
    """
    cache_key = hashlib.md5(f"{ckpt_path}:{test_root}".encode()).hexdigest()
    ckpt_name = Path(ckpt_path).stem
    prefix = "tta_test_" if TTA else "test_"
    return _get_cache_dir() / f"{prefix}{ckpt_name}_{cache_key}.npy"


def extract_test_probabilities(
    ckpt_paths: List[str], 
    test_root: Path, 
    video_dirs: List[Path], 
    device: torch.device, 
    use_cache: bool = True,
    TTA: bool = False,
    tta_modules: List[torch.nn.Module] = None
) -> np.ndarray:
    
    all_expert_probs = []
    sample_list = [(p, 0) for p in video_dirs] # Dummy labels (0)

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\n--- Extracting Test Features for Expert {i+1}/{len(ckpt_paths)}: {Path(ckpt_path).name} ---")
        
        # 0. Check cache first
        cache_path = _get_test_cache_path(ckpt_path, str(test_root), TTA) if use_cache else None
        if use_cache and cache_path.exists():
            print(f"Loading cached test probabilities from {cache_path.name}...")
            expert_probs = np.load(cache_path)
            all_expert_probs.append(expert_probs)
            continue
        
        # Load Model
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = OmegaConf.create(ckpt["config"])
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        if TTA:
            print("Applying Learned Test Time Augmentation...")
            # 1. Get raw logits for all 6 transforms on the test set
            tta_logits, _ = precompute_tta_logits(model, str(test_root), sample_list, cfg, device)
            
            # 2. Retrieve the matching TTA weights learned on the Val set
            tta_module = tta_modules[i].cpu() # Move to CPU to match tta_logits
            tta_module.eval()
            
            # 3. Apply the learned weights to get the final blended logits
            with torch.no_grad():
                combined_logits = tta_module(list(tta_logits))
                
            # 4. Convert to probabilities
            expert_probs = torch.softmax(combined_logits, dim=1).numpy()
            
        else:
            # Standard single-pass extraction
            use_imagenet_norm = cfg.model.get("pretrained", False)
            transform = VideoTransform(cfg, is_training=False, use_imagenet_norm=use_imagenet_norm)

            dataset = VideoFrameDataset(
                root_dir=test_root,
                num_frames=int(ckpt.get("num_frames", cfg.dataset.num_frames)),
                transform=transform,
                sample_list=sample_list,
            )

            loader = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=False, num_workers=4
            )

            model_probs = []
            with torch.no_grad():
                for batch, _ in tqdm(loader, desc=f"Extracting Probabilities"):
                    batch = batch.to(device)
                    raw_logits = model(batch)
                    probs = torch.softmax(raw_logits, dim=1).cpu().numpy()
                    model_probs.append(probs)

            expert_probs = np.vstack(model_probs)

        # Cache & Cleanup
        all_expert_probs.append(expert_probs)
        if use_cache and cache_path:
            np.save(cache_path, expert_probs)
            print(f"Cached test probabilities to {cache_path.name}")

        del model
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()

    return np.hstack(all_expert_probs)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================
    # KAGGLE ROSTER
    # =========================================================
    my_models = [
        "checkpoints/timesformer_best_24-98.pt",
        "checkpoints/attn_stage2_best_38-99.pt",
        "checkpoints/best_model_cnn_lstm_30-75.pt",
        "checkpoints/best_model_trn_32-90.pt",
        "checkpoints/best_model_x3d_xs_29-64.pt",
        "checkpoints/R2Plus1D_high_ov_34-29.pt",
        "checkpoints/convnextv2_nano_30-36.pt",
        "checkpoints/efficientnetb0_motion_37-33.pt",
        "checkpoints/efficientnetb0_spatial_41-59.pt",
        "checkpoints/efficientnetb0_spatial_assym_41-11.pt",
        "checkpoints/efficientnetb0_tdn_40-13.pt",
        "checkpoints/efficientformer_tsm_attn_35-67.pt",
        "checkpoints/coatnet_tsm_37-21.pt",
        "checkpoints/mae_small_phase2_22-22.pt",
        "checkpoints/resnext_tsm_39-76.pt",
    ]

    # PHASE 1: Train Meta-Learner & TTA Weights
    print("\n" + "="*50)
    print("PHASE 1: Training Meta-Learner & TTA Weights (Val Set)")
    print("="*50)
    
    use_tta = True # <--- Master TTA Flag
    
    val_dir = Path(cfg.dataset.val_dir).resolve()
    meta_model, label_encoder, tta_modules = evaluate_and_stack_n_models(
        my_models,
        str(val_dir),
        meta_learner='attention',           # <--- Handles both LR and Attention dynamically
        use_bayesian_optimization=False,
        TTA=use_tta,
        use_cache=True
    )

    # PHASE 2: Discover Test Videos
    print("\n" + "="*50)
    print("PHASE 2: Parsing Test Dataset")
    print("="*50)
    test_root = Path(cfg.dataset.test_dir).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest")

    if manifest_cfg:
        manifest_path = Path(str(manifest_cfg)).resolve()
        video_names = load_manifest_video_names(manifest_path)
        video_dirs = resolve_video_dirs(test_root, video_names)
    else:
        video_names, video_dirs = discover_all_test_videos(test_root)

    print(f"Found {len(video_dirs)} test videos.")

    # PHASE 3: Extract Meta-Features
    print("\n" + "="*50)
    print("PHASE 3: Extracting Expert Probabilities (Test Set)")
    print("="*50)
    X_test = extract_test_probabilities(
        ckpt_paths=my_models, 
        test_root=test_root, 
        video_dirs=video_dirs, 
        device=device, 
        use_cache=True,
        TTA=use_tta,
        tta_modules=tta_modules
    )
    print(f"Final Test Feature Matrix Shape: {X_test.shape}")

    # PHASE 4: Predict & Generate CSV
    print("\n" + "="*50)
    print("PHASE 4: Generating Kaggle Predictions")
    print("="*50)
    
    if hasattr(meta_model, 'predict'):
        # 1. Scikit-Learn Route (XGBoost / LR) -> Needs decoding
        final_preds = meta_model.predict(X_test)
        if meta_model.__class__.__name__ != 'CombinedLRAttentionModel': 
            # Standard XGBoost or pure LR needs decoding
            predictions = label_encoder.inverse_transform(final_preds)
        else:
            # The Combined model already outputs the correct class indices (0-32)
            predictions = final_preds
    else:
        # 2. PyTorch Route -> Already in correct Kaggle format (0-32)
        meta_model.eval()
        with torch.no_grad():
            num_classes = X_test.shape[1] // len(my_models)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(
                X_test.shape[0], len(my_models), num_classes
            ).transpose(0, 1)
            
            if meta_model.__class__.__name__ == 'LearnedWeightedMean':
                probs = meta_model(list(X_test_tensor)).cpu().numpy()
            else:
                probs = meta_model(X_test_tensor).cpu().numpy()
                
            predictions = probs.argmax(axis=1) # Directly use these!

    # Save to CSV
    output_path = Path("submissions/ensemble_submission_tta_both.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, predictions):
            w.writerow([name, pred])

    print(f"✅ Kaggle Submission successfully written to: {output_path}")

if __name__ == "__main__":
    main()