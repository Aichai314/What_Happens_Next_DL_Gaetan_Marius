#!/usr/bin/env python3
"""
Ensemble Submission Generator

This script:
1. Calls evaluate_and_stack_n_models to train the Logistic Regression router on the Val set.
2. Loops through the chosen experts to extract Softmax probabilities on the Test set.
3. Feeds the Test probabilities into the router to generate final predictions.
4. Writes the kaggle submission CSV.
"""

import csv
import gc
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

# 1. Import your highly-optimized directory parsing tools[cite: 5]
from create_submission import (
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
)

# 2. Import your Meta-Learner training function
# Note: Ensure this function returns 'meta_model' at the end of it!
from evaluate_ensemble import evaluate_and_stack_n_models


def extract_test_probabilities(
    ckpt_paths: List[str], test_root: Path, video_dirs: List[Path], device: torch.device
) -> np.ndarray:
    """
    Sequentially loads models to extract test set probabilities without crashing VRAM.
    Crucially, applies the exact same Softmax normalization as the Val set extraction.
    """
    all_expert_probs = []
    
    # Create dummy labels (0) since the test set has no ground truth[cite: 5]
    sample_list = [(p, 0) for p in video_dirs]

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\n--- Extracting Test Features for Expert {i+1}/{len(ckpt_paths)}: {Path(ckpt_path).name} ---")
        
        # Load Model
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = OmegaConf.create(ckpt["config"])
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Setup Transform & Dataset
        use_imagenet_norm = cfg.model.get("pretrained", False)
        transform = VideoTransform(cfg, is_training=False, use_imagenet_norm=use_imagenet_norm)

        dataset = VideoFrameDataset(
            root_dir=test_root,
            num_frames=int(ckpt.get("num_frames", cfg.dataset.num_frames)),
            transform=transform,
            sample_list=sample_list,
        )

        # Shuffle MUST be False to align the rows across all N models[cite: 3]
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=4
        )

        model_probs = []
        with torch.no_grad():
            for batch, _ in tqdm(loader, desc=f"Extracting Probabilities"):
                batch = batch.to(device)
                raw_logits = model(batch)
                # The Equalizer: Convert to 0.0-1.0 scale
                probs = torch.softmax(raw_logits, dim=1).cpu().numpy()
                model_probs.append(probs)

        # Stack this expert's array
        all_expert_probs.append(np.vstack(model_probs))

        # Clear VRAM[cite: 3]
        del model
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()

    # Horizontally stack (e.g., 4 models * 33 classes = 132 features per video)[cite: 3]
    X_test = np.hstack(all_expert_probs)
    return X_test


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================
    # CONFIGURATION: Define your Kaggle Roster here
    # =========================================================
    my_models = [
        "best_model_tsm_35-80.pt",
        "best_model_cnn_lstm_30-75.pt",
        "best_model_trn_29-53.pt",
        "best_model_x3d_xs_29-44.pt",
        "best_model_r2plus1d_30-97.pt",
    ]

    # PHASE 1: Train the Meta-Learner (Logistic Regression)
    print("\n" + "="*50)
    print("PHASE 1: Training the Meta-Learner on Validation Set")
    print("="*50)
    val_dir = Path(cfg.dataset.val_dir).resolve()
    meta_model = evaluate_and_stack_n_models(my_models, val_dir)

    # PHASE 2: Discover Test Videos using create_submission logic[cite: 5]
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

    # PHASE 3: Extract Meta-Features for Test Set
    print("\n" + "="*50)
    print("PHASE 3: Extracting Expert Probabilities (Test Set)")
    print("="*50)
    X_test = extract_test_probabilities(my_models, test_root, video_dirs, device)
    print(f"Final Test Feature Matrix Shape: {X_test.shape}")

    # PHASE 4: Predict & Generate CSV
    print("\n" + "="*50)
    print("PHASE 4: Generating Kaggle Predictions")
    print("="*50)
    predictions = meta_model.predict(X_test)

    # Save to CSV[cite: 5]
    output_path = Path("submissions/ensemble_submission_final.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, predictions):
            w.writerow([name, pred])

    print(f"✅ Kaggle Submission successfully written to: {output_path}")

if __name__ == "__main__":
    main()