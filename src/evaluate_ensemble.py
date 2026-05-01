import torch
import numpy as np
import gc
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List

# Import your existing utilities
from train import build_model
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import VideoTransform

@torch.no_grad()
def evaluate_and_stack_n_models(ckpt_paths: List[str], val_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_samples = collect_video_samples(Path(val_dir))
    
    all_expert_probs = []
    y_true = None  # We will extract labels during the first model's loop

    # ---------------------------------------------------------
    # PART 1: SEQUENTIAL FEATURE EXTRACTION (VRAM SAFE)
    # ---------------------------------------------------------
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\n--- Processing Expert {i+1}/{len(ckpt_paths)}: {Path(ckpt_path).name} ---")
        
        # 1. Load Model dynamically
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = OmegaConf.create(ckpt["config"])
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # 2. Setup Dataloader specific to this model's config
        use_imagenet_norm = cfg.model.get("pretrained", False)
        transform = VideoTransform(cfg, is_training=False, use_imagenet_norm=use_imagenet_norm)
        
        dataset = VideoFrameDataset(
            root_dir=val_dir, 
            num_frames=4, 
            transform=transform, 
            sample_list=val_samples
        )
        
        # shuffle=False is the most critical parameter here to ensure row alignment across N models
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

        # 3. Extract Logits
        model_probs = []
        model_labels = []
        
        for batch, labels in tqdm(loader, desc=f"Extracting Logits"):
            batch = batch.to(device)
            # Get logits and move to CPU immediately
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            model_probs.append(probs)
            
            # We only need to collect true labels once
            if i == 0:
                model_labels.append(labels.numpy())

        # Stack this expert's logits and save to our master list
        all_expert_probs.append(np.vstack(model_probs))
        
        if i == 0:
            y_true = np.concatenate(model_labels)

        # 4. CRITICAL: Clear VRAM before loading the next expert
        del model
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PART 2: META-LEARNER TRAINING
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Training N-Expert Meta-Learner...")
    
    # Horizontally stack all expert probs
    # If 3 models and 33 classes, X_meta shape becomes [N_videos, 99]
    X_meta = np.hstack(all_expert_probs)
    print(f"Combined Feature Shape: {X_meta.shape}")
    
    # Split data: 50% to train the router, 50% to honestly evaluate it[cite: 1]
    X_train, X_test, y_train, y_test = train_test_split(
        X_meta, y_true, test_size=0.5, random_state=42, stratify=y_true
    )
    
    # Train the Logistic Regression Router[cite: 1]
    meta_model = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced')
    meta_model.fit(X_train, y_train)
    
    # Evaluate the Smart Ensemble[cite: 1]
    meta_preds = meta_model.predict(X_test)
    final_acc = accuracy_score(y_test, meta_preds)
    
    print(f"\n✅ Smart Stacking Accuracy: {final_acc:.4f}")
    
    # ---------------------------------------------------------
    # PART 3: BASELINE COMPARISON
    # ---------------------------------------------------------
    print("\n--- Baseline Comparison (on the exact same test split) ---")
    num_classes = all_expert_probs[0].shape[1]
    
    # Calculate standalone accuracy for every expert on the test split
    for i, ckpt_path in enumerate(ckpt_paths):
        # Extract just this expert's features from the concatenated test array
        start_idx = i * num_classes
        end_idx = (i + 1) * num_classes
        expert_test_logits = X_test[:, start_idx:end_idx]
        
        expert_preds = expert_test_logits.argmax(axis=1)
        expert_acc = accuracy_score(y_test, expert_preds)
        print(f"Expert {i+1} ({Path(ckpt_path).name}): {expert_acc:.4f}")

    return meta_model

if __name__ == "__main__":
    # You can now pass any number of models!
    my_models = [
        "best_model_tsm.pt",
        "best_model_r2plus1d.pt",
        "best_model_cnn_lstm_29-50.pt",
        # Add a 3rd, 4th, or 5th model easily
    ]
    
    evaluate_and_stack_n_models(
        ckpt_paths=my_models,
        val_dir="processed_data/val2/val"
    )