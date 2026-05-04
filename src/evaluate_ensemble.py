import torch
import gc
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy
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
        all_expert_probs.append(numpy.vstack(model_probs))
        
        if i == 0:
            y_true = numpy.concatenate(model_labels)
        
        print(f"Expert {i+1} accuracy on Validation Set: {accuracy_score(y_true, numpy.argmax(all_expert_probs[-1], axis=1)):.4f}")

        # 4. CRITICAL: Clear VRAM before loading the next expert
        del model
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PART 2: BULLETPROOF META-LEARNER TRAINING (K-FOLD)
    # ---------------------------------------------------------
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import numpy as np

    print("\n" + "="*50)
    print("Evaluating N-Expert Meta-Learner with 5-Fold CV...")
    
    # Horizontally stack all expert probs
    X_meta = numpy.hstack(all_expert_probs)
    print(f"Combined Feature Shape: {X_meta.shape}")
    
    # 1. The Heavy Regularization Model
    # C=0.1 applies strong L2 regularization to prevent validation memorization
    meta_model = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced')
    
    # 2. Stratified K-Fold (The Truth Teller)
    # Ensures every fold has the exact same ratio of the 33 classes
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Run the Cross Validation
    cv_scores = cross_val_score(meta_model, X_meta, y_true, cv=cv, scoring='accuracy')
    
    print("\n✅ K-Fold Validation Results:")
    for fold, score in enumerate(cv_scores):
        print(f"   Fold {fold + 1}: {score:.4f}")
        
    mean_acc = cv_scores.mean()
    std_acc = cv_scores.std()
    print(f"\n🚀 TRUTH SCORE (Mean Accuracy): {mean_acc:.4f} (±{std_acc:.4f})")
    
    # 3. Final Deployment
    # Now that we know it works safely, train the final model on 100% of the Validation Set 
    # so it is as smart as possible for the Kaggle Test Set.
    print("\nTraining Final Meta-Learner on 100% of Validation Data for Kaggle Submission...")
    meta_model.fit(X_meta, y_true)

    return meta_model

if __name__ == "__main__":
    # You can now pass any number of models!
    my_models = [
        "best_model_tsm_35-72.pt",
        "best_model_cnn_lstm_30-75.pt",
        "best_model_trn_29-53.pt",
        "best_model_x3d_xs_29-44.pt",
        "best_model_r2plus1d_30-97.pt",
    ]
    
    evaluate_and_stack_n_models(
        ckpt_paths=my_models,
        val_dir="processed_data/val2/val"
    )