import torch
import hydra
import gc
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from omegaconf import DictConfig
import numpy
from typing import List, Optional, Dict
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import hashlib

# Import your existing utilities
from train import build_model
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import VideoTransform


def _get_cache_dir() -> Path:
    """Get or create the cache directory for storing computed logits."""
    cache_dir = Path(".ensemble_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _get_cache_path(ckpt_path: str, val_dir: str) -> Path:
    """
    Compute a unique cache file path for a checkpoint's validation logits.
    Hash is based on checkpoint path and validation directory to ensure uniqueness.
    """
    cache_key = hashlib.md5(f"{ckpt_path}:{val_dir}".encode()).hexdigest()
    ckpt_name = Path(ckpt_path).stem
    return _get_cache_dir() / f"{ckpt_name}_{cache_key}.npy"


def _get_labels_cache_path(val_dir: str) -> Path:
    """
    Compute a unique cache file path for validation labels.
    Hash is based on validation directory to ensure uniqueness.
    """
    cache_key = hashlib.md5(val_dir.encode()).hexdigest()
    return _get_cache_dir() / f"labels_{cache_key}.npy"

def _xgboost_stratified_kfold_cv(
    model_template: XGBClassifier,
    X_meta: numpy.ndarray,
    y_true: numpy.ndarray,
    n_splits: int = 5
) -> List[float]:
    """
    Manual k-fold cross-validation for XGBoost that handles class imbalance per fold.
    
    This bypasses sklearn's cross_val_score which fails when some folds don't have all classes.
    Each fold gets a fresh model instance to avoid state issues.
    """
    from sklearn.model_selection import StratifiedKFold
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    best_iterations = [] # NEW: Track the stopping points
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_meta, y_true)):
        # Create a fresh model for each fold
        fold_model = XGBClassifier(**model_template.get_params())
        
        X_train, X_val = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y_true[train_idx], y_true[val_idx]
        
        # Train on this fold
        fold_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on validation set
        fold_score = fold_model.score(X_val, y_val)
        fold_scores.append(fold_score)
        best_iterations.append(fold_model.best_iteration)
    
    return fold_scores, best_iterations

def _create_xgboost_model(hyperparams: Optional[Dict] = None) -> XGBClassifier:
    """
    Create an XGBClassifier with default or custom hyperparameters.
    
    Default hyperparams are tuned for late fusion MOE with 264 features (8 experts × 33 classes):
    - max_depth=4: Moderate tree depth to avoid overfitting high-dimensional features
    - learning_rate=0.05: Conservative step size for stable boosting
    - n_estimators=500: Many boosting rounds to compensate for lower learning rate
    - subsample=0.8: Row sampling to reduce variance
    - colsample_bytree=0.8: Feature sampling per tree
    - lambda=1.0: L2 regularization (analogous to LogReg's C=0.1)
    - alpha=0.2: L1 regularization for sparsity
    - early_stopping_rounds=30: Stop if no improvement for 30 rounds
    - objective='multi:softprob': Multi-class classification
    
    Note: num_class is NOT hardcoded and will be inferred by XGBoost from the data.
    This allows safe cross-validation where some folds may not have all classes.
    """
    if hyperparams is None:
        hyperparams = {}
    {'max_depth': 4, 'learning_rate': 0.04, 'n_estimators': 500, 'subsample': 0.7, 'colsample_bytree': 0.5, 'min_child_weight': 5, 'lambda': 4, 'alpha': 0.75}
    defaults = {
        'max_depth': 4,
        'learning_rate': 0.04,
        'n_estimators': 500,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'min_child_weight': 5,
        'lambda': 4,
        'alpha': 0.75,
        'early_stopping_rounds': 30,
        'objective': 'multi:softprob',
        'random_state': 42,
        'tree_method': 'hist' if torch.cuda.is_available() else 'auto',
        'verbosity': 0,
    }
    
    # Merge user hyperparams with defaults (user params override)
    defaults.update(hyperparams)
    return XGBClassifier(**defaults)


def _optimize_xgboost_hyperparams_optuna(
    X_meta: numpy.ndarray, 
    y_true: numpy.ndarray, 
    n_trials: int = 50
) -> Dict:
    """
    Use Bayesian optimization (Optuna) to find optimal XGBoost hyperparameters.
    Optimizes 5-fold stratified cross-validation accuracy with early pruning to speed up.
    
    Pruning: Trials with poor performance are stopped early to save computational time.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
            # Force L2 and L1 regularization not to be too low to prevent overfitting on validation set
            'lambda': trial.suggest_float('lambda', 5.0, 30.0),
            'alpha': trial.suggest_float('alpha', 1.0, 5.0),
            'early_stopping_rounds': 30,
        }
        
        model = _create_xgboost_model(params)
        
        # Evaluate on 5-fold CV with intermediate reporting for pruning
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_meta, y_true)):
            X_train, X_val = X_meta[train_idx], X_meta[val_idx]
            y_train, y_val = y_true[train_idx], y_true[val_idx]
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            fold_score = model.score(X_val, y_val)
            fold_scores.append(fold_score)
            
            # Report intermediate value for pruning
            intermediate_accuracy = numpy.mean(fold_scores)
            trial.report(intermediate_accuracy, step=fold_idx)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return numpy.mean(fold_scores)
    
    # Create study with TPE sampler and Median pruner for reproducibility and fast convergence
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
    
    print(f"\n🔍 Starting Bayesian Optimization with {n_trials} trials (pruning enabled)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"\n✅ Bayesian Optimization Complete!")
    print(f"   Best CV Accuracy: {best_score:.4f}")
    print(f"   Best Hyperparameters: {best_params}")
    
    return best_params


@torch.no_grad()
def evaluate_and_stack_n_models(
    ckpt_paths: List[str], 
    val_dir: str,
    meta_learner: str = 'logistic',
    use_bayesian_optimization: bool = False,
    bayesian_optimization_trials: int = 50,
    use_cache: bool = True,
):
    """
    Evaluate an ensemble of N video classification models using late fusion MOE.
    
    This function:
    1. Loads multiple pre-trained expert models sequentially (VRAM-safe)
    2. Extracts softmax probabilities from each expert on validation data (cached to .ensemble_cache/)
    3. Horizontally stacks all expert predictions (late fusion)
    4. Trains a meta-learner to combine expert predictions
    5. Reports 5-fold stratified cross-validation accuracy
    6. Trains final meta-learner on 100% validation data for Kaggle submission
    
    Args:
        ckpt_paths: List of checkpoint paths to load models from
        val_dir: Path to validation data directory
        meta_learner: Choice of meta-learner algorithm:
            - 'logistic': LogisticRegression with L2 regularization (C=0.1, balanced class weights)
            - 'xgboost': XGBClassifier with gradient boosting (default max_depth=4, learning_rate=0.05)
        use_bayesian_optimization: If True and meta_learner='xgboost', use Optuna to tune XGBoost hyperparams
            via 5-fold CV. Ignored if meta_learner='logistic'. Default: False (uses tuned default params)
        bayesian_optimization_trials: Number of Optuna trials for hyperparameter search.
            Only used if use_bayesian_optimization=True and meta_learner='xgboost'. Default: 50
        use_cache: If True, cache computed logits to .ensemble_cache/ and reuse on subsequent runs.
            Significantly speeds up re-runs with the same checkpoints and validation set. Default: True
    
    Returns:
        Trained meta-learner model (sklearn-compatible object)
    
    Example Usage:
        # Use LogisticRegression (default, fast)
        meta_model = evaluate_and_stack_n_models(
            ckpt_paths=my_models, 
            val_dir=\"processed_data/val2/val\"
        )
        
        # Use XGBoost with default hyperparameters (tuned for 264 features + 33 classes)
        meta_model = evaluate_and_stack_n_models(
            ckpt_paths=my_models,
            val_dir=\"processed_data/val2/val\",
            meta_learner='xgboost',
            use_bayesian_optimization=False
        )
        
        # Use XGBoost with Bayesian optimization to find best hyperparameters (~50 trials)
        meta_model = evaluate_and_stack_n_models(
            ckpt_paths=my_models,
            val_dir=\"processed_data/val2/val\",
            meta_learner='xgboost',
            use_bayesian_optimization=True,
            bayesian_optimization_trials=50
        )
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_samples = collect_video_samples(Path(val_dir))
    
    all_expert_probs = []
    y_true = None  # We will extract or load labels during the first model's loop
    
    # Try to load cached labels
    labels_cache_path = _get_labels_cache_path(val_dir) if use_cache else None
    if use_cache and labels_cache_path and labels_cache_path.exists():
        print(f"Loading cached labels from {labels_cache_path.name}...")
        y_true = numpy.load(labels_cache_path)
        print(f"Loaded {len(y_true)} labels from cache")

    # ---------------------------------------------------------
    # PART 1: SEQUENTIAL FEATURE EXTRACTION (VRAM SAFE)
    # ---------------------------------------------------------
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\n--- Processing Expert {i+1}/{len(ckpt_paths)}: {Path(ckpt_path).name} ---")
        
        # 0. Check cache first
        cache_path = _get_cache_path(ckpt_path, val_dir) if use_cache else None
        if use_cache and cache_path.exists():
            print(f"Loading cached logits from {cache_path.name}...")
            expert_probs = numpy.load(cache_path)
            all_expert_probs.append(expert_probs)
            
            # Still need to extract labels on first iteration if not already loaded from cache
            if i == 0 and y_true is None:
                val_samples_list = collect_video_samples(Path(val_dir))
                use_imagenet_norm = False  # Default fallback
                transform = VideoTransform({}, is_training=False, use_imagenet_norm=use_imagenet_norm) if hasattr(VideoTransform, '__init__') else None
                dataset = VideoFrameDataset(
                    root_dir=val_dir, 
                    num_frames=4, 
                    transform=transform, 
                    sample_list=val_samples_list
                )
                loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
                model_labels = []
                for batch, labels in loader:
                    model_labels.append(labels.numpy())
                y_true = numpy.concatenate(model_labels)
                
                # Cache labels for future runs
                if use_cache and labels_cache_path:
                    numpy.save(labels_cache_path, y_true)
                    print(f"Cached labels to {labels_cache_path.name}")
            
            print(f"Expert {i+1} accuracy on Validation Set: {accuracy_score(y_true, numpy.argmax(all_expert_probs[-1], axis=1)):.4f}")
            continue
        
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
            if i == 0 and y_true is None:
                model_labels.append(labels.numpy())

        # Stack this expert's logits and save to our master list
        expert_probs = numpy.vstack(model_probs)
        all_expert_probs.append(expert_probs)
        
        # Cache the logits for future runs
        if use_cache and cache_path:
            numpy.save(cache_path, expert_probs)
            print(f"Cached logits to {cache_path.name}")
        
        # Cache labels if not already loaded/cached
        if i == 0 and y_true is None:
            y_true = numpy.concatenate(model_labels)
            if use_cache and labels_cache_path:
                numpy.save(labels_cache_path, y_true)
                print(f"Cached labels to {labels_cache_path.name}")
        
        print(f"Expert {i+1} accuracy on Validation Set: {accuracy_score(y_true, numpy.argmax(all_expert_probs[-1], axis=1)):.4f}")

        # 4. CRITICAL: Clear VRAM before loading the next expert
        del model
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PART 2: BULLETPROOF META-LEARNER TRAINING (K-FOLD)
    # ---------------------------------------------------------

    print("\n" + "="*50)
    print("Evaluating N-Expert Meta-Learner with 5-Fold CV...")
    
    # Horizontally stack all expert probs
    X_meta = numpy.hstack(all_expert_probs)
    # ==========================================
    # FIX: SQUASH LABEL GAPS FOR XGBOOST (class 27 missing)
    # ==========================================
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    
    print(f"Combined Feature Shape: {X_meta.shape}")
    print(f"Original Unique Classes: {len(numpy.unique(y_true))}")
    print(f"Number of Features (N_experts × N_classes): {X_meta.shape[1]}")
    
    # Stratified K-Fold (The Truth Teller)
    # Ensures every fold has the exact same ratio of the 33 classes
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ---------------------------------------------------------
    # META-LEARNER SELECTION AND OPTIMIZATION
    # ---------------------------------------------------------
    if meta_learner == 'logistic':
        print("\n" + "="*50)
        print("Meta-Learner: LogisticRegression (L2 Regularization)")
        print("="*50)
        
        # The Heavy Regularization Model
        # C=0.1 applies strong L2 regularization to prevent validation memorization
        meta_model = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced')
        
    elif meta_learner == 'xgboost':
        print("\n" + "="*50)
        print("Meta-Learner: XGBoost (Gradient Boosting)")
        print("="*50)
        
        if use_bayesian_optimization:
            # Optimize hyperparameters using Bayesian search
            best_params = _optimize_xgboost_hyperparams_optuna(
                X_meta, y_true_encoded, n_trials=bayesian_optimization_trials
            )
            meta_model = _create_xgboost_model(best_params)
        else:
            # Use default hyperparameters
            # print("\nUsing default hyperparameters (no Bayesian optimization)")
            print("\nUsing stored hyperparameters from previous Bayesian optimization (+ regularized a little more for safety)")
            # Bayesian optim parameters:
            meta_model = _create_xgboost_model({'learning_rate': 0.09949269300375387, 'n_estimators': 181, 'subsample': 0.706441323736157, 'colsample_bytree': 0.6507684455172424, 'min_child_weight': 14, 'lambda': 27.36454008492742, 'alpha': 2.2069984738604624})
            print(f"Default Hyperparameters:")
            print(f"   max_depth: 4")
            print(f"   learning_rate: 0.04")
            print(f"   n_estimators: 500")
            print(f"   subsample: 0.7")
            print(f"   colsample_bytree: 0.5")
            print(f"   lambda (L2): 4")
            print(f"   alpha (L1): 0.75")
    
    # Run the Cross Validation
    print("\nEvaluating Meta-Learner with 5-Fold Stratified Cross-Validation...")
    
    if meta_learner == 'xgboost':
        # Unpack both the scores and the iterations
        cv_scores_list, best_iters = _xgboost_stratified_kfold_cv(
            meta_model, X_meta, y_true_encoded, n_splits=5
        )
        cv_scores = numpy.array(cv_scores_list)
        
        # Calculate the magical "Blind Run" target
        optimal_trees = int(numpy.mean(best_iters) * 1.1)  # Add 10% buffer to prevent underfitting
        print(f"\n🧠 Calculated Optimal Trees from CV: {optimal_trees}")
    else:
        # Logistic Regression doesn't use early stopping, so standard CV is fine
        cv_scores = cross_val_score(
            meta_model, X_meta, y_true_encoded, cv=cv, scoring='accuracy'
        )
    
    print("\n✅ K-Fold Validation Results:")
    for fold, score in enumerate(cv_scores):
        print(f"   Fold {fold + 1}: {score:.4f}")
        
    mean_acc = cv_scores.mean()
    std_acc = cv_scores.std()
    print(f"\n🚀 TRUTH SCORE (Mean Accuracy): {mean_acc:.4f} (±{std_acc:.4f})")
    
    # ---------------------------------------------------------
    # FINAL DEPLOYMENT
    # Now that we know it works safely, train the final model on 100% of the Validation Set 
    # so it is as smart as possible for the Kaggle Test Set.
    # ---------------------------------------------------------
    print("\nTraining Final Meta-Learner on 100% of Validation Data for Kaggle Submission...")
    
    if meta_learner == 'xgboost':
        final_params = meta_model.get_params()
        final_params['n_estimators'] = optimal_trees
        
        # We MUST remove early stopping parameters so it doesn't crash 
        # when we don't provide an eval_set
        if 'early_stopping_rounds' in final_params:
            del final_params['early_stopping_rounds']
        
        # Re-instantiate with optimal trees and no early stopping for final training
        meta_model = XGBClassifier(**final_params)
        
        # Train blindly on 100% of the data!
        meta_model.fit(X_meta, y_true_encoded, verbose=False)
        print(f"Final Model: Trained blindly on 100% data for exactly {optimal_trees} trees.")
    else:
        # LogisticRegression on full validation set
        meta_model.fit(X_meta, y_true_encoded)
        print(f"Final Model: Trained on 100% of validation data")

    return meta_model, le  # Return the label encoder for decoding predictions later

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================
    # CONFIGURATION: Define your Kaggle Roster here
    # =========================================================
    my_models = [
        "checkpoints/best_model_tsm_36-03.pt",
        "checkpoints/tsm_full_tdm_34-50.pt",
        "checkpoints/low_overfit_tdm_6channels_34-94.pt",
        "checkpoints/tsm_6channels_stem_37-95.pt",
        "checkpoints/attn_stage2_best_38-99.pt",
        "checkpoints/best_model_cnn_lstm_30-75.pt",
        "checkpoints/best_model_trn_29-53.pt",
        "checkpoints/best_model_x3d_xs_29-44.pt",
        "checkpoints/best_model_r2plus1d_30-97.pt",
        #"checkpoints/best_model_cnn_gru_30-54.pt",
        #"checkpoints/cnn_lstm_6channels_35-20.pt",
        "checkpoints/convnext_best_27-04.pt",
        "checkpoints/timesformer_best_24-33.pt",
        #"checkpoints/mobilenet_spatial_expert_38-09.pt",
        "checkpoints/mobilenet_motion_expert_33-92.pt",
        #"checkpoints/tsm_tdm_6channels_36_28.pt",
        #"checkpoints/mobilenet_6channels_37-58.pt",
        "checkpoints/efficientnet_6channels_39-78.pt",
        "checkpoints/efficientnet_attn_40-79.pt",
        "checkpoints/efficientnet_spatial_40-96.pt",
        "checkpoints/best_model_cnn_lstm_31-71.pt",
        "checkpoints/best_model_trn_32-90.pt",
        "checkpoints/efficientnet_tdm_39-87.pt",
    ]

    val_dir = str(Path(cfg.dataset.val_dir).resolve())
    
    # ========================================================================
    # NOTE: Logits are cached in .ensemble_cache/ to speed up re-runs!
    # Delete .ensemble_cache/ if you change validation data or model checkpoints
    # ========================================================================
    
    # ========================================================================
    # USAGE EXAMPLES (uncomment one to run)
    # ========================================================================
    
    # Example 1: Default - LogisticRegression with L2 regularization
    # Fast, lightweight, good baseline
    # meta_model, _ = evaluate_and_stack_n_models(
    #     ckpt_paths=my_models,
    #     val_dir=val_dir,
    #     meta_learner='logistic'
    # )
    
    # Example 2: XGBoost with default hyperparameters (tuned for 264 features)
    # Faster than Bayesian optimization, uses good defaults
    # On first run: computes and caches logits (~10 min)
    # On subsequent runs: loads from cache (~1 min)
    # meta_model, _ = evaluate_and_stack_n_models(
    #     ckpt_paths=my_models,
    #     val_dir=val_dir,
    #     meta_learner='xgboost',
    #     use_bayesian_optimization=False,
    #     use_cache=True  # Set to False to recompute logits
    # )
    
    # Example 3: XGBoost with Bayesian Optimization (find best hyperparams)
    # Slower but can improve accuracy by optimizing for your specific ensemble
    # Runtime: ~45 minutes for 50 trials with 5-fold CV (after logits are cached)
    meta_model, _ = evaluate_and_stack_n_models(
        ckpt_paths=my_models,
        val_dir=val_dir,
        meta_learner='xgboost',
        use_bayesian_optimization=True,
        bayesian_optimization_trials=75,
        use_cache=True
    )

if __name__ == "__main__":
    main()