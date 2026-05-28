import torch
import torch.nn.functional as F
import hydra
import gc
from pathlib import Path
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from omegaconf import DictConfig
import numpy as np
from typing import List, Optional, Dict
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import hashlib

# Import your existing utilities
from train import build_model
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import LearnedWeightedMean, VideoTransform, TTATransform, ExpertAttentionMeta


def _get_cache_dir() -> Path:
    """Get or create the cache directory for storing computed logits."""
    cache_dir = Path(".ensemble_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _get_cache_path(ckpt_path: str, val_dir: str, TTA: bool = False) -> Path:
    """
    Compute a unique cache file path for a checkpoint's validation logits.
    Hash is based on checkpoint path and validation directory to ensure uniqueness.
    """
    cache_key = hashlib.md5(f"{ckpt_path}:{val_dir}".encode()).hexdigest()
    ckpt_name = Path(ckpt_path).stem
    return _get_cache_dir() / f"{'tta_' if TTA else ''}{ckpt_name}_{cache_key}.npy"

def _get_tta_weights_cache_path(ckpt_path: str, val_dir: str) -> Path:
    """Compute a unique cache file path for the trained TTA weights."""
    cache_key = hashlib.md5(f"{ckpt_path}:{val_dir}".encode()).hexdigest()
    ckpt_name = Path(ckpt_path).stem
    return _get_cache_dir() / f"tta_weights_{ckpt_name}_{cache_key}.pt"


def _get_labels_cache_path(val_dir: str) -> Path:
    """
    Compute a unique cache file path for validation labels.
    Hash is based on validation directory to ensure uniqueness.
    """
    cache_key = hashlib.md5(val_dir.encode()).hexdigest()
    return _get_cache_dir() / f"labels_{cache_key}.npy"

def precompute_tta_logits(model, val_dir, val_samples, cfg, device):
    """
    Extracts logits for all TTA transformations for a given model.
    Returns:
        tta_logits: torch.Tensor of shape (N_tta, N_samples, num_classes)
        val_labels: torch.Tensor of shape (N_samples,)
    """
    
    use_imagenet_norm = cfg.model.get("pretrained", False)
    image_size = int(cfg.dataset.get("image_size", 224))
    
    # Initialize the 6 TTA transforms
    tta_transforms = TTATransform(cfg, image_size=image_size, use_imagenet_norm=use_imagenet_norm).get_transforms()
    
    N_tta = len(tta_transforms)
    all_tta_logits = []
    val_labels = None
    
    model.eval()
    with torch.no_grad():
        for t_idx, transform in enumerate(tta_transforms):
            print(f"  -> Extracting TTA {t_idx + 1}/{N_tta}...")
            
            # Create a dataset & loader specifically for this TTA transform
            dataset = VideoFrameDataset(
                root_dir=val_dir, 
                num_frames=int(cfg.dataset.get("num_frames", 4)), 
                transform=transform, 
                sample_list=val_samples
            )
            
            # shuffle=False is critical to maintain row alignment across the 6 passes!
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=False, num_workers=4
            )
            
            model_logits = []
            current_labels = []
            
            for batch, labels in tqdm(loader, desc=f"TTA {t_idx + 1}/{N_tta}", leave=False):
                batch = batch.to(device)
                logits = model(batch)
                
                # Move to CPU immediately to prevent VRAM overflow
                model_logits.append(logits.cpu())
                
                # We only need to collect the true labels on the first pass
                if t_idx == 0:
                    current_labels.append(labels.clone())
                    
            # Stack logits for this specific transform
            all_tta_logits.append(torch.cat(model_logits, dim=0))
            
            if t_idx == 0:
                val_labels = torch.cat(current_labels, dim=0)
                
    # Return as (N_tta, N_samples, num_classes)
    return torch.stack(all_tta_logits, dim=0), val_labels

def train_weights(
    model_name: str,
    input: torch.Tensor,
    val_labels: torch.Tensor,
    logit: bool = True,
    n_epochs: int = 200,
    lr: float = 1e-2,
    device: str = 'cpu',
):
    """
    input : (N_input, N_samples, num_classes) can be either TTA logits (N_tta, N_samples, num_classes) or expert probs (N_experts, N_samples, num_classes)
    val_labels           : (N_samples,)
    
    Retourne LearnedWeightedMean entraîné
    """
    
    # logits : (N_tta, N_samples, num_classes) - fixed input data
    # val_labels : (N_samples,) - targets
    input = input.to(device).detach()  # Ensure on device and detached
    val_labels = val_labels.to(device).long()  # Ensure correct dtype and device
    
    N_input = input.shape[0]
    model = LearnedWeightedMean(n_input=N_input).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # logits déjà précalculés → entraînement très rapide
    with torch.enable_grad():
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # combined : (N_samples, num_classes)
            combined = model(list(input))
            if logit:
                loss = F.cross_entropy(combined, val_labels)
            else: # input is already probabilities, so we take log for NLL loss
                true_class_probs = combined[range(combined.shape[0]), val_labels]
                loss = -torch.log(true_class_probs + 1e-9).mean()
            
            loss.backward()
            optimizer.step()
    
    # Affiche les poids finaux pour inspection
    final_weights = F.softmax(model.weights, dim=0)
    print(f"{model_name} Learned weights: {final_weights.detach().cpu().numpy()}")
    
    return model

def train_expert_attention_meta(
    logits: torch.Tensor,
    val_labels: torch.Tensor,
    n_epochs: int = 200,
    lr: float = 1e-2,
    device: str = 'cpu',
):
    """
    logits : (N_experts, N_samples, num_classes)
    val_labels : (N_samples,)
    
    Retourne ExpertAttentionMeta entraîné
    """
    
    logits = logits.to(device).detach()  # Ensure on device and detached
    val_labels = val_labels.to(device).long()  # Ensure correct dtype and device
    
    N_experts = logits.shape[0]
    meta_model = ExpertAttentionMeta(n_experts=N_experts).to(device)
    optimizer = torch.optim.AdamW(meta_model.parameters(), lr=lr, weight_decay=1e-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    
    with torch.enable_grad():
        iteration_bar = trange(n_epochs, desc="Training Attention")
        for epoch in iteration_bar:
            optimizer.zero_grad()
            
            combined = meta_model(logits)
            loss = F.cross_entropy(combined, val_labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            iteration_bar.set_postfix({"loss": loss.item()})

    meta_model.eval()
    
    return meta_model

def predict_with_tta(
    model,
    frames_pil,
    tta_transforms,
    tta_module: LearnedWeightedMean,
    device,
):
    model.eval()
    logits_list = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            tensor = transform(frames_pil).unsqueeze(0).to(device)
            logits_list.append(model(tensor))
    
    return tta_module(logits_list)  # (1, num_classes)

def _xgboost_stratified_kfold_cv(
    model_template: XGBClassifier,
    X_meta: np.ndarray,
    y_true: np.ndarray,
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
        'max_depth': 2,
        'learning_rate': 0.04,
        'n_estimators': 500,
        'subsample': 0.65,
        'colsample_bytree': 0.5,
        'min_child_weight': 5,
        'lambda': 20.0,
        'alpha': 2.0,
        'early_stopping_rounds': 30,
        'objective': 'multi:softprob',
        'random_state': 42,
        'tree_method': 'hist' if torch.cuda.is_available() else 'auto',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'verbosity': 0,
    }
    
    # Merge user hyperparams with defaults (user params override)
    defaults.update(hyperparams)
    return XGBClassifier(**defaults)


def _optimize_xgboost_hyperparams_optuna(
    X_meta: np.ndarray, 
    y_true: np.ndarray, 
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
            # ==========================================
            # STRUCTURAL PRIORS (Hardcoded Defenses)
            # ==========================================
            'max_depth': 2,
            'subsample': 0.65,
            'lambda': 20.0,
            'alpha': 2.0,
            'early_stopping_rounds': 30,
            
            # ==========================================
            # OPTUNA SEARCH SPACE (Only 5 Dimensions!)
            # ==========================================
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),
            'gamma': trial.suggest_float('gamma', 1.0, 5.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.6),
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
            intermediate_accuracy = np.mean(fold_scores)
            trial.report(intermediate_accuracy, step=fold_idx)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(fold_scores)
    
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

def find_best_alpha(
    lr_probs: np.ndarray,      # (N_samples, num_classes)
    xgb_probs: np.ndarray,     # (N_samples, num_classes)
    val_labels: np.ndarray,    # (N_samples,)
    n_points: int = 21,
    verbose: bool = True
) -> tuple[float, float]:
    
    best_alpha, best_acc = 0.5, 0.0
    
    for alpha in np.linspace(0, 1, n_points):
        combined = alpha * lr_probs + (1 - alpha) * xgb_probs
        acc = (combined.argmax(1) == val_labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
    
    if verbose:
        print(f"Best alpha: {best_alpha:.2f} (LR weight)")
        print(f"Best acc  : {best_acc:.4f}")
        print(f"LR alone  : {(lr_probs.argmax(1) == val_labels).mean():.4f}")
        print(f"XGB alone : {(xgb_probs.argmax(1) == val_labels).mean():.4f}")
    
    return best_alpha, best_acc

class CombinedLRAttentionModel:
    """
    Wrapper that combines LogisticRegression and ExpertAttentionMeta predictions
    using a learnable alpha parameter.
    """
    def __init__(self, lr_model, attention_model, alpha: float, num_classes: int, num_experts: int):
        self.lr_model = lr_model
        self.attention_model = attention_model
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_experts = num_experts
    
    def predict(self, X):
        return self.predict_proba(X).argmax(1)
    
    def predict_proba(self, X):
        lr_probs_raw = self.lr_model.predict_proba(X)
        lr_probs = self.lr_model.predict_proba(X)
        
        # Pad Logistic Regression output to 33 columns in case a class is entirely missing
        # Reshape X for the Attention Model
        lr_probs = np.zeros((X.shape[0], self.num_classes))
        lr_probs[:, self.lr_model.classes_] = lr_probs_raw
        
        # Reshape X for the Attention Model
        X_tensor = torch.tensor(X, dtype=torch.float32).reshape(
            X.shape[0], self.num_experts, self.num_classes
        ).transpose(0, 1)
        
        self.attention_model.eval()
        with torch.no_grad():
            attention_probs = self.attention_model(X_tensor).numpy()
            
        return self.alpha * lr_probs + (1 - self.alpha) * attention_probs

@torch.no_grad()
def evaluate_and_stack_n_models(
    ckpt_paths: List[str], 
    val_dir: str,
    meta_learner: str = 'logistic',
    use_bayesian_optimization: bool = False,
    bayesian_optimization_trials: int = 50,
    use_cache: bool = True,
    TTA: bool = False,
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
            - 'logistic': LogisticRegression with L2 regularization (C=0.1)
            - 'xgboost': XGBClassifier with gradient boosting (default max_depth=4, learning_rate=0.05)
            - 'both': Trains both XGBoost (without BO) and LogisticRegression, combines with find_best_alpha
        use_bayesian_optimization: If True and meta_learner='xgboost', use Optuna to tune XGBoost hyperparams
            via 5-fold CV. Ignored if meta_learner='logistic' or 'both'. Default: False (uses tuned default params)
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
        
        # Use combined XGBoost + LogisticRegression
        meta_model = evaluate_and_stack_n_models(
            ckpt_paths=my_models,
            val_dir=\"processed_data/val2/val\",
            meta_learner='both'
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
    tta_modules = []
    y_true = np.array([sample[1] for sample in val_samples])

    # ---------------------------------------------------------
    # PART 1: SEQUENTIAL FEATURE EXTRACTION (VRAM SAFE)
    # ---------------------------------------------------------
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\n--- Processing Expert {i+1}/{len(ckpt_paths)}: {Path(ckpt_path).name} ---")
        
        # 0. Check cache first
        cache_path = _get_cache_path(ckpt_path, val_dir, TTA) if use_cache else None
        weights_cache_path = _get_tta_weights_cache_path(ckpt_path, val_dir) if use_cache and TTA else None
        if use_cache and cache_path.exists():
            # If TTA is active but we lost the weights cache, force a recompute
            if TTA and not weights_cache_path.exists():
                print("WARNING: Cached logits found but TTA weights missing! Recomputing...")
            else:
                print(f"Loading cached logits from {cache_path.name}...")
                expert_probs = np.load(cache_path)
                all_expert_probs.append(expert_probs)
                
                # --- THE FIX: Hydrate the TTA Module ---
                if TTA:
                    print(f"Loading cached TTA weights from {weights_cache_path.name}...")
                    # Initialize a blank module with our 6 standard transforms
                    tta_module = LearnedWeightedMean(n_input=6) 
                    tta_module.load_state_dict(torch.load(weights_cache_path, map_location='cpu'))
                    tta_modules.append(tta_module)
                else:
                    tta_modules.append(None)
                
                print(f"Expert {i+1} accuracy on Validation Set: {accuracy_score(y_true, np.argmax(all_expert_probs[-1], axis=1)):.4f}")
                continue
        
        # 1. Load Model dynamically
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = OmegaConf.create(ckpt["config"])
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        if TTA:
            print("Precomputing TTA weights for this expert...")
            
            # 1. Get raw logits for all 6 transforms
            tta_logits, tta_labels = precompute_tta_logits(model, val_dir, val_samples, cfg, device)
            
            # 2. Train the TTA blending weights (Fast CPU optimization)
            model_name = Path(ckpt_path).stem
            tta_module = train_weights(model_name, tta_logits, tta_labels, n_epochs=200, lr=1e-2)
            
            # 3. Apply the learned weights to get the final blended logits
            tta_module.eval()
            with torch.no_grad():
                # list(tta_logits) unbinds dimension 0 into a list of 6 tensors
                combined_logits = tta_module(list(tta_logits))
                
            # 4. Convert to probabilities for XGBoost/Logistic Regression
            expert_probs = torch.softmax(combined_logits, dim=1).numpy()
            
            tta_modules.append(tta_module)
            
            if use_cache and weights_cache_path:
                torch.save(tta_module.state_dict(), weights_cache_path)
                print(f"Cached TTA weights to {weights_cache_path.name}")
        else:
            tta_modules.append(None)
            # 2. Setup Dataloader specific to this model's config
            use_imagenet_norm = cfg.model.get("pretrained", False)
            transform = VideoTransform(cfg, is_training=False, use_imagenet_norm=use_imagenet_norm)
            
            dataset = VideoFrameDataset(
                root_dir=val_dir, 
                num_frames=int(cfg.dataset.get("num_frames", 4)),
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

            # Stack this expert's logits and save to our master list
            expert_probs = np.vstack(model_probs)

        all_expert_probs.append(expert_probs)
        
        # Cache the logits for future runs
        if use_cache and cache_path:
            np.save(cache_path, expert_probs)
            print(f"Cached logits to {cache_path.name}")
        
        print(f"Expert {i+1} accuracy on Validation Set: {accuracy_score(y_true, np.argmax(all_expert_probs[-1], axis=1)):.4f}")

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
    X_meta = np.hstack(all_expert_probs)
    
    # ==========================================
    # FIX: SQUASH LABEL GAPS FOR XGBOOST (class 27 missing)
    # ==========================================
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    
    print(f"Combined Feature Shape: {X_meta.shape}")
    print(f"Original Unique Classes: {len(np.unique(y_true))}")
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
        # C=0.3 applies strong L2 regularization to prevent validation memorization
        meta_model = LogisticRegression(max_iter=2000, C=0.3, class_weight=None)
        
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
            meta_model = _create_xgboost_model({'learning_rate': 0.048939493196935975, 'n_estimators': 567, 'min_child_weight': 9, 'gamma': 1.2776574578150808, 'colsample_bytree': 0.20371327111730006})
            print(f"Default Hyperparameters:")
            print(f"   max_depth: 2")
            print(f"   learning_rate: 0.04")
            print(f"   n_estimators: 500")
            print(f"   subsample: 0.65")
            print(f"   colsample_bytree: 0.5")
            print(f"   lambda (L2): 20")
            print(f"   alpha (L1): 2")
    
    elif meta_learner == 'both':
        print("\n" + "="*50)
        print("Meta-Learner: Combined Attention + LogisticRegression")
        print("="*50)
        print("\nWill train both models and combine with find_best_alpha")

        # xgb_params = {'max_depth': 3, 'learning_rate': 0.047835958212693014, 'n_estimators': 498, 'subsample': 0.6943002505190707, 'colsample_bytree': 0.4224184464008965, 'min_child_weight': 9, 'lambda': 21.58031280816208, 'alpha': 1.62159974968791} # 49.73%
        # xgb_params = {'max_depth': 3, 'learning_rate': 0.041457793421589034, 'n_estimators': 465, 'subsample': 0.6584760556126513, 'colsample_bytree': 0.5170739168034588, 'min_child_weight': 14, 'lambda': 19.219909328630457, 'alpha': 1.52774573314708851} # 49.64%
        # xgb_params = {'learning_rate': 0.048939493196935975, 'n_estimators': 567, 'min_child_weight': 9, 'gamma': 1.2776574578150808, 'colsample_bytree': 0.20371327111730006}
    
    # Run the Cross Validation
    print("\nEvaluating Meta-Learner with 5-Fold Stratified Cross-Validation...")
    
    if meta_learner == 'xgboost':
        # Unpack both the scores and the iterations
        cv_scores_list, best_iters = _xgboost_stratified_kfold_cv(
            meta_model, X_meta, y_true_encoded, n_splits=5
        )
        cv_scores = np.array(cv_scores_list)
        
        # Calculate the magical "Blind Run" target
        optimal_trees = int(np.mean(best_iters) * 1.05)  # Add 5% buffer to prevent underfitting
        print(f"\n🧠 Calculated Optimal Trees from CV: {optimal_trees}")
    elif meta_learner == 'weighted_mean' or meta_learner == 'attention':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_list = []
        
        # Calculate number of classes per expert to reshape flattened features
        num_classes = X_meta.shape[1] // len(ckpt_paths)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_meta, y_true_encoded)):
            X_train_np, X_val_np = X_meta[train_idx], X_meta[val_idx]
            y_train, y_val = y_true[train_idx], y_true[val_idx]

            # Convert numpy arrays to torch tensors and reshape for LearnedWeightedMean
            # X_train_np: (N_train, N_experts * num_classes) -> (N_experts, N_train, num_classes)
            X_train = torch.tensor(X_train_np, dtype=torch.float32).reshape(
                X_train_np.shape[0], len(ckpt_paths), num_classes
            ).transpose(0, 1)
            y_train_torch = torch.tensor(y_train, dtype=torch.long)
            
            if meta_learner == 'weighted_mean':
                fold_model: LearnedWeightedMean = train_weights("WeightedMean", X_train, y_train_torch, logit=False, n_epochs=200, lr=1e-2)
            else:  # meta_learner == 'attention'
                fold_model: ExpertAttentionMeta = train_expert_attention_meta(X_train, y_train_torch, n_epochs=100, lr=2e-2)

            # Convert validation data to torch tensors and reshape
            # X_val_np: (N_val, N_experts * num_classes) -> (N_experts, N_val, num_classes)
            X_val = torch.tensor(X_val_np, dtype=torch.float32).reshape(
                X_val_np.shape[0], len(ckpt_paths), num_classes
            ).transpose(0, 1)
            if meta_learner == 'weighted_mean':
                X_val = list(X_val)  # LearnedWeightedMean expects a list of tensors per expert
            
            fold_model.eval()
            # Forward pass returns (N_val, num_classes)
            val_probs = fold_model(X_val).numpy()
            val_preds = val_probs.argmax(1)
            fold_acc = (val_preds == y_val).mean()
            cv_scores_list.append(fold_acc.item())
        
        cv_scores = np.array(cv_scores_list)

    elif meta_learner == 'both':
        # Manual k-fold CV for combined model
        combined_scores = []
        lr_only_scores = []
        attention_only_scores = []
        best_alphas = []
        
        num_experts = len(ckpt_paths)
        num_classes = X_meta.shape[1] // num_experts
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_meta, y_true)):
            X_train, X_val = X_meta[train_idx], X_meta[val_idx]
            y_train, y_val = y_true[train_idx], y_true[val_idx]
            
            # Train LR on this fold
            lr_fold = LogisticRegression(max_iter=2000, C=0.3, class_weight=None)
            lr_fold.fit(X_train, y_train)
            
            # Reshape for Attention
            X_train_attention = torch.tensor(X_train, dtype=torch.float32).reshape(
                X_train.shape[0], num_experts, num_classes
            ).transpose(0, 1)
            y_train_torch = torch.tensor(y_train, dtype=torch.long)
            
            # Train Attention Model
            attention_fold: ExpertAttentionMeta = train_expert_attention_meta(
                X_train_attention, y_train_torch, n_epochs=100, lr=2e-2
            )

            # Reshape Val for Attention
            X_val_attention = torch.tensor(X_val, dtype=torch.float32).reshape(
                X_val.shape[0], num_experts, num_classes
            ).transpose(0, 1)
    
            attention_fold.eval()
            with torch.no_grad():
                attention_probs = attention_fold(X_val_attention).numpy()
            
            # Extract the raw probabilities (e.g. 32 columns if 1 class is missing)
            lr_probs_raw = lr_fold.predict_proba(X_val)

            # Pad it safely to 33 columns based on the explicitly seen classes
            lr_probs = np.zeros((X_val.shape[0], num_classes))
            lr_probs[:, lr_fold.classes_] = lr_probs_raw
            
            # Combine using find_best_alpha
            best_alpha, best_acc = find_best_alpha(lr_probs, attention_probs, y_val, verbose=False)
            
            lr_acc = (lr_probs.argmax(1) == y_val).mean()
            attention_acc = (attention_probs.argmax(1) == y_val).mean()
            
            combined_scores.append(best_acc)
            lr_only_scores.append(lr_acc)
            attention_only_scores.append(attention_acc)
            best_alphas.append(best_alpha)
            
            print(f"   Fold {fold_idx + 1}: Combined={best_acc:.4f}, LR={lr_acc:.4f}, Attention={attention_acc:.4f}, α={best_alpha:.2f}")
        
        cv_scores = np.array(combined_scores)
        lr_only_scores = np.array(lr_only_scores)
        attention_only_scores = np.array(attention_only_scores)
        best_alphas = np.array(best_alphas)
        
        print(f"\n   LR Only Mean:  {lr_only_scores.mean():.4f} (±{lr_only_scores.std():.4f})")
        print(f"   Attention Only Mean: {attention_only_scores.mean():.4f} (±{attention_only_scores.std():.4f})")
        print(f"   Avg Best Alpha: {best_alphas.mean():.2f}")
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
    elif meta_learner == 'weighted_mean' or meta_learner == 'attention':
        # Train final LearnedWeightedMean on 100% of validation data
        # Reshape X_meta for LearnedWeightedMean: (N_samples, N_experts * num_classes) -> (N_experts, N_samples, num_classes)
        num_classes = X_meta.shape[1] // len(ckpt_paths)
        X_meta_tensor = torch.tensor(X_meta, dtype=torch.float32).reshape(
            X_meta.shape[0], len(ckpt_paths), num_classes
        ).transpose(0, 1)
        y_meta_tensor = torch.tensor(y_true, dtype=torch.long)
        
        if meta_learner == 'weighted_mean':
            meta_model = train_weights("WeightedMean_Final", X_meta_tensor, y_meta_tensor, logit=False, n_epochs=200, lr=1e-2)
        elif meta_learner == 'attention':
            meta_model = train_expert_attention_meta(X_meta_tensor, y_meta_tensor, n_epochs=100, lr=2e-2)
        print(f"Final Model: Trained on 100% of validation data with learned weighted averaging")
    elif meta_learner == 'both':
        num_experts = len(ckpt_paths)
        num_classes = X_meta.shape[1] // num_experts
        
        # Train LR on full data
        lr_final = LogisticRegression(max_iter=2000, C=0.3, class_weight=None)
        lr_final.fit(X_meta, y_true)
        
        # Train Attention on full data
        X_meta_tensor = torch.tensor(X_meta, dtype=torch.float32).reshape(
            X_meta.shape[0], num_experts, num_classes
        ).transpose(0, 1)
        y_meta_tensor = torch.tensor(y_true, dtype=torch.long)
        
        attention_final = train_expert_attention_meta(X_meta_tensor, y_meta_tensor, n_epochs=100, lr=2e-2)
        
        # Use the true, unbiased mean alpha from the Out-Of-Fold CV
        best_alpha_final = best_alphas.mean()
              
        # Create the combined model
        meta_model = CombinedLRAttentionModel(lr_final, attention_final, best_alpha_final, num_classes, num_experts)
        
        print(f"Final Model: Combined LR+Attention with α={best_alpha_final:.2f}")
    else:
        # LogisticRegression on full validation set
        meta_model.fit(X_meta, y_true_encoded)
        print(f"Final Model: Trained on 100% of validation data")

    return meta_model, le, tta_modules  # Return the label encoder for decoding predictions later

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================
    # CONFIGURATION: Define your Kaggle Roster here
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
        "checkpoints/efficientnetb0_6chan_39-93.pt",
        "checkpoints/efficientnetb0_tdn_40-13.pt",
        "checkpoints/efficientformer_tsm_attn_35-67.pt",
        "checkpoints/coatnet_tsm_37-21.pt",
        "checkpoints/mae_small_phase2_22-22.pt",
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
    # meta_model, _, tta_modules = evaluate_and_stack_n_models(
    #     ckpt_paths=my_models,
    #     val_dir=val_dir,
    #     meta_learner='logistic',
    #     TTA=True,
    # )
    
    # Example 2: XGBoost with default hyperparameters (tuned for 264 features)
    # Faster than Bayesian optimization, uses good defaults
    # meta_model, _, tta_modules = evaluate_and_stack_n_models(
    #     ckpt_paths=my_models,
    #     val_dir=val_dir,
    #     meta_learner='xgboost',
    #     use_bayesian_optimization=False,
    #     bayesian_optimization_trials=75,
    #     use_cache=True  # Set to False to recompute logits
    # )
    
    # Example 3: Combined Attention + LogisticRegression
    # Trains both models and combines with find_best_alpha
    meta_model, _, tta_modules = evaluate_and_stack_n_models(
        ckpt_paths=my_models,
        val_dir=val_dir,
        meta_learner='both',
        use_cache=True,
        TTA=True
    )
    
    # Example 4: Simple Weighted Mean (simple gradient-based meta-learner)
    # meta_model, _, _ = evaluate_and_stack_n_models(
    #     ckpt_paths=my_models,
    #     val_dir=val_dir,
    #     meta_learner='weighted_mean',
    #     use_cache=True,
    #     TTA=True
    # )
    
    # Example 5: Expert Attention Meta-Learner (more complex gradient-based meta-learner)
    # meta_model, _, _ = evaluate_and_stack_n_models(
    #     ckpt_paths=my_models,
    #     val_dir=val_dir,
    #     meta_learner='attention',
    #     use_cache=True,
    #     TTA=True
    # )

if __name__ == "__main__":
    main()