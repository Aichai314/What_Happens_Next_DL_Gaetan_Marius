#!/usr/bin/env python3
"""
Last-day Kaggle search.

Premises (encoded as choices below — read the source):
    1. t16 solo on Kaggle = 70.78%, the 6-model weighted_mean = 70.11%, power_mean
       = 69.56%. Adding weak models DILUTES the strong signal. So we search over
       SUBSETS of {tdn_ft, v1, t16, videomae_k, videomae_s} (no specialist —
       see #2) and fusion modes (mean / geom_mean / weighted_mean / power_mean),
       plus an optional t16-weight boost ∈ {1, 2, 3, 5} for weighted_mean only.
    2. The specialist is excluded from the main fusion: its 9-class Bayes factor
       made every '_spec' variant lose ~4 pts on val2/val (the 8-9x boost on
       its target classes amplifies its mistakes). It can be re-tested later.
    3. Two val sets are used jointly to mitigate leakage:
         - val2/val : 4 fixed frames extracted by the challenge organisers.
           Closest distribution to Kaggle test, but v1 + tdn_ft were SELECTED on
           this → their softmax is inflated here.
         - ssv2_32f/val : 20 frames/video, we sample 4 RANDOM frames per video
           (seeded). Closer to "unseen 4-frame combination" but t16 + videomae
           were SELECTED on this → THEIR softmax is inflated here.
       Variants that win on BOTH are likely robust to the leakage direction.
    4. XGBoost is added with hard anti-overfit settings AND cross-val-set
       generalisation as the metric (train on val A, test on val B). The
       previous XGBoost run lost 5.7pts CV→Kaggle on val2/val alone; the
       cross-set metric is the real safety check.

Outputs three CSVs in submissions/:
    ensemble_submission_search_best.csv      — best closed-form variant
    ensemble_submission_search_xgb.csv        — XGBoost trained on val2/val
    ensemble_submission_search_safe.csv       — geom_mean({t16, v1}) safety net
"""

import csv
import gc
import hashlib
import itertools
import random
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from train import build_model
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import VideoTransform

from create_submission import (
    discover_all_test_videos, load_manifest_video_names, resolve_video_dirs,
)
from create_submission_ensemble_marius import (
    SPECIALIST_CLASSES, NUM_CHALLENGE_CLASSES,
    _pick_batch_size, _gpu_is_contested,
    _cache_path as _test_cache_path,
    extract_one_model_probs as extract_test_one_model,
    fuse,
)

# Friendly name → checkpoint path.
MODELS: List[Tuple[str, str]] = [
    ("tdn_ft",     "/Data/marius.truquin/Model_checkpoints/best_model_tdn_ssv2_ft.pt"),
    ("v1",         "/Data/marius.truquin/Model_checkpoints/best_model_vjepa2_large_v1.pt"),
    ("t16",        "/Data/marius.truquin/Model_checkpoints/best_model_vjepa2_large_t16.pt"),
    ("videomae_k", "/Data/marius.truquin/Model_checkpoints/best_model_videomae_large_kinetics.pt"),
    ("videomae_s", "/Data/marius.truquin/Model_checkpoints/best_model_videomae_v2_ssv2_2.pt"),
]
T16_IDX = next(i for i, (n, _) in enumerate(MODELS) if n == "t16")

VAL_DIRS = {
    "val2":     "processed_data/val2/val",
    "ssv2_32f": "processed_data/ssv2_32f/val",
}
RANDOM_SEED = 1337


# ────────────────────────────────────────────────────────────────────────────
# Val probability extraction (one cache per (ckpt × val_dir × sampling_mode))
# ────────────────────────────────────────────────────────────────────────────
def _val_cache_path(ckpt_path: str, val_dir: str, random_sampling: bool) -> Path:
    cache_dir = Path(".search_cache")
    cache_dir.mkdir(exist_ok=True)
    tag = "rand4f" if random_sampling else "lin4f"
    key = hashlib.md5(f"{ckpt_path}:{val_dir}:{tag}:seed={RANDOM_SEED}".encode()).hexdigest()
    return cache_dir / f"{Path(ckpt_path).stem}_{key}.npy"


def _val_labels_path(val_dir: str) -> Path:
    cache_dir = Path(".search_cache")
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.md5(val_dir.encode()).hexdigest()
    return cache_dir / f"labels_{key}.npy"


@torch.no_grad()
def extract_val_probs(
    ckpt_path: str,
    val_dir: Path,
    sample_list: List[Tuple[Path, int]],
    device: torch.device,
    random_sampling: bool,
) -> np.ndarray:
    """Extract softmax on a val set; cache per (ckpt, val_dir, sampling)."""
    cache = _val_cache_path(ckpt_path, str(val_dir), random_sampling)
    if cache.exists():
        probs = np.load(cache)
        print(f"  [cache] {cache.name}  shape={probs.shape}")
        return probs

    meta = torch.load(ckpt_path, map_location="cpu")
    cfg = OmegaConf.create(meta["config"])
    model_name = cfg.model.get("name", "")
    is_tdn = "tdn" in model_name.lower()
    is_t16 = "t16" in Path(ckpt_path).stem.lower()

    if is_tdn:
        model_device = torch.device("cpu")
    elif device.type == "cuda" and _gpu_is_contested():
        print(f"  ⚠ GPU contested → CPU for {Path(ckpt_path).stem}")
        model_device = torch.device("cpu")
    else:
        model_device = device

    model = build_model(cfg)
    model.load_state_dict(meta["model_state_dict"])
    model = model.to(model_device).eval()

    use_imagenet_norm = cfg.model.get("pretrained", False)
    image_size = int(cfg.dataset.get("image_size", 224))
    transform = VideoTransform(
        cfg, is_training=False, use_imagenet_norm=use_imagenet_norm, image_size=image_size
    )
    num_frames = int(meta.get("num_frames", cfg.dataset.num_frames))

    # Reseed Python's RNG so VideoFrameDataset's random_sample is reproducible
    # (the dataset uses random.sample under the hood when random_temporal_sampling=True).
    random.seed(RANDOM_SEED)

    dataset = VideoFrameDataset(
        root_dir=val_dir, num_frames=num_frames, transform=transform,
        sample_list=sample_list,
        random_temporal_sampling=random_sampling,
    )
    bs = _pick_batch_size(is_tdn, is_t16, num_frames, image_size)

    # num_workers=0 so the seeded random.sample stays deterministic. With workers,
    # each fork gets its own random state and the cache becomes non-reproducible.
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=0
    )
    print(f"  device={model_device.type} bs={bs} num_frames={num_frames} "
          f"image_size={image_size} random_sampling={random_sampling}")

    all_probs = []
    for batch, _ in tqdm(loader, desc=f"  {Path(ckpt_path).stem}"):
        batch = batch.to(model_device)
        logits = model(batch)
        if model_device.type == "cuda":
            torch.cuda.synchronize()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        if is_t16 and model_device.type == "cuda":
            torch.cuda.empty_cache()

    probs = np.vstack(all_probs)
    np.save(cache, probs)
    print(f"  [cache] saved {cache.name}")

    del model, meta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs


# ────────────────────────────────────────────────────────────────────────────
# Closed-form variant search
# ────────────────────────────────────────────────────────────────────────────
def search_closed_form(
    probs_by_model: List[np.ndarray],   # (N, 33) per model — main experts only
    val_acc_by_model: List[float],
    labels_a: np.ndarray, labels_b: np.ndarray,
    probs_a: List[np.ndarray], probs_b: List[np.ndarray],
) -> List[dict]:
    """Try every (roster_subset, mode, t16_boost). Score on both val sets.

    roster_subset is a tuple of model indices; we require t16 in every subset
    (we never want to throw away the strongest expert).
    """
    n_models = len(probs_by_model)
    results = []
    modes = ["mean", "geom_mean", "weighted_mean", "power_mean"]
    boosts = [1.0, 2.0, 3.0, 5.0]

    indices = list(range(n_models))
    subsets = []
    for r in range(1, n_models + 1):
        for combo in itertools.combinations(indices, r):
            if T16_IDX in combo:
                subsets.append(combo)

    for subset in subsets:
        for mode in modes:
            # t16 boost only meaningful for weighted_mean
            boost_list = boosts if mode == "weighted_mean" else [1.0]
            for boost in boost_list:
                # Build per-subset arrays for both val sets
                p_a = [probs_a[i] for i in subset]
                p_b = [probs_b[i] for i in subset]
                w   = [val_acc_by_model[i] for i in subset]
                w   = [(wt * boost if i == T16_IDX else wt) for i, wt in zip(subset, w)]
                is_spec = [False] * len(subset)
                fused_a = fuse(p_a, is_spec, w, mode)
                fused_b = fuse(p_b, is_spec, w, mode)
                acc_a = float((fused_a.argmax(1) == labels_a).mean())
                acc_b = float((fused_b.argmax(1) == labels_b).mean())
                # combined: minimum of the two (robust — winner must hold on BOTH)
                combined = min(acc_a, acc_b)
                results.append({
                    "subset_idx": subset,
                    "mode": mode,
                    "t16_boost": boost,
                    "acc_val2": acc_a,
                    "acc_ssv2": acc_b,
                    "min_acc": combined,
                })
    results.sort(key=lambda r: -r["min_acc"])
    return results


def _apply_to_test(
    probs_by_model_test: List[np.ndarray],
    val_acc_by_model: List[float],
    subset: Tuple[int, ...],
    mode: str,
    t16_boost: float,
) -> np.ndarray:
    p = [probs_by_model_test[i] for i in subset]
    w = [val_acc_by_model[i] for i in subset]
    w = [(wt * t16_boost if i == T16_IDX else wt) for i, wt in zip(subset, w)]
    is_spec = [False] * len(subset)
    return fuse(p, is_spec, w, mode)


# ────────────────────────────────────────────────────────────────────────────
# XGBoost cross-val-set
# ────────────────────────────────────────────────────────────────────────────
def train_eval_xgb(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval: np.ndarray, y_eval: np.ndarray,
    X_test: np.ndarray,
):
    """Train a hard-regularised XGB on (X_train, y_train); report eval acc; predict test.

    Settings chosen to MINIMISE overfit on small-leaky data:
      max_depth=2     (was 4)  : minimal interactions, less mem of train rows
      n_estimators=100 (was 500): fewer rounds, no early stopping needed
      lambda=5.0       (was 1.0): strong L2
      alpha=2.0        (was 0.2): strong L1
      learning_rate=0.05         : low; combined with n_estim=100 gives mild fit
      subsample=0.7              : row sampling
      colsample_bytree=0.7       : feature sampling
    Class labels: encoded contiguous because challenge class 27 is absent.
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    # Map eval labels through the same encoder; unseen classes (shouldn't happen
    # on these val sets) get -1 and will silently miss-classify.
    y_eval_enc = np.array([
        np.where(le.classes_ == y)[0][0] if y in le.classes_ else -1 for y in y_eval
    ])

    clf = XGBClassifier(
        max_depth=2,
        learning_rate=0.05,
        n_estimators=100,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=2,
        reg_lambda=5.0,
        reg_alpha=2.0,
        objective="multi:softprob",
        random_state=42,
        tree_method="hist" if torch.cuda.is_available() else "auto",
        verbosity=0,
    )
    w_tr = compute_sample_weight("balanced", y=y_train_enc)
    clf.fit(X_train, y_train_enc, sample_weight=w_tr, verbose=False)

    eval_pred = clf.predict(X_eval)
    eval_acc = float((eval_pred == y_eval_enc).mean())
    test_pred = le.inverse_transform(clf.predict(X_test))
    return eval_acc, test_pred


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if _gpu_is_contested():
            print("⚠ GPU contested — fragile models will route to CPU.")
        else:
            print("✓ GPU free for fragile models.")

    test_root = Path(cfg.dataset.test_dir).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest")
    if manifest_cfg:
        manifest_path = Path(str(manifest_cfg)).resolve()
        video_names = load_manifest_video_names(manifest_path)
        video_dirs = resolve_video_dirs(test_root, video_names)
    else:
        video_names, video_dirs = discover_all_test_videos(test_root)
    print(f"Test set: {len(video_dirs)} videos\n")

    # ── 1. Extract / load test probs (cached by create_submission_ensemble_marius) ──
    print("=" * 60)
    print("STEP 1 — Test probabilities (per main model)")
    print("=" * 60)
    test_probs: List[np.ndarray] = []
    val_acc_stored: List[float] = []
    for name, ckpt in MODELS:
        print(f"\n--- {name}  ({Path(ckpt).name}) ---")
        probs, va = extract_test_one_model(ckpt, test_root, video_dirs, device)
        if probs.shape[1] != NUM_CHALLENGE_CLASSES:
            raise RuntimeError(f"{name} is not 33-class — refusing to mix.")
        test_probs.append(probs)
        val_acc_stored.append(va)
        print(f"  val_acc(stored)={va:.4f}  shape={probs.shape}")

    # ── 2. Val sets ──
    val_probs_by_set: dict = {}
    val_labels_by_set: dict = {}
    for set_name, raw_path in VAL_DIRS.items():
        print("\n" + "=" * 60)
        print(f"STEP 2.{set_name} — Val probabilities  ({raw_path})")
        print("=" * 60)
        val_dir = Path(raw_path).resolve()
        # ssv2_32f → random 4-frame sampling (videos have 20 frames). val2 is
        # already only 4 frames; sampling=True is a no-op there (the dataset
        # falls back to linspace when num_available <= num_frames).
        random_sampling = (set_name == "ssv2_32f")
        sample_list = collect_video_samples(val_dir)
        # cache labels
        labels_path = _val_labels_path(str(val_dir))
        if labels_path.exists():
            labels = np.load(labels_path)
        else:
            labels = np.array([s[1] for s in sample_list], dtype=np.int64)
            np.save(labels_path, labels)
        print(f"{len(sample_list)} videos, {len(np.unique(labels))} classes present.\n")

        probs_list = []
        for name, ckpt in MODELS:
            print(f"--- {name} on {set_name} ---")
            probs = extract_val_probs(ckpt, val_dir, sample_list, device, random_sampling)
            probs_list.append(probs)
            acc = float((probs.argmax(1) == labels).mean())
            print(f"  solo acc on {set_name} = {acc*100:.2f}%")
        val_probs_by_set[set_name] = probs_list
        val_labels_by_set[set_name] = labels

    # ── 3. Solo accuracies summary ──
    print("\n" + "=" * 60)
    print("STEP 3 — Solo accuracies on both val sets")
    print("=" * 60)
    print(f"{'model':<14} {'ckpt.val_acc':>13} {'val2/val':>10} {'ssv2_32f':>10}")
    for i, (name, _) in enumerate(MODELS):
        a = float((val_probs_by_set["val2"][i].argmax(1) == val_labels_by_set["val2"]).mean())
        b = float((val_probs_by_set["ssv2_32f"][i].argmax(1) == val_labels_by_set["ssv2_32f"]).mean())
        print(f"{name:<14} {val_acc_stored[i]:>13.4f} {a*100:>9.2f}% {b*100:>9.2f}%")

    # ── 4. Closed-form search (subset × mode × t16_boost), scored on BOTH sets ──
    print("\n" + "=" * 60)
    print("STEP 4 — Closed-form search (require winner on BOTH val sets)")
    print("=" * 60)
    # Use val2 solo acc as the per-model weight (it's the closest distribution
    # to Kaggle test; the val accuracy on ssv2_32f is more inflated for the
    # SSv2-pretrained models, so picking val2 as the weight source is safer).
    weights_for_search = [
        float((val_probs_by_set["val2"][i].argmax(1) == val_labels_by_set["val2"]).mean())
        for i in range(len(MODELS))
    ]
    results = search_closed_form(
        probs_by_model=test_probs,  # unused in this fn except for shape; we pass val below
        val_acc_by_model=weights_for_search,
        labels_a=val_labels_by_set["val2"], labels_b=val_labels_by_set["ssv2_32f"],
        probs_a=val_probs_by_set["val2"], probs_b=val_probs_by_set["ssv2_32f"],
    )

    print(f"\nTop 12 variants by min(acc_val2, acc_ssv2_32f):")
    print(f"{'rank':>4} {'subset':<35} {'mode':<14} {'boost':>5} {'val2':>7} {'ssv2':>7} {'min':>7}")
    for rk, r in enumerate(results[:12]):
        names = ",".join(MODELS[i][0] for i in r["subset_idx"])
        print(f"{rk+1:>4} {names:<35} {r['mode']:<14} {r['t16_boost']:>5.1f} "
              f"{r['acc_val2']*100:>6.2f}% {r['acc_ssv2']*100:>6.2f}% {r['min_acc']*100:>6.2f}%")

    # ── 5. XGBoost cross-val-set ──
    print("\n" + "=" * 60)
    print("STEP 5 — XGBoost (hard-regularised) trained on val2/val,")
    print("         evaluated on ssv2_32f/val for honest generalisation")
    print("=" * 60)
    X_train = np.hstack(val_probs_by_set["val2"])
    y_train = val_labels_by_set["val2"]
    X_eval = np.hstack(val_probs_by_set["ssv2_32f"])
    y_eval = val_labels_by_set["ssv2_32f"]
    X_test = np.hstack(test_probs)
    xgb_eval_acc, xgb_test_pred = train_eval_xgb(X_train, y_train, X_eval, y_eval, X_test)
    print(f"\nXGB acc on ssv2_32f/val (trained on val2/val) = {xgb_eval_acc*100:.2f}%")
    # also the reverse direction for symmetry
    X_train2, y_train2 = X_eval, y_eval
    X_eval2, y_eval2 = X_train, y_train
    xgb_eval_acc2, xgb_test_pred2 = train_eval_xgb(X_train2, y_train2, X_eval2, y_eval2, X_test)
    print(f"XGB acc on val2/val (trained on ssv2_32f/val) = {xgb_eval_acc2*100:.2f}%")

    # ── 6. Write 3 CSVs ──
    out_dir = Path("submissions")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Sub 1: best closed-form
    best = results[0]
    sub1_probs = _apply_to_test(test_probs, weights_for_search,
                                 best["subset_idx"], best["mode"], best["t16_boost"])
    sub1_preds = sub1_probs.argmax(1)
    sub1_path = out_dir / "ensemble_submission_search_best.csv"
    with sub1_path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_name", "predicted_class"])
        for n, p in zip(video_names, sub1_preds): w.writerow([n, int(p)])
    bnames = ",".join(MODELS[i][0] for i in best["subset_idx"])
    print(f"\n✅ Sub 1 (best closed-form): {sub1_path}")
    print(f"    {bnames}  mode={best['mode']}  t16_boost={best['t16_boost']}  "
          f"val2={best['acc_val2']*100:.2f}%  ssv2={best['acc_ssv2']*100:.2f}%")

    # --- Sub 2: XGBoost (whichever direction generalised better)
    if xgb_eval_acc >= xgb_eval_acc2:
        sub2_preds = xgb_test_pred
        sub2_meta = f"trained on val2, eval on ssv2={xgb_eval_acc*100:.2f}%"
    else:
        sub2_preds = xgb_test_pred2
        sub2_meta = f"trained on ssv2, eval on val2={xgb_eval_acc2*100:.2f}%"
    sub2_path = out_dir / "ensemble_submission_search_xgb.csv"
    with sub2_path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_name", "predicted_class"])
        for n, p in zip(video_names, sub2_preds): w.writerow([n, int(p)])
    print(f"✅ Sub 2 (XGBoost cross-set): {sub2_path}")
    print(f"    {sub2_meta}")

    # --- Sub 3: safety net = geom_mean({t16, v1})
    v1_idx = next(i for i, (n, _) in enumerate(MODELS) if n == "v1")
    safe_subset = (T16_IDX, v1_idx)
    safe_probs_test = _apply_to_test(test_probs, weights_for_search, safe_subset, "geom_mean", 1.0)
    safe_preds = safe_probs_test.argmax(1)
    sub3_path = out_dir / "ensemble_submission_search_safe.csv"
    with sub3_path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_name", "predicted_class"])
        for n, p in zip(video_names, safe_preds): w.writerow([n, int(p)])
    # also score the safe on val for context
    safe_a = fuse([val_probs_by_set["val2"][i] for i in safe_subset], [False, False],
                   [weights_for_search[i] for i in safe_subset], "geom_mean")
    safe_b = fuse([val_probs_by_set["ssv2_32f"][i] for i in safe_subset], [False, False],
                   [weights_for_search[i] for i in safe_subset], "geom_mean")
    safe_acc_a = float((safe_a.argmax(1) == val_labels_by_set["val2"]).mean())
    safe_acc_b = float((safe_b.argmax(1) == val_labels_by_set["ssv2_32f"]).mean())
    print(f"✅ Sub 3 (safety geom_mean({{t16,v1}})): {sub3_path}")
    print(f"    val2={safe_acc_a*100:.2f}%  ssv2={safe_acc_b*100:.2f}%")

    print("\nDone. Submit in this order on Kaggle:")
    print(f"   1) {sub1_path.name}")
    print(f"   2) {sub2_path.name}")
    print(f"   3) {sub3_path.name}")


if __name__ == "__main__":
    main()
