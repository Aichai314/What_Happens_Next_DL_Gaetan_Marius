#!/usr/bin/env python3
"""
Kaggle submission generator — Logistic Regression meta-learner.

Late fusion via L2-regularised LogisticRegression on the stacked softmax of N
expert models. Trained on val2/val (100% of rows), applied to the Kaggle test
set. Reuses the cached softmax already on disk:
  - val2/val softmax  → .ensemble_cache/   (populated by evaluate_ensemble_marius.py)
  - test softmax      → .submission_cache/ (populated by create_submission_ensemble_marius.py)

Run both of those first if any cache is missing — this script never recomputes
expert probabilities.

Usage:
    # Default (C=0.1, matches the eval script):
    uv run python src/create_submission_logreg_marius.py \\
        dataset.val_dir=$(pwd)/processed_data/val2/val \\
        dataset.test_dir=$(pwd)/processed_data/val2/test

    # Override C to sweep regularisation strength:
    uv run python src/create_submission_logreg_marius.py \\
        dataset.val_dir=$(pwd)/processed_data/val2/val \\
        dataset.test_dir=$(pwd)/processed_data/val2/test \\
        +meta.C=0.3

Output:
    submissions/ensemble_submission_logreg_C{C}.csv
"""

import csv
from pathlib import Path
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from create_submission import (
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
)
from create_submission_ensemble_marius import _cache_path
from evaluate_ensemble_marius import _get_cache_path, _get_labels_cache_path


# Roster matching evaluate_ensemble_marius.py — these 5 models have both a
# val2/val cache (for training the meta-learner) AND a test cache (for predicting).
# vjepa2_large_t16 is omitted: no val2/val cache exists for it, so it cannot
# contribute features to a meta-learner trained on val2/val.
MODELS: List[str] = [
    "/Data/marius.truquin/Model_checkpoints/best_model_tdn_ssv2.pt",
    "/Data/marius.truquin/Model_checkpoints/best_model_vjepa2_large_v1.pt",
    "/Data/marius.truquin/Model_checkpoints/best_model_vjepa2_specialist.pt",
    "/Data/marius.truquin/Model_checkpoints/best_model_videomae_large_kinetics.pt",
    "/Data/marius.truquin/Model_checkpoints/best_model_videomae_v2_ssv2_2.pt",
]


def _load_or_die(path: Path, what: str) -> np.ndarray:
    if not path.exists():
        raise SystemExit(
            f"❌ Missing cache for {what}: {path}\n"
            f"   Populate the cache first by running the appropriate script:\n"
            f"     - val2 caches  : uv run python src/evaluate_ensemble_marius.py\n"
            f"     - test caches  : uv run python src/create_submission_ensemble_marius.py"
        )
    return np.load(path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    val_dir = str(Path(cfg.dataset.val_dir).resolve())
    test_root = Path(cfg.dataset.test_dir).resolve()
    C = float(cfg.get("meta", {}).get("C", 0.1))

    print(f"\n=== LogReg meta-learner submission ===")
    print(f"  val_dir   : {val_dir}")
    print(f"  test_dir  : {test_root}")
    print(f"  C         : {C}  (smaller = stronger L2 regularisation)")
    print(f"  N experts : {len(MODELS)}\n")

    # ──────────────────────────────────────────────────────────
    # 1) Load val2 softmax stack (rows = val2 videos, cols = expert × classes)
    # ──────────────────────────────────────────────────────────
    val_probs: List[np.ndarray] = []
    for ckpt in MODELS:
        cp = _get_cache_path(ckpt, val_dir)
        p = _load_or_die(cp, f"val cache for {Path(ckpt).name}")
        val_probs.append(p)
        print(f"  [val]  {Path(ckpt).stem:<45} shape={p.shape}")

    X_train = np.hstack(val_probs)
    y_raw = _load_or_die(_get_labels_cache_path(val_dir), "val labels")
    print(f"\n  X_train: {X_train.shape}   y_train: {y_raw.shape}")

    # Encode labels (handles the missing class 27 in the challenge)
    le = LabelEncoder()
    y_train = le.fit_transform(y_raw)

    # ──────────────────────────────────────────────────────────
    # 2) Load test softmax stack — must use the SAME model order
    # ──────────────────────────────────────────────────────────
    print()
    test_probs: List[np.ndarray] = []
    for ckpt in MODELS:
        cp = _cache_path(ckpt, str(test_root))
        p = _load_or_die(cp, f"test cache for {Path(ckpt).name}")
        test_probs.append(p)
        print(f"  [test] {Path(ckpt).stem:<45} shape={p.shape}")

    X_test = np.hstack(test_probs)
    print(f"\n  X_test:  {X_test.shape}")
    assert X_train.shape[1] == X_test.shape[1], (
        f"Feature dims mismatch — train {X_train.shape[1]} vs test {X_test.shape[1]}. "
        f"The model list or the cached softmax shapes differ between val2 and test."
    )

    # ──────────────────────────────────────────────────────────
    # 3) Train LogReg on 100% of val2 (same recipe as evaluate_ensemble_marius)
    # ──────────────────────────────────────────────────────────
    print(f"\nTraining LogisticRegression(C={C}, class_weight='balanced')…")
    meta = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    meta.fit(X_train, y_train)
    print(f"  train accuracy (sanity, expected high): {meta.score(X_train, y_train):.4f}")

    # ──────────────────────────────────────────────────────────
    # 4) Predict on test and decode labels back to the challenge space
    # ──────────────────────────────────────────────────────────
    y_pred_encoded = meta.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)

    # ──────────────────────────────────────────────────────────
    # 5) Match predictions to test video names and write the CSV
    # ──────────────────────────────────────────────────────────
    manifest_cfg = cfg.dataset.get("test_manifest")
    if manifest_cfg:
        video_names = load_manifest_video_names(Path(str(manifest_cfg)).resolve())
        video_dirs = resolve_video_dirs(test_root, video_names)
    else:
        video_names, video_dirs = discover_all_test_videos(test_root)
    assert len(video_names) == len(y_pred), (
        f"Test video count ({len(video_names)}) ≠ test cache row count "
        f"({len(y_pred)}). The cached test softmax was computed for a different "
        f"test set — clear .submission_cache/ and regenerate."
    )

    unique, counts = np.unique(y_pred, return_counts=True)
    top5 = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1])[:5]
    print(f"\n  classes used: {len(unique)}  (28-32 expected since class 27 is absent)")
    print(f"  top-5 predicted classes (idx, count): {top5}")

    out = Path("submissions") / f"ensemble_submission_logreg_C{C}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, y_pred):
            w.writerow([name, int(pred)])
    print(f"\n  ✅ wrote {out}")
    print(f"\nDone — submit this CSV to Kaggle.")


if __name__ == "__main__":
    main()
