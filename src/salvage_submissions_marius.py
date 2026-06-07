#!/usr/bin/env python3
"""
Last-minute submission salvage.

Uses the test softmax probabilities ALREADY cached in .submission_cache/ to
generate 3 small, defensible ensembles in seconds. No new model inference.

Reasoning (based on Kaggle results so far):
  - t16 solo Kaggle = 70.78% (best known so far)
  - 6-model weighted_mean = 70.11% (LOSES — weak models dilute t16)
  - 6-model power_mean = 69.56% (LOSES — same problem)
  → Trim the roster to the strongest, most-diverse models.

Pairwise agreement (from earlier run):
  tdn_ft vs v1                 : 45.2% ← most diverse pair
  v1 vs t16                    : 58.5% ← strong + diverse
  videomae* vs t16             : 65-68% (too redundant with t16, drop them)

Three submissions:
  1. geom_mean({t16, v1})                    — strongest pair, log-pool (robust)
  2. weighted_mean({t16, v1, tdn_ft}, boost) — adds tdn_ft (most diverse) with t16 boosted
  3. weighted_mean({t16, v1})                — fallback if Sub 1 disappoints

Usage:
    uv run python src/salvage_submissions_marius.py \
       dataset.test_dir=$(pwd)/processed_data/val2/test
"""

import csv
from pathlib import Path
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig

from create_submission import (
    discover_all_test_videos, load_manifest_video_names, resolve_video_dirs,
)
from create_submission_ensemble_marius import (
    NUM_CHALLENGE_CLASSES, _cache_path, fuse,
)


MODELS = {
    "tdn_ft":     "/Data/marius.truquin/Model_checkpoints/best_model_tdn_ssv2_ft.pt",
    "v1":         "/Data/marius.truquin/Model_checkpoints/best_model_vjepa2_large_v1.pt",
    "t16":        "/Data/marius.truquin/Model_checkpoints/best_model_vjepa2_large_t16.pt",
}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    test_root = Path(cfg.dataset.test_dir).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest")
    if manifest_cfg:
        manifest_path = Path(str(manifest_cfg)).resolve()
        video_names = load_manifest_video_names(manifest_path)
        video_dirs = resolve_video_dirs(test_root, video_names)
    else:
        video_names, video_dirs = discover_all_test_videos(test_root)
    print(f"Test set: {len(video_dirs)} videos.\n")

    # Load cached probs (refuse to run if any is missing — these are required).
    probs_by_name = {}
    for name, ckpt in MODELS.items():
        cp = _cache_path(ckpt, str(test_root))
        if not cp.exists():
            raise SystemExit(
                f"❌ Missing cache for {name}: {cp}\n"
                f"   Run create_submission_ensemble_marius.py first to populate it."
            )
        p = np.load(cp)
        assert p.shape[1] == NUM_CHALLENGE_CLASSES, (
            f"{name}: expected {NUM_CHALLENGE_CLASSES} classes, got {p.shape[1]}"
        )
        probs_by_name[name] = p
        print(f"  [cache] {name:<10}  shape={p.shape}")

    # Weights = stored val_accuracy from each checkpoint. For weighted_mean,
    # values close enough that the boost on t16 (×2) is what shifts the mix.
    import torch
    weights_by_name = {}
    for name, ckpt in MODELS.items():
        meta = torch.load(ckpt, map_location="cpu")
        weights_by_name[name] = float(meta.get("val_accuracy", 0.5))
        del meta
        print(f"  {name:<10}  ckpt.val_acc={weights_by_name[name]:.4f}")

    out_dir = Path("submissions")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, preds: np.ndarray) -> None:
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_name", "predicted_class"])
            for n, p in zip(video_names, preds):
                w.writerow([n, int(p)])

    # --- Sub 1: geom_mean({t16, v1}) ---
    subset = ["t16", "v1"]
    p = [probs_by_name[n] for n in subset]
    w = [weights_by_name[n] for n in subset]
    is_spec = [False] * len(subset)
    fused = fuse(p, is_spec, w, "geom_mean")
    sub1 = out_dir / "ensemble_submission_salvage1_geom_t16_v1.csv"
    _write(sub1, fused.argmax(1))
    print(f"\n✅ Sub 1 (geom_mean t16+v1): {sub1}")

    # --- Sub 2: weighted_mean({t16, v1, tdn_ft}) with t16 weight × 2 ---
    subset = ["t16", "v1", "tdn_ft"]
    p = [probs_by_name[n] for n in subset]
    w = [weights_by_name[n] for n in subset]
    w[0] *= 2.0   # boost t16
    is_spec = [False] * len(subset)
    fused = fuse(p, is_spec, w, "weighted_mean")
    sub2 = out_dir / "ensemble_submission_salvage2_wmean_top3_t16boost.csv"
    _write(sub2, fused.argmax(1))
    print(f"✅ Sub 2 (weighted_mean t16+v1+tdn_ft, t16×2): {sub2}")

    # --- Sub 3: weighted_mean({t16, v1}) raw ---
    subset = ["t16", "v1"]
    p = [probs_by_name[n] for n in subset]
    w = [weights_by_name[n] for n in subset]
    is_spec = [False] * len(subset)
    fused = fuse(p, is_spec, w, "weighted_mean")
    sub3 = out_dir / "ensemble_submission_salvage3_wmean_t16_v1.csv"
    _write(sub3, fused.argmax(1))
    print(f"✅ Sub 3 (weighted_mean t16+v1): {sub3}")

    # Sanity: how much do they disagree among themselves?
    print("\n--- Inter-submission disagreement (sanity) ---")
    p1 = np.fromiter((int(r[1]) for r in csv.reader(open(sub1)) if r[0] != "video_name"),
                     dtype=np.int64)
    p2 = np.fromiter((int(r[1]) for r in csv.reader(open(sub2)) if r[0] != "video_name"),
                     dtype=np.int64)
    p3 = np.fromiter((int(r[1]) for r in csv.reader(open(sub3)) if r[0] != "video_name"),
                     dtype=np.int64)
    print(f"  sub1 vs sub2 agree: {(p1 == p2).mean()*100:.1f}%")
    print(f"  sub1 vs sub3 agree: {(p1 == p3).mean()*100:.1f}%")
    print(f"  sub2 vs sub3 agree: {(p2 == p3).mean()*100:.1f}%")

    print("\nSubmit in this order:")
    print(f"   1) {sub1.name}     (geom_mean t16+v1 — robust)")
    print(f"   2) {sub2.name}     (3-model + t16 boosted — best upside)")
    print(f"   3) {sub3.name}     (fallback)")


if __name__ == "__main__":
    main()
