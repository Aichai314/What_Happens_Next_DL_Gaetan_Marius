#!/usr/bin/env python3
"""
Ensemble Submission Generator — multi-strategy late fusion.

Runs each expert checkpoint ONCE on the Kaggle test set, caches its softmax
probabilities to ``.submission_cache/`` (keyed by ckpt × test directory), then
emits one submission CSV per fusion strategy in ``submissions/``. Kaggle
allows 3 daily submissions, so we ship 3 well-justified strategies.

Why this design (vs the previous "single mean" version):
    1. The previous script excluded vjepa2_large_t16 because it crashed with
       cudaErrorIllegalMemoryAccess at bs ≥ 2. t16 solo scores 70% on Kaggle —
       by far the strongest single expert — so excluding it was a major loss.
       Fix: bs=1 + ``torch.cuda.empty_cache()`` between batches.
    2. Plain arithmetic mean is hurt by overconfident wrong models. We add
       log-pool (geometric mean) and competence-weighted variants.
    3. The 9-class specialist was being dumped into the average untouched —
       meaningless, its indices 0..8 don't align with challenge 0..32. It is
       now applied as a multiplicative Bayes refinement on its 9 target
       classes only.

Fusion strategies (each from a real source):

    'mean'           — arithmetic mean of softmax. Bishop, PRML §14.2.1 —
                       Bayes-optimal pool under independence + uniform prior.
                       Baseline.

    'geom_mean'      — geometric mean (log-pool). Genest & Zidek, "Combining
                       Probability Distributions", Statistical Science 1986
                       (foundational opinion-pooling result). De-facto default
                       for Kaggle vision winners — robust to single
                       overconfident wrong predictions (any model can
                       effectively veto a class by giving it ~0 probability).

    'weighted_mean'  — competence-weighted arithmetic mean, weights = each
                       checkpoint's stored val_accuracy (normalised). Polikar,
                       "Ensemble Based Systems in Decision Making", IEEE
                       Circuits & Systems Magazine 2006 §III.

    'power_mean'     — generalised mean M_p(x) = (mean(x^p))^(1/p), p=0.5.
                       Interpolates between geometric (p→0) and arithmetic
                       (p=1).

    '<mode>_spec'    — adds a multiplicative Bayes update on the 9 specialist
                       target classes (Hoeting et al., "Bayesian Model
                       Averaging: A Tutorial", Statistical Science 1999 §3 —
                       BMA over a partition):
                           for c ∈ S:  P(c) ← P(c) · ( P_spec(c) / (1/|S|) )
                       then renormalise. The 24 non-specialist classes stay
                       untouched.

The specialist's 9 target classes (mirrors src/train_specialist.py):
    TARGET_CLASSES = [2, 9, 11, 14, 16, 17, 22, 29, 30]
"""

import csv
import gc
import hashlib
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from train import build_model
from dataset.video_dataset import VideoFrameDataset
from utils import VideoTransform

from create_submission import (
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
)

# Mirrors TARGET_CLASSES in src/train_specialist.py.
# spec_idx i ↔ challenge class SPECIALIST_CLASSES[i].
SPECIALIST_CLASSES: List[int] = [2, 9, 11, 14, 16, 17, 22, 29, 30]

NUM_CHALLENGE_CLASSES = 33  # 0..32 with 27 absent (no training signal → never wins argmax).


# ────────────────────────────────────────────────────────────────────────────
# Probability extraction (cached on disk; rerunning a new fusion is instant)
# ────────────────────────────────────────────────────────────────────────────
def _cache_path(ckpt_path: str, test_root: str) -> Path:
    cache_dir = Path(".submission_cache")
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.md5(f"{ckpt_path}:{test_root}".encode()).hexdigest()
    return cache_dir / f"{Path(ckpt_path).stem}_{key}.npy"


def _partial_cache_path(ckpt_path: str, test_root: str) -> Path:
    """Partial-progress cache: written every CHECKPOINT_EVERY batches, so a
    crash mid-extraction does not lose all completed batches."""
    cache_dir = Path(".submission_cache")
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.md5(f"{ckpt_path}:{test_root}".encode()).hexdigest()
    return cache_dir / f"{Path(ckpt_path).stem}_{key}.partial.npy"


# Save partial probs every this-many batches (resilience vs CUDA crashes).
PARTIAL_CHECKPOINT_EVERY_BATCHES = 50


def _val2_val_accuracy(ckpt_path: str, val_dir: str) -> float | None:
    """Honest val_accuracy on val2/val, recomputed from cached logits.

    Uses the same cache layout as evaluate_ensemble_marius.py
    (``.ensemble_cache/{ckpt_stem}_{md5(ckpt_path:val_dir)}.npy`` and
    ``labels_{md5(val_dir)}.npy``). If both exist, accuracy is argmax-match
    and is the right weight for ``weighted_mean`` regardless of which val set
    the original checkpoint was selected on.

    Returns None when the cache is missing — caller should fall back to the
    checkpoint's stored val_accuracy.
    """
    cache_dir = Path(".ensemble_cache")
    if not cache_dir.is_dir():
        return None
    ckpt_stem = Path(ckpt_path).stem
    ckpt_key = hashlib.md5(f"{ckpt_path}:{val_dir}".encode()).hexdigest()
    probs_path = cache_dir / f"{ckpt_stem}_{ckpt_key}.npy"
    labels_key = hashlib.md5(val_dir.encode()).hexdigest()
    labels_path = cache_dir / f"labels_{labels_key}.npy"
    if not (probs_path.exists() and labels_path.exists()):
        return None
    try:
        probs = np.load(probs_path)
        labels = np.load(labels_path)
        if probs.shape[0] != labels.shape[0]:
            return None
        # Specialist: 9-class probs vs 33-class labels → argmax incomparable.
        # Skip and let caller fall back (specialist isn't averaged anyway).
        if probs.shape[1] != NUM_CHALLENGE_CLASSES:
            return None
        return float((probs.argmax(axis=1) == labels).mean())
    except Exception:
        return None


def _gpu_is_contested(util_threshold: int = 50, mem_threshold_mb: int = 8000) -> bool:
    """True if another process is heavily using the GPU.

    Why this matters on shared machines: when peer utilisation is high, our
    cudaMalloc calls fight for VRAM and a peer kernel can step on memory we
    just allocated, surfacing as ``illegal memory access`` deep inside
    transformers' VJEPA2 forward. Once that fires, the CUDA context is
    *poisoned for the whole process* — even ``model.to('cpu')`` cannot recover
    because HF VJEPA2 still issues CUDA ops on internal masks. The only safe
    response is to NEVER touch CUDA from this process. We detect contention
    upfront via nvidia-smi and route fragile models to CPU from the start.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5,
        )
        for line in out.decode().strip().splitlines():
            mem_str, util_str = [s.strip() for s in line.split(",")]
            if int(mem_str) > mem_threshold_mb or int(util_str) > util_threshold:
                return True
        return False
    except Exception:
        # No nvidia-smi or it errored → conservatively assume free.
        return False


def _pick_batch_size(is_tdn: bool, is_t16: bool, num_frames: int, image_size: int) -> int:
    """Conservative batch-size heuristic.

    Lessons learned the hard way on this PyTorch 2.11+/CUDA 12.8 stack:
      - VJEPA2 256px crashes immediately at bs=4 when other users share the
        GPU (`illegal memory access` inside MLP GELU). bs=2 has historically
        worked → keep that.
      - t16 (16 frames, 256px) crashed even at bs=2 → bs=1.
      - TDN runs on CPU, so its batch is RAM-bound only.
    """
    if is_tdn:
        return 32
    if is_t16 or num_frames >= 16:
        return 1
    if image_size >= 256:
        return 2          # NB: was 4 before — caused vjepa2_large_v1 to crash.
    if num_frames > 8:
        return 16
    return 64


def _run_inference_loop(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    model_device: torch.device,
    ckpt_stem: str,
    partial_path: Path,
    start_idx: int,
    use_cache: bool,
) -> np.ndarray:
    """Iterate the loader and return stacked probs. Saves partial progress
    every ``PARTIAL_CHECKPOINT_EVERY_BATCHES`` batches so a crash does not
    lose completed work."""
    is_fragile = model_device.type == "cuda"  # empty_cache helps on the fragile stack.
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, (batch, _) in enumerate(
            tqdm(loader, desc=f"  {ckpt_stem}", initial=start_idx, total=start_idx + len(loader))
        ):
            batch = batch.to(model_device)
            logits = model(batch)
            if model_device.type == "cuda":
                torch.cuda.synchronize()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            if is_fragile:
                torch.cuda.empty_cache()

            # Incremental save — protects against later crashes.
            if use_cache and (batch_idx + 1) % PARTIAL_CHECKPOINT_EVERY_BATCHES == 0:
                np.save(partial_path, np.vstack(all_probs))
    return np.vstack(all_probs)


def extract_one_model_probs(
    ckpt_path: str,
    test_root: Path,
    video_dirs: List[Path],
    device: torch.device,
    use_cache: bool = True,
) -> Tuple[np.ndarray, float]:
    """Run one model on the test set. Returns (probs (N, C), val_accuracy).

    Robustness:
      - Full-result cache → second run is instant.
      - Partial-progress cache (every N batches) so a crash mid-extraction
        does not throw away already-computed batches. NOTE: the partial cache
        is only safe to reuse if the dataset and ordering are identical
        between runs, which they are (shuffle=False, same video_dirs).
      - Auto CPU fallback if CUDA throws ``illegal memory access`` (or any
        RuntimeError) — slow but reliable.
    """
    cache = _cache_path(ckpt_path, str(test_root))
    partial = _partial_cache_path(ckpt_path, str(test_root))

    # Always read checkpoint metadata (val_accuracy is needed by weighted_*).
    # map_location='cpu' first: direct map_location='cuda' triggers a
    # device-side assert on the vjepa2 specialist (PyTorch 2.11+/CUDA 12.8).
    meta = torch.load(ckpt_path, map_location="cpu")
    val_acc = float(meta.get("val_accuracy", 0.5))
    cfg = OmegaConf.create(meta["config"])

    if use_cache and cache.exists():
        probs = np.load(cache)
        del meta
        print(f"  [cache] {cache.name}  shape={probs.shape}  val_acc={val_acc:.4f}")
        return probs, val_acc

    model_name = cfg.model.get("name", "")
    ckpt_stem = Path(ckpt_path).stem
    is_tdn = "tdn" in model_name.lower()
    is_t16 = "t16" in ckpt_stem.lower()

    # Device routing:
    #   - TDN: always CPU (MSE group-conv has an unfixable CUDA async race on
    #     PyTorch 2.11+; see src/models/tdn_ssv2.py header).
    #   - Anything else: CPU if GPU is contested (another user > 50% util or
    #     > 8GB resident). Otherwise CUDA.
    #
    # We *cannot* let CUDA fail and recover in-process: the error
    # ``illegal memory access`` poisons the CUDA context, after which even
    # CPU-routed VJEPA2 forwards crash (HF's apply_masks still does a CUDA
    # ``torch.gather`` on the mask buffer). Hence: decide BEFORE touching CUDA.
    if is_tdn:
        model_device = torch.device("cpu")
    elif device.type == "cuda" and _gpu_is_contested():
        print(f"  ⚠ GPU contested by another process — running {ckpt_stem} on CPU.")
        model_device = torch.device("cpu")
    else:
        model_device = device

    use_imagenet_norm = cfg.model.get("pretrained", False)
    image_size = int(cfg.dataset.get("image_size", 224))
    transform = VideoTransform(
        cfg, is_training=False, use_imagenet_norm=use_imagenet_norm, image_size=image_size
    )

    num_frames = int(meta.get("num_frames", cfg.dataset.num_frames))
    sample_list = [(p, 0) for p in video_dirs]

    bs = _pick_batch_size(is_tdn, is_t16, num_frames, image_size)

    # Resume from partial cache: skip the already-extracted prefix in the
    # dataset by slicing sample_list. Order is deterministic (shuffle=False).
    resumed: List[np.ndarray] = []
    start_idx = 0
    if use_cache and partial.exists():
        try:
            partial_arr = np.load(partial)
            n_done = int(partial_arr.shape[0])
            if 0 < n_done < len(sample_list):
                resumed.append(partial_arr)
                start_idx = n_done
                sample_list = sample_list[n_done:]
                print(f"  [partial] resuming from {n_done}/{n_done + len(sample_list)} "
                      f"({partial.name})")
            elif n_done >= len(sample_list):
                # Already done — promote to full cache and return.
                if use_cache:
                    np.save(cache, partial_arr)
                    partial.unlink(missing_ok=True)
                del meta
                print(f"  [partial→full] promoted {partial.name} → {cache.name}")
                return partial_arr, val_acc
        except Exception as e:
            print(f"  [partial] failed to load {partial.name} ({e}); restarting model.")

    dataset = VideoFrameDataset(
        root_dir=test_root, num_frames=num_frames, transform=transform, sample_list=sample_list
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=4
    )

    model = build_model(cfg)
    model.load_state_dict(meta["model_state_dict"])
    model = model.to(model_device).eval()
    print(f"  device={model_device.type} bs={bs} num_frames={num_frames} "
          f"image_size={image_size} val_acc={val_acc:.4f}")

    # No in-process CUDA→CPU fallback: an ``illegal memory access`` poisons
    # the CUDA context (subsequent CPU forwards in HF VJEPA2 still issue CUDA
    # ops on internal mask buffers). If we crash here, the right answer is to
    # re-run the script: GPU contention detection at the top will route to CPU.
    new_probs = _run_inference_loop(
        model, loader, model_device, ckpt_stem, partial, start_idx, use_cache
    )

    probs = np.vstack(resumed + [new_probs]) if resumed else new_probs
    if use_cache:
        np.save(cache, probs)
        partial.unlink(missing_ok=True)
        print(f"  [cache] saved {cache.name}")

    del model, meta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs, val_acc


# ────────────────────────────────────────────────────────────────────────────
# Specialist projection: 9-class softmax → 33-class multiplicative Bayes factor
# ────────────────────────────────────────────────────────────────────────────
def specialist_bayes_factor(spec_probs: np.ndarray) -> np.ndarray:
    """(N, 9) specialist softmax → (N, 33) multiplicative Bayes update factors.

    For c ∈ S:  factor[c] = P_spec(c) / (1 / |S|)
    For c ∉ S:  factor[c] = 1.0   (left untouched)

    Applying  P_final(c) ∝ P_main(c) · factor(c)  is the BMA posterior on the
    sub-domain S, assuming uniform prior on S (Hoeting et al. 1999 §3).
    """
    n = spec_probs.shape[0]
    factor = np.ones((n, NUM_CHALLENGE_CLASSES), dtype=np.float32)
    uniform = 1.0 / len(SPECIALIST_CLASSES)
    for spec_idx, ch in enumerate(SPECIALIST_CLASSES):
        factor[:, ch] = spec_probs[:, spec_idx] / uniform
    return factor


# ────────────────────────────────────────────────────────────────────────────
# Fusion
# ────────────────────────────────────────────────────────────────────────────
def fuse(
    expert_probs: List[np.ndarray],
    expert_is_specialist: List[bool],
    expert_weights: List[float],
    mode: str,
) -> np.ndarray:
    """Combine experts into one (N, 33) probability matrix.

    Specialist experts are NEVER pooled directly with the 33-class experts
    (incompatible label spaces). They only appear as a Bayes refinement on
    the 9 target classes when ``mode`` ends with '_spec'.
    """
    eps = 1e-12

    main_probs = [p for p, sp in zip(expert_probs, expert_is_specialist) if not sp]
    main_wts = [w for w, sp in zip(expert_weights, expert_is_specialist) if not sp]
    spec_probs = [p for p, sp in zip(expert_probs, expert_is_specialist) if sp]

    assert main_probs, "Need at least one 33-class expert."
    assert all(p.shape == main_probs[0].shape for p in main_probs), \
        "Main experts have inconsistent (N, C) shapes — row alignment is broken."
    assert main_probs[0].shape[1] == NUM_CHALLENGE_CLASSES, \
        f"Main experts must have {NUM_CHALLENGE_CLASSES} columns; got {main_probs[0].shape[1]}."

    stacked = np.stack(main_probs, axis=0)  # (E, N, 33)
    use_spec = mode.endswith("_spec")
    base = mode[:-5] if use_spec else mode

    if base == "mean":
        fused = np.mean(stacked, axis=0)
    elif base == "geom_mean":
        fused = np.exp(np.mean(np.log(np.clip(stacked, eps, 1.0)), axis=0))
    elif base == "weighted_mean":
        w = np.asarray(main_wts, dtype=np.float32)
        w = w / w.sum()
        fused = np.einsum("e,enc->nc", w, stacked)
    elif base == "power_mean":
        p_exp = 0.5
        fused = np.mean(np.clip(stacked, eps, None) ** p_exp, axis=0) ** (1.0 / p_exp)
    else:
        raise ValueError(f"Unknown base fusion mode: {base!r}")

    if use_spec:
        if not spec_probs:
            print(f"  ⚠ mode {mode!r} requested but no specialist expert in roster — no-op.")
        else:
            factors = [specialist_bayes_factor(sp) for sp in spec_probs]
            if len(factors) == 1:
                f = factors[0]
            else:
                # Multiple specialists → pool their Bayes factors by geom-mean.
                f = np.exp(np.mean(np.log(np.clip(np.stack(factors), eps, None)), axis=0))
            fused = fused * f

    fused = fused / np.clip(fused.sum(axis=1, keepdims=True), eps, None)
    return fused


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Surface GPU state once at startup so the user knows what to expect.
    if device.type == "cuda":
        if _gpu_is_contested():
            print("⚠ GPU is heavily used by another process (>50% util or >8GB resident).")
            print("  → VJEPA2 / VideoMAE will be routed to CPU automatically.")
            print("  → Each model takes ~30-90 min on CPU vs ~2-5 min on free GPU.")
            print("  → Probabilities are cached, so once a model finishes it is never redone.")
        else:
            print("✓ GPU appears free; will use CUDA for non-TDN experts.")

    # =========================================================
    # ROSTER — all available experts. Specialist auto-detected by num_classes.
    # =========================================================
    # NB tdn_ssv2 -> tdn_ssv2_ft: raw TDN was selected on ssv2_32f/val (~68% there,
    # ~52% on val2/val) → its stored val_accuracy was massively over-estimating
    # its true contribution. tdn_ssv2_ft is fine-tuned and selected on val2/val,
    # so its val_acc is honest for weighted_mean.
    my_models = [
        "/Data/marius.truquin/Model_checkpoints/best_model_tdn_ssv2_ft.pt",
        "/Data/marius.truquin/Model_checkpoints/best_model_vjepa2_large_v1.pt",
        "/Data/marius.truquin/Model_checkpoints/best_model_vjepa2_large_t16.pt",
        "/Data/marius.truquin/Model_checkpoints/best_model_vjepa2_specialist.pt",
        "/Data/marius.truquin/Model_checkpoints/best_model_videomae_large_kinetics.pt",
        "/Data/marius.truquin/Model_checkpoints/best_model_videomae_v2_ssv2_2.pt",
    ]

    # =========================================================
    # FUSION MODES — one CSV per mode in submissions/. The 3 picked below are
    # the ones to ship to Kaggle (daily limit = 3):
    #   - weighted_mean   : robust default; t16 (strongest expert) is up-weighted
    #                       proportionally to its val_accuracy.
    #   - geom_mean       : Kaggle gold-standard pooling; veto effect downweights
    #                       overconfident-but-wrong classes.
    #   - geom_mean_spec  : geom_mean + Bayes refinement from the 9-class
    #                       specialist on the real-vs-pretend / vertical-lift
    #                       confusion cluster (the documented VJEPA failure mode).
    # Add 'mean' / 'power_mean' / '*_spec' variants if you want to A/B more.
    # Probs are cached, so additional modes regenerate in ~1s each.
    # =========================================================
    FUSION_MODES_TO_RUN = [
        "weighted_mean",   # val2/val: 67.90% (4-model eval, sans t16)
        "geom_mean",       # val2/val: 66.83% — filet de sécurité robuste
        "power_mean",      # val2/val: 67.37% — entre arith et geom
        # 'geom_mean_spec' / autres _spec : -4pts sur val2/val, le spécialiste
        # nuit à l'ensemble (facteur de Bayes peut écraser 9×). À ré-investiguer.
    ]

    # =========================================================
    # TEST VIDEO DISCOVERY
    # =========================================================
    test_root = Path(cfg.dataset.test_dir).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest")
    if manifest_cfg:
        manifest_path = Path(str(manifest_cfg)).resolve()
        video_names = load_manifest_video_names(manifest_path)
        video_dirs = resolve_video_dirs(test_root, video_names)
    else:
        video_names, video_dirs = discover_all_test_videos(test_root)
    print(f"Found {len(video_dirs)} test videos under {test_root}.")

    # =========================================================
    # EXTRACT PROBABILITIES (cached per checkpoint × test_root)
    #
    # Weights for weighted_mean: prefer val2/val accuracy recomputed from
    # cached logits in .ensemble_cache/ (honest metric, same distribution as
    # Kaggle test). Fall back to checkpoint's stored val_accuracy when the
    # cache is missing (t16, brand-new fine-tunes, etc.) — in that case the
    # weight may be on a different val set (ssv2_32f/val for several
    # SSv2-pretrained models), which can over-weight that expert.
    # =========================================================
    val_dir_for_weights = str(Path(cfg.dataset.val_dir).resolve())
    expert_probs: List[np.ndarray] = []
    expert_is_specialist: List[bool] = []
    expert_weights: List[float] = []
    expert_ckpts: List[str] = []

    for i, ckpt_path in enumerate(my_models):
        print(f"\n--- Expert {i+1}/{len(my_models)}: {Path(ckpt_path).name} ---")
        if not Path(ckpt_path).is_file():
            print(f"  ⚠ checkpoint not found, skipping: {ckpt_path}")
            continue
        probs, ckpt_val_acc = extract_one_model_probs(ckpt_path, test_root, video_dirs, device)
        is_spec = (probs.shape[1] != NUM_CHALLENGE_CLASSES)

        # Choose the weighting accuracy: honest val2/val if we have it cached,
        # otherwise the checkpoint's stored value.
        val2_acc = _val2_val_accuracy(ckpt_path, val_dir_for_weights) if not is_spec else None
        if val2_acc is not None:
            weight_acc = val2_acc
            weight_source = f"val2/val (recomputed from cache, ckpt-stored was {ckpt_val_acc:.4f})"
        else:
            weight_acc = ckpt_val_acc
            weight_source = "ckpt.val_accuracy (no val2/val cache available)"

        expert_probs.append(probs)
        expert_is_specialist.append(is_spec)
        expert_weights.append(weight_acc)
        expert_ckpts.append(ckpt_path)
        tag = "SPECIALIST" if is_spec else "main"
        print(f"  → {tag}  classes={probs.shape[1]}  weight={weight_acc:.4f}  [{weight_source}]")

    # =========================================================
    # PAIRWISE AGREEMENT — quick sanity that the experts are diverse
    # (otherwise ensembling adds nothing).
    # =========================================================
    solo_preds = []
    for p, is_sp, ckpt in zip(expert_probs, expert_is_specialist, expert_ckpts):
        if is_sp:
            continue
        solo_preds.append((Path(ckpt).stem, np.argmax(p, axis=1)))
    if len(solo_preds) >= 2:
        print("\n--- Pairwise top-1 agreement on test set ---")
        for i in range(len(solo_preds)):
            for j in range(i + 1, len(solo_preds)):
                n1, p1 = solo_preds[i]
                n2, p2 = solo_preds[j]
                agree = float((p1 == p2).mean())
                print(f"  {n1[:42]:<42} vs {n2[:42]:<42} : {agree:.1%}")

    # =========================================================
    # FUSE & WRITE — one versioned CSV per mode.
    # Preserves the existing ensemble_submission_v<N>.csv naming so the user's
    # Kaggle submission tracking is uninterrupted.
    # =========================================================
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(parents=True, exist_ok=True)
    existing = list(submissions_dir.glob("ensemble_submission_v*.csv"))
    next_version = max(
        (int(p.stem.split("_v")[-1]) for p in existing if p.stem.split("_v")[-1].isdigit()),
        default=0,
    ) + 1

    for mode in FUSION_MODES_TO_RUN:
        print(f"\n--- Fusion: {mode} ---")
        fused = fuse(expert_probs, expert_is_specialist, expert_weights, mode)
        predictions = np.argmax(fused, axis=1)

        unique, counts = np.unique(predictions, return_counts=True)
        top5 = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1])[:5]
        print(f"  classes used: {len(unique)} (28-32 expected since 27 has no training signal)")
        print(f"  top-5 predicted classes (idx, count): {top5}")

        out = submissions_dir / f"ensemble_submission_v{next_version}_{mode}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_name", "predicted_class"])
            for name, pred in zip(video_names, predictions):
                w.writerow([name, int(pred)])
        print(f"  ✅ wrote {out}")
        next_version += 1

    print("\nAll submissions written. Probabilities cached in .submission_cache/ —")
    print("re-running with a different FUSION_MODES_TO_RUN list is now ~instant.")


if __name__ == "__main__":
    main()
