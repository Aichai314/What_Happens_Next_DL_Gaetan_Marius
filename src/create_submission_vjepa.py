#!/usr/bin/env python3
"""
Single-model V-JEPA submission generator.

Copy of create_submission.py with one fix: image_size is read from the
checkpoint's training config (or runtime cfg) and passed to VideoTransform.
The shared create_submission.py defaults to 224, which silently downscales
V-JEPA inputs from its native 256x256 and tanks Kaggle accuracy.

Usage (from src/)::

    python create_submission_vjepa.py training.checkpoint_path=/path/to/best_model_vjepa2_large.pt
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset
from train import build_model
from utils import VideoTransform, set_seed
from create_submission import (
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
    build_model_from_checkpoint,
    run_inference,
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Model on device: {device}", flush=True)

    num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(ckpt.get("pretrained", cfg.model.pretrained))

    # The fix: prefer the image_size baked into the checkpoint's training cfg
    # so create_submission_vjepa works even when invoked without
    # `experiment=vjepa2_large`. Falls back to runtime cfg, then to 224.
    ckpt_cfg = OmegaConf.create(ckpt["config"]) if ckpt.get("config") else None
    if ckpt_cfg is not None and "image_size" in ckpt_cfg.dataset:
        image_size = int(ckpt_cfg.dataset.image_size)
    else:
        image_size = int(cfg.dataset.get("image_size", 224))
    print(f"Inference image_size: {image_size} (num_frames={num_frames})", flush=True)
    eval_transform = VideoTransform(cfg, is_training=False, use_imagenet_norm=pretrained, image_size=image_size)

    test_root = Path(cfg.dataset.test_dir).resolve()

    val_accuracy = float(ckpt.get("val_accuracy", 0.0))
    prefix = "pretrained" if pretrained else "fromscratch"
    acc_str = f"{val_accuracy * 100:.2f}".replace(".", "_")
    submissions_dir = checkpoint_path.parent / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    output_path = submissions_dir / f"{prefix}_vjepa_{acc_str}.csv"
    manifest_cfg = cfg.dataset.get("test_manifest")

    print(f"Indexing video folders under: {test_root}", flush=True)
    if manifest_cfg:
        manifest_path = Path(str(manifest_cfg)).resolve()
        print(f"Reading manifest: {manifest_path}", flush=True)
        video_names = load_manifest_video_names(manifest_path)
        video_dirs = resolve_video_dirs(test_root, video_names)
        print(f"Resolved {len(video_dirs)} video folders from manifest.", flush=True)
    else:
        print("No dataset.test_manifest provided; using all video_* folders.", flush=True)
        video_names, video_dirs = discover_all_test_videos(test_root)
        print(f"Discovered {len(video_dirs)} video folders.", flush=True)
    sample_list: List[Tuple[Path, int]] = [(p, 0) for p in video_dirs]

    dataset = VideoFrameDataset(
        root_dir=test_root,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=sample_list,
    )
    batch_size = int(cfg.training.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Starting inference: {len(dataset)} clips, batch_size={batch_size}, "
        f"{len(loader)} batches",
        flush=True,
    )
    predictions = run_inference(model, loader, device, total_videos=len(dataset))
    print("Inference finished.", flush=True)

    if len(predictions) != len(video_names):
        raise RuntimeError(
            f"Prediction count {len(predictions)} != manifest length {len(video_names)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing submission CSV: {output_path}", flush=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, predictions):
            w.writerow([name, pred])

    print(f"Done. Wrote {len(predictions)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
