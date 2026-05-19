#!/usr/bin/env python3
"""
Generate one submission CSV per checkpoint in MY_MODELS.

Used to A/B test individual checkpoints on the Kaggle leaderboard so we can
pick the best one per architecture before stacking into XGBoost.

Output: /Data/marius.truquin/Model_checkpoints/submissions/<ckpt_stem>.csv

Run from project root::

    PYTHONPATH=src python src/create_submissions_individual_marius.py hydra.run.dir=.
"""

from __future__ import annotations

import csv
import gc
from pathlib import Path
from typing import List, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset
from utils import VideoTransform, set_seed

from create_submission import (
    build_model_from_checkpoint,
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
    run_inference,
)

MY_MODELS: List[str] = [
    "/Data/marius.truquin/Model_checkpoints/best_model_videomae_large_kinetics.pt",
    "/Data/marius.truquin/Model_checkpoints/best_model_videomae_large_kinetics_V2.pt",
    "/Data/marius.truquin/Model_checkpoints/best_model_videomae_large_kinetics_v3.pt",
    "/Data/marius.truquin/Model_checkpoints/best_model_videomae_v2_ssv2.pt",
    "/Data/marius.truquin/Model_checkpoints/best_model_videomae_v2_ssv2_2.pt",
    "/Data/marius.truquin/Model_checkpoints/best_model_videomae_v2_ssv2_v2.pt",
]

OUTPUT_DIR = Path("/Data/marius.truquin/Model_checkpoints/submissions")


def submit_one(
    ckpt_path: Path,
    test_root: Path,
    video_names: List[str],
    video_dirs: List[Path],
    cfg: DictConfig,
    device: torch.device,
) -> Path:
    print(f"\nLoading checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(ckpt.get("pretrained", cfg.model.pretrained))
    eval_transform = VideoTransform(cfg, is_training=False, use_imagenet_norm=pretrained)

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
        f"Inference: {len(dataset)} clips, batch_size={batch_size}, {len(loader)} batches",
        flush=True,
    )
    predictions = run_inference(model, loader, device, total_videos=len(dataset))

    if len(predictions) != len(video_names):
        raise RuntimeError(
            f"Prediction count {len(predictions)} != manifest length {len(video_names)}"
        )

    output_path = OUTPUT_DIR / f"{ckpt_path.stem}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, predictions):
            w.writerow([name, pred])
    print(f"Wrote {len(predictions)} rows to {output_path}", flush=True)

    del model
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()

    return output_path


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    test_root = Path(cfg.dataset.test_dir).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest")

    print(f"Indexing video folders under: {test_root}", flush=True)
    if manifest_cfg:
        manifest_path = Path(str(manifest_cfg)).resolve()
        video_names = load_manifest_video_names(manifest_path)
        video_dirs = resolve_video_dirs(test_root, video_names)
    else:
        video_names, video_dirs = discover_all_test_videos(test_root)
    print(f"Resolved {len(video_dirs)} test videos.", flush=True)

    missing = [p for p in MY_MODELS if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"Missing checkpoints: {missing}")

    written: List[Path] = []
    for i, ckpt_path_str in enumerate(MY_MODELS, start=1):
        ckpt_path = Path(ckpt_path_str)
        print(f"\n{'=' * 60}\n[{i}/{len(MY_MODELS)}] {ckpt_path.name}\n{'=' * 60}")
        written.append(submit_one(ckpt_path, test_root, video_names, video_dirs, cfg, device))

    print("\nAll submissions written:")
    for p in written:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
