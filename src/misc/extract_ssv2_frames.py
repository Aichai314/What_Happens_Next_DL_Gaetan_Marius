"""
Extrait les frames des vidéos SSv2 pour les 32 classes du challenge.
Produit la même structure que processed_data/val2 :
  {dest}/{split}/{NNN_Class_name}/video_{id}/frame_000.jpg ...

Usage:
  python src/misc/extract_ssv2_frames.py
  python src/misc/extract_ssv2_frames.py --num_frames 16 --workers 12
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SSV2_ROOT = Path("/Data/marius.truquin/ssv2_dataset")
VIDEOS_DIR = SSV2_ROOT / "20bn-something-something-v2"
LABELS_DIR = SSV2_ROOT / "labels"
DEST_ROOT  = SSV2_ROOT / "processed"

# Mapping SSv2 template -> dossier challenge (NNN_Class_name)
TEMPLATE_TO_FOLDER = {
    "Closing something":                                                       "000_Closing_something",
    "Covering something with something":                                       "001_Covering_something_with_something",
    "Dropping something into something":                                       "002_Dropping_something_into_something",
    "Folding something":                                                       "003_Folding_something",
    "Hitting something with something":                                        "004_Hitting_something_with_something",
    "Holding something":                                                       "005_Holding_something",
    "Moving something away from something":                                    "006_Moving_something_away_from_something",
    "Moving something closer to something":                                    "007_Moving_something_closer_to_something",
    "Moving something down":                                                   "008_Moving_something_down",
    "Moving something up":                                                     "009_Moving_something_up",
    "Opening something":                                                       "010_Opening_something",
    "Picking something up":                                                    "011_Picking_something_up",
    "Pouring something into something":                                        "012_Pouring_something_into_something",
    "Pouring something out of something":                                      "013_Pouring_something_out_of_something",
    "Pretending to pick something up":                                         "014_Pretending_to_pick_something_up",
    "Pretending to pour something out of something, but something is empty":   "015_Pretending_to_pour_something_out_of_something_but_something_",
    "Pretending to put something into something":                              "016_Pretending_to_put_something_into_something",
    "Pretending to throw something":                                           "017_Pretending_to_throw_something",
    "Pulling something from left to right":                                    "018_Pulling_something_from_left_to_right",
    "Pulling something from right to left":                                    "019_Pulling_something_from_right_to_left",
    "Putting something behind something":                                      "020_Putting_something_behind_something",
    "Putting something in front of something":                                 "021_Putting_something_in_front_of_something",
    "Putting something into something":                                        "022_Putting_something_into_something",
    "Putting something next to something":                                     "023_Putting_something_next_to_something",
    "Putting something onto something":                                        "024_Putting_something_onto_something",
    "Showing something to the camera":                                         "025_Showing_something_to_the_camera",
    "Spilling something next to something":                                    "026_Spilling_something_next_to_something",
    "Taking something out of something":                                       "028_Taking_something_out_of_something",
    "Throwing something":                                                      "029_Throwing_something",
    "Turning something upside down":                                           "030_Turning_something_upside_down",
    "Uncovering something":                                                    "031_Uncovering_something",
    "Unfolding something":                                                     "032_Unfolding_something",
}


def get_frame_count(video_path: Path) -> int:
    """Estime le nombre de frames via ffmpeg -i (duration × fps)."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path)],
        capture_output=True, text=True
    )
    info = result.stderr
    dur_match = re.search(r'Duration:\s*(\d+):(\d+):([\d.]+)', info)
    fps_match = re.search(r'(\d+(?:\.\d+)?)\s*fps', info)
    if dur_match and fps_match:
        h, m, s = int(dur_match.group(1)), int(dur_match.group(2)), float(dur_match.group(3))
        duration = h * 3600 + m * 60 + s
        fps = float(fps_match.group(1))
        return max(1, int(duration * fps))
    return 30  # fallback


def extract_frames(args):
    video_path, out_dir, num_frames = args
    out_dir.mkdir(parents=True, exist_ok=True)

    total = get_frame_count(video_path)

    n = min(num_frames, total)
    # Indices uniformément espacés
    indices = [int(round(i * (total - 1) / (n - 1))) for i in range(n)] if n > 1 else [0]

    select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)
    out_pattern = str(out_dir / "frame_%03d.jpg")

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vf", f"select='{select_expr}',scale=224:224",
         "-vsync", "vfr", "-q:v", "2", out_pattern],
        capture_output=True
    )
    return result.returncode == 0, str(video_path)


def normalize_template(template: str) -> str:
    """'Closing [something]' -> 'Closing something'"""
    return re.sub(r'\[([^\]]+)\]', r'\1', template)


def load_split(json_path):
    with open(json_path) as f:
        data = json.load(f)
    result = []
    for entry in data:
        normalized = normalize_template(entry["template"])
        if normalized in TEMPLATE_TO_FOLDER:
            result.append((entry["id"], normalized))
    return result


def process_split(split_name, json_path, num_frames, workers):
    print(f"\n==> Split: {split_name}")
    entries = load_split(json_path)
    print(f"    {len(entries)} vidéos à extraire (sur 32 classes)")

    tasks = []
    for vid_id, template in entries:
        video_path = VIDEOS_DIR / f"{vid_id}.webm"
        if not video_path.exists():
            continue
        folder = TEMPLATE_TO_FOLDER[template]
        out_dir = DEST_ROOT / split_name / folder / f"video_{vid_id}"
        if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) >= num_frames:
            continue  # déjà extrait
        tasks.append((video_path, out_dir, num_frames))

    print(f"    {len(tasks)} vidéos à traiter (non encore extraites)")
    if not tasks:
        return

    ok = err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(extract_frames, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            success, path = fut.result()
            if success:
                ok += 1
            else:
                err += 1
            if i % 500 == 0:
                print(f"    {i}/{len(tasks)} — ok={ok} err={err}")

    print(f"    Terminé: ok={ok} err={err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    print(f"Destination : {DEST_ROOT}")
    print(f"Frames/vidéo : {args.num_frames}  |  Workers : {args.workers}")

    process_split("train", LABELS_DIR / "train.json", args.num_frames, args.workers)
    process_split("val",   LABELS_DIR / "validation.json", args.num_frames, args.workers)

    print("\n==> Extraction terminée !")
    print(f"Données dans : {DEST_ROOT}")
    print("\nPour entraîner :")
    print(f"  python src/train.py experiment=videomae_large_kinetics \\")
    print(f"    dataset.train_dir={DEST_ROOT}/train \\")
    print(f"    dataset.val_dir={DEST_ROOT}/val \\")
    print(f"    dataset.num_frames=16")


if __name__ == "__main__":
    main()
