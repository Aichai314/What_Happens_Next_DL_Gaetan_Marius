#!/bin/bash
# Setup complet SSv2 sur une nouvelle machine.
# Usage: bash src/misc/setup_ssv2.sh [DATA_ROOT]
#   DATA_ROOT : racine des données (défaut: /Data/$USER)
#
# Ce script :
#   1. Télécharge les vidéos SSv2 (~19 GB)
#   2. Télécharge les labels
#   3. Extrait 20 frames par vidéo pour les 32 classes du challenge
#   4. Met à jour le symlink processed_data/ssv2_32f

set -e

DATA_ROOT="${1:-/Data/$USER}"
SSV2_DIR="$DATA_ROOT/ssv2_dataset"
PROCESSED_DIR="$SSV2_DIR/processed"
VIDEOS_DIR="$SSV2_DIR/20bn-something-something-v2"
LABELS_DIR="$SSV2_DIR/labels"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SYMLINK="$PROJECT_ROOT/processed_data/ssv2_32f"

echo "========================================="
echo "  Setup SSv2"
echo "  DATA_ROOT    : $DATA_ROOT"
echo "  SSV2_DIR     : $SSV2_DIR"
echo "  PROJECT_ROOT : $PROJECT_ROOT"
echo "========================================="

mkdir -p "$SSV2_DIR"

# ── 1. Vidéos ────────────────────────────────
if [ -d "$VIDEOS_DIR" ] && [ "$(ls "$VIDEOS_DIR" | wc -l)" -ge 220000 ]; then
    echo "[1/4] Vidéos déjà présentes, étape ignorée."
else
    echo "[1/4] Téléchargement des vidéos (~19 GB)..."
    cd "$SSV2_DIR"
    wget -c --show-progress \
        "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00" \
        -O 20bn-something-something-v2-00.zip
    wget -c --show-progress \
        "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01" \
        -O 20bn-something-something-v2-01.zip

    echo "    Extraction de l'archive tar..."
    cat 20bn-something-something-v2-00.zip 20bn-something-something-v2-01.zip | tar -xzf -
    rm -f 20bn-something-something-v2-??.zip
    echo "    Vidéos extraites : $(ls "$VIDEOS_DIR" | wc -l)"
fi

# ── 2. Labels ────────────────────────────────
if [ -f "$LABELS_DIR/train.json" ]; then
    echo "[2/4] Labels déjà présents, étape ignorée."
else
    echo "[2/4] Téléchargement des labels..."
    cd "$SSV2_DIR"
    wget -c --show-progress \
        "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip" \
        -O labels.zip
    unzip -o labels.zip -d "$SSV2_DIR"
    rm -f labels.zip
    echo "    Labels extraits."
fi

# ── 3. Extraction des frames ─────────────────
TOTAL_FRAMES=$(find "$PROCESSED_DIR" -name "*.jpg" 2>/dev/null | wc -l)
if [ "$TOTAL_FRAMES" -ge 1000000 ]; then
    echo "[3/4] Frames déjà extraites ($TOTAL_FRAMES frames), étape ignorée."
else
    echo "[3/4] Extraction des frames (20 frames/vidéo, 12 workers)..."
    cd "$PROJECT_ROOT"
    python src/misc/extract_ssv2_frames.py \
        --num_frames 20 \
        --workers 12
fi

# ── 4. Symlink ───────────────────────────────
echo "[4/4] Mise à jour du symlink..."
mkdir -p "$(dirname "$SYMLINK")"
ln -sfn "$PROCESSED_DIR" "$SYMLINK"
echo "    $SYMLINK -> $PROCESSED_DIR"

echo ""
echo "========================================="
echo "  Setup terminé !"
echo "  Données : $PROCESSED_DIR"
echo "  Symlink : $SYMLINK"
echo ""
echo "  Pour entraîner avec ces données :"
echo "  python src/train.py experiment=<exp> \\"
echo "    dataset.train_dir=$PROCESSED_DIR/train \\"
echo "    dataset.val_dir=$PROCESSED_DIR/val \\"
echo "    dataset.num_frames=16"
echo "========================================="
