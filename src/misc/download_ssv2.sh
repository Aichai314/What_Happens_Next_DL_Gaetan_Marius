#!/bin/bash
set -e

DEST="/Data/marius.truquin/ssv2_dataset"
mkdir -p "$DEST"
cd "$DEST"

echo "==> Téléchargement des zips dans $DEST ..."

wget -c --show-progress \
  "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00" \
  -O 20bn-something-something-v2-00.zip

wget -c --show-progress \
  "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01" \
  -O 20bn-something-something-v2-01.zip

echo "==> Extraction de l'archive tar ..."
cat 20bn-something-something-v2-00.zip 20bn-something-something-v2-01.zip | tar -xvzf -

echo "==> Nettoyage des parties ..."
rm -f 20bn-something-something-v2-??.zip

echo "==> Téléchargement des labels ..."
wget -c --show-progress \
  "https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip" \
  -O 20bn-something-something-download-package-labels.zip

echo "==> Extraction des labels ..."
unzip -o 20bn-something-something-download-package-labels.zip -d "$DEST"
rm -f 20bn-something-something-download-package-labels.zip

echo "==> Terminé ! Vidéos et labels extraits dans $DEST"
ls "$DEST" | head -20
