#!/usr/bin/env bash
set -euo pipefail

mkdir -p data
kaggle datasets download \
  -d paultimothymooney/chest-xray-pneumonia \
  -p data \
  --force
unzip -o data/chest-xray-pneumonia.zip -d data
echo "Dataset extracted. Expected training root: data/chest_xray"
