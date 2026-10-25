#!/bin/bash
# Perceptual stage: NeRF, Zero-1-to-3 on the RGBA image, Stable Diffusion on a UMAP-sampled caption.
set -euo pipefail
cd "$(dirname "$0")"

INDEX="${1:?test index, matching umap_cache/weights/test_XXXX.pt}"
RGBA="${2:?path to the segmented RGBA image}"
BANK="${3:-../mindeye/umap_cache/bank_sd.pt}"
WEIGHTS="${4:-../mindeye/umap_cache/weights/test_$(printf '%04d' "$INDEX").pt}"
GPU="${5:-0}"

python launch.py \
  --config configs/brain3d-coarse-sd.yaml \
  --train \
  --gpu "$GPU" \
  "data.image_path=${RGBA}" \
  "system.prompt_processor.bank_path=${BANK}" \
  "system.prompt_processor.weights_path=${WEIGHTS}" \
  trainer.max_steps=2000
