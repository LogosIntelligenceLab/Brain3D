#!/bin/bash
# Semantic stage: convert the coarse NeRF to DMTet and refine color with depth ControlNet.
set -euo pipefail
cd "$(dirname "$0")"

INDEX="${1:?test index}"
RGBA="${2:?path to the segmented RGBA image}"
COARSE_CKPT="${3:?path to the coarse last.ckpt}"
BANK="${4:-../mindeye/umap_cache/bank_sd.pt}"
WEIGHTS="${5:-../mindeye/umap_cache/weights/test_$(printf '%04d' "$INDEX").pt}"
GPU="${6:-0}"

python launch.py \
  --config configs/brain3d-refine-sd.yaml \
  --train \
  --gpu "$GPU" \
  "data.image_path=${RGBA}" \
  "system.prompt_processor.bank_path=${BANK}" \
  "system.prompt_processor.weights_path=${WEIGHTS}" \
  "system.geometry_convert_from=${COARSE_CKPT}" \
  trainer.max_steps=2000
