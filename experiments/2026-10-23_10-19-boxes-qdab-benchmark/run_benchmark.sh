#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
CHECKPOINT="$EXP/logs/ordinary-four-aqowx5f8/cycle-4-train/attempt-1u32cfzo/train/control.pt"
uv run python -u -m alpha_go.boxes.qdab_benchmark \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f \
  --games 100 --empty-games 20 --seed 20261112 \
  --seconds 5 --hard-timeout 30 \
  --output-root "$EXP/logs" "$@"
