#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run python -u -m alpha_go.boxes.qdab_training_collect \
  --checkpoint "${1:?Pass the selected checkpoint path}" \
  --checkpoint-sha256 "${CHECKPOINT_SHA256:-c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f}" \
  --games "${GAMES:-60}" --hours "${HOURS:-3}" \
  --seconds 5 --hard-timeout 30 --seed "${SEED:-20261123}"
