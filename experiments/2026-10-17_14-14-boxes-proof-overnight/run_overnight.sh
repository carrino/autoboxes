#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export GAME_DATA_DIR="${GAME_DATA_DIR:-$HOME/autoboxes-data/game_data_root}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1
uv run python -u experiments/2026-10-17_14-14-boxes-proof-overnight/overnight.py "$@"
