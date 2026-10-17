#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export GAME_DATA_DIR="${GAME_DATA_DIR:-$HOME/autoboxes-data/game_data_root}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
uv run python -u -m alpha_go.boxes.curriculum --hours "${HOURS:-8}" "$@"
