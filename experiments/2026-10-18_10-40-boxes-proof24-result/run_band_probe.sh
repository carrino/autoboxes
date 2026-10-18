#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run python -u experiments/2026-10-18_10-40-boxes-proof24-result/band_probe.py "$@"
