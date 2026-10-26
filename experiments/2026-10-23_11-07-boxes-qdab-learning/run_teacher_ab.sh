#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run python -u experiments/2026-10-23_11-07-boxes-qdab-learning/run_teacher_ab.py "$@"
