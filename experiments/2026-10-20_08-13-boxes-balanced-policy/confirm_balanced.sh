#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../.."
EXP=experiments/2026-10-08_14-42-boxes-3x3-loop
RUN="$EXP/logs/balanced-policy-k3s3hb6v"
A="$RUN/train/attempt-t3mgl6kr/train/balanced.pt"
B="$RUN/inputs/iter24.pt"
OUT=$(mktemp -d "$RUN/arena-vs-iter24-XXXXXX")
echo "Results directory: $OUT"
git rev-parse HEAD > "$OUT/code-commit.txt"
printf '%s  %s\n' \
  0cbf8ca0ddd2216a308125394c625b76d21c73ae2321442a0c132ac7b34f69d3 "$A" \
  73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d "$B" \
  | tee "$OUT/checkpoints.sha256" | sha256sum -c -
printf '%s\n' 'seed=20261102; games=200; workers=2; opening_moves=4; sims=400; solver=34; solver_budget=50000; policy_temperature=1; prove_terminals=1; shape_terminals=1; margin_utility_lambda=0.5; margin_utility_k=6; c_puct=1.5; leaf_batch_size=16' > "$OUT/settings.txt"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -u -m alpha_go.boxes.arena \
  --rows 5 --cols 5 --a "ckpt:$A" --b "ckpt:$B" \
  --num_games 200 --num_workers 2 --seed 20261102 --opening_moves 4 \
  --sims 400 --solver 34 --solver_budget 50000 \
  --policy_temperature 1 --prove_terminals 1 --shape_terminals 1 \
  --margin_utility_lambda 0.5 --margin_utility_k 6 \
  --c_puct 1.5 --leaf_batch_size 16 \
  --save_games "$OUT/games.jsonl" 2>&1 | tee "$OUT/run.log"
tar -czf "$OUT/results.tgz" -C "$OUT" run.log games.jsonl code-commit.txt checkpoints.sha256 settings.txt
echo "Send this file: $OUT/results.tgz"
