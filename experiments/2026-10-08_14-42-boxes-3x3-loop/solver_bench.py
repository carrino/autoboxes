"""Solve-time distribution of the exact endgame solver on positions from real games.

For each N in --undrawn, samples --num-positions positions with exactly N undrawn edges from
the NPZ games under $GAME_DATA_DIR/experiments/<this folder>/<tag>/ (self-play iterations
produce realistic endgames; bootstrap games are random play) and times `BoxesSolver.value`
with a fresh table per N, reporting p50 / p99 / max milliseconds and nodes, and the share
solved within each node budget in --budgets. The MCTS leaf setting is the largest N whose
p99 fits the per-leaf time you can afford (PLAN.md §4.1). Writes data/solver_bench-<tag>.csv.

  uv run solver_bench.py --tag 5x5 --undrawn 24 28 32 36
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import time
from pathlib import Path

import alpha_go_cpp
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")).resolve()


def positions_with(game_dir: Path, rows: int, cols: int, undrawn: int, n: int,
                   rng: random.Random, selfplay_only: bool) -> list[int]:
    """Edge masks of distinct positions with exactly `undrawn` undrawn edges."""
    masks: set[int] = set()
    pattern = "selfplay-*/**/*.npz" if selfplay_only else "**/*.npz"
    for path in sorted(game_dir.glob(pattern)):
        board = alpha_go_cpp.BoxesBoard(rows, cols)
        for row, col in np.load(path)["moves"]:
            if row < 0:
                break
            if board.num_edges() - board.move_count() == undrawn:
                masks.add(board.edges())
            board.play(int(row), int(col))
    out = sorted(masks)
    rng.shuffle(out)
    return out[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=None)
    p.add_argument("--rows", type=int, default=None,
                   help="board rows; default from a tag like 5x5-solver, else 5")
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--undrawn", type=int, nargs="+", default=[20, 24, 28, 32])
    p.add_argument("--num-positions", type=int, default=200)
    p.add_argument("--budgets", type=int, nargs="+", default=[2_000, 20_000, 200_000])
    p.add_argument("--table-entries", type=int, default=1 << 22)
    p.add_argument("--selfplay-only", action="store_true",
                   help="ignore bootstrap games (random play gives easy endgames)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mtdf", type=int, default=1, help="0: one full-window search per position")
    args = p.parse_args()
    size = re.match(r"(\d+)x(\d+)", args.tag or "")
    rows = args.rows or (int(size.group(1)) if size else 5)
    cols = args.cols or (int(size.group(2)) if size and not args.rows else rows)
    tag = args.tag or f"{rows}x{cols}"
    game_dir = GAME_DATA_DIR / "experiments" / EXP_NAME / tag
    rng = random.Random(args.seed)
    solver = alpha_go_cpp.BoxesSolver(rows, cols, args.table_entries)
    print(f"BoxesSolver {rows}x{cols}: table {solver.table_entries()} entries = "
          f"{solver.table_bytes() / 1e6:.1f} MB; positions from {game_dir}")

    rows_out: list[dict[str, object]] = []
    for undrawn in args.undrawn:
        masks = positions_with(game_dir, rows, cols, undrawn, args.num_positions, rng,
                               args.selfplay_only)
        if not masks:
            print(f"N={undrawn}: no positions")
            continue
        solver = alpha_go_cpp.BoxesSolver(rows, cols, args.table_entries)  # fresh table per N
        solver.set_mtdf(bool(args.mtdf))
        ms, nodes = [], []
        for mask in masks:
            t0 = time.perf_counter()
            solver.value(mask)
            ms.append((time.perf_counter() - t0) * 1e3)
            nodes.append(solver.nodes())
        row: dict[str, object] = {
            "undrawn": undrawn, "positions": len(masks),
            "ms_p50": round(float(np.percentile(ms, 50)), 3),
            "ms_p99": round(float(np.percentile(ms, 99)), 3), "ms_max": round(max(ms), 3),
            "nodes_p50": int(np.percentile(nodes, 50)), "nodes_p99": int(np.percentile(nodes, 99)),
            "nodes_max": max(nodes),
        }
        for budget in args.budgets:
            row[f"solved_within_{budget}"] = round(float(np.mean([n <= budget for n in nodes])), 3)
        rows_out.append(row)
        print(f"N={undrawn}: {len(masks)} positions, ms p50 {row['ms_p50']} p99 {row['ms_p99']} "
              f"max {row['ms_max']}; nodes p50 {row['nodes_p50']} p99 {row['nodes_p99']} "
              f"max {row['nodes_max']}; solved within "
              + ", ".join(f"{b}: {row[f'solved_within_{b}']}" for b in args.budgets))

    out = EXP_DIR / "data" / f"solver_bench-{tag}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
