"""Exact solver labels for midgame positions: a supervised foothold for the value head.

Samples distinct search-decided positions with --min-undrawn..--max-undrawn undrawn edges from
the games under $GAME_DATA_DIR/experiments/<this folder>/<positions-tag>/ (every NPZ game, as
oracle_eval.py samples them), solves each with the C++ endgame solver within --node-budget
nodes (positions it cannot settle in the budget, or whose children it cannot, are skipped),
and writes them under $GAME_DATA_DIR/experiments/<this folder>/<save-name>/ in the game NPZ
format BoxesDataset reads (`dataset.save_labelled_positions`): exact final margin, a uniform
policy target over the optimal edges, the exact win as root value. Point a run at them with
EXTRA_DATA in run_iteration_local.sh; they join every iteration's training set.

  uv run solver_label.py --positions-tag 5x5-solver --min-undrawn 26 --max-undrawn 34 \
      --num-positions 20000 --save-name 5x5-oracle-26-34
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import alpha_go_cpp
import numpy as np

from alpha_go.boxes.dataset import save_labelled_positions
from alpha_go.boxes.rules import geometry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_eval import late_positions  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")).resolve()
CHUNK = 512  # positions per NPZ file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--positions-tag", required=True, help="run whose games supply the positions")
    p.add_argument("--save-name", required=True, help="output subdir under experiments/<folder>/")
    p.add_argument("--rows", type=int, default=None, help="default from the tag, else 5")
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--min-undrawn", type=int, default=26)
    p.add_argument("--max-undrawn", type=int, default=34)
    p.add_argument("--num-positions", type=int, default=20000)
    p.add_argument("--node-budget", type=int, default=200_000)
    p.add_argument("--table-entries", type=int, default=1 << 22)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    size = re.match(r"(\d+)x(\d+)", args.positions_tag)
    rows = args.rows or (int(size.group(1)) if size else 5)
    cols = args.cols or (int(size.group(2)) if size and not args.rows else rows)
    geo = geometry(rows, cols)
    out_dir = GAME_DATA_DIR / "experiments" / EXP_NAME / args.save_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    boards = late_positions(GAME_DATA_DIR / "experiments" / EXP_NAME / args.positions_tag, rows,
                            cols, args.min_undrawn, args.max_undrawn, args.num_positions,
                            random.Random(args.seed))
    print(f"{len(boards)} positions with {args.min_undrawn}..{args.max_undrawn} undrawn edges "
          f"sampled in {time.time() - t0:.0f}s")
    solver = alpha_go_cpp.BoxesSolver(rows, cols, args.table_entries)
    print(f"BoxesSolver table {solver.table_bytes() / 1e6:.0f} MB, node budget {args.node_budget}")

    def child_value(mask: int, e: int) -> int | None:
        after = mask | 1 << e
        gained = sum(all(after >> s & 1 for s in geo.box_edges[b]) for b in geo.edge_boxes[e])
        rest = solver.value_within(after, args.node_budget)
        return None if rest is None else (gained + rest if gained else -rest)

    by_margin: dict[int, list[tuple[np.ndarray, int, list[int]]]] = defaultdict(list)
    skipped = 0
    t1 = time.time()
    for i, board in enumerate(boards):
        mask = int(board.edges())
        value = solver.value_within(mask, args.node_budget)
        children = [] if value is None else [(e, child_value(mask, e))
                                             for e in board.get_legal_moves_flat()]
        if value is None or any(v is None for _, v in children):
            skipped += 1
            continue
        optimal = [e for e, v in children if v == value]
        mover_final = int(board.margin()) + int(value)
        margin_p1 = mover_final if int(board.to_play()) == 1 else -mover_final
        by_margin[margin_p1].append((board.to_numpy(), int(board.to_play()) - 1, optimal))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(boards)} solved, {skipped} skipped, {time.time() - t1:.0f}s",
                  flush=True)

    files = 0
    for margin_p1, items in sorted(by_margin.items()):
        for c in range(0, len(items), CHUNK):
            chunk = items[c:c + CHUNK]
            grids = np.stack([g for g, _, _ in chunk]).astype(np.int8)
            to_play = np.array([t for _, t, _ in chunk], dtype=np.int8)
            sign = "p" if margin_p1 >= 0 else "m"
            path = out_dir / f"oracle-{sign}{abs(margin_p1):02d}-{c // CHUNK:03d}.npz"
            save_labelled_positions(path, grids, to_play, [o for _, _, o in chunk], margin_p1, geo)
            files += 1
    labelled = sum(len(v) for v in by_margin.values())
    print(f"{labelled} positions labelled ({skipped} skipped on budget) in "
          f"{time.time() - t1:.0f}s; {files} files under {out_dir}")
    print("margins (player 1): "
          + ", ".join(f"{m:+d}: {len(v)}" for m, v in sorted(by_margin.items())))


if __name__ == "__main__":
    main()
