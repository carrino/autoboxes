"""Where were an arena's games decided? Replays games saved by `arena --save_games` against
the exact solver.

From the first position after the opening with at most --max-undrawn undrawn edges to the
end, every position is solved exactly (one solver per game, so the table carries along the
line; a position over --node-budget nodes is skipped). That gives A's exact final margin
under optimal play from each position on, so every game splits into:

  entry   A's exact margin at the first solved position: who was winning when the game came
          within reach
  errors  every move inside the reach that changed the exact margin; A's errors lower it, B's
          raise it, by the number of boxes the move gave away
  final   the margin actually played (= entry - A's error boxes + B's error boxes)

A game A lost from a winning entry was lost inside the reach. One A was already losing at
entry was decided earlier, above --max-undrawn, where only the nets' search and the classical
evaluation play. Writes data/arena_exact-<games file stem>-u<max undrawn>.csv (one row per
game) and prints the summary.

  uv run -m alpha_go.boxes.arena --rows 5 --a ckpt:<iter.pt> --b boxes-classic ... \\
      --save_games experiments/2026-10-08_14-42-boxes-3x3-loop/data/arena-gate12-classic.jsonl
  uv run arena_exact.py --games data/arena-gate12-classic.jsonl --rows 5 --max-undrawn 34
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np

EXP_DIR = Path(__file__).resolve().parent


def replay(game: dict[str, Any], rows: int, cols: int, max_undrawn: int, node_budget: int,
           table_entries: int) -> dict[str, Any]:
    """Solve every position of `game` within reach; return its entry value and errors."""
    board = alpha_go_cpp.BoxesBoard(rows, cols)
    solver = alpha_go_cpp.BoxesSolver(rows, cols, table_entries)
    a_player = 0 if game["a_first"] else 1
    sign = 1 if a_player == 0 else -1
    exact: list[tuple[int, bool, int | None]] = []  # (undrawn, A to move, A's exact margin)
    for k, (row, col) in enumerate(game["moves"]):
        undrawn = board.num_edges() - board.move_count()
        if k >= game["opening"] and undrawn <= max_undrawn:
            value = solver.value_within(board.edges(), node_budget)
            first = None if value is None else round(board.score()) + (
                value if board.player() == 0 else -value)
            exact.append((undrawn, board.player() == a_player,
                          None if first is None else sign * first))
        board.play(row, col)
    final = sign * round(board.score())
    assert board.is_game_over() and final == game["a_margin"], (game["index"], final)
    exact.append((0, False, final))
    solved = [e for e in exact if e[2] is not None]
    a_errors = b_errors = 0
    a_error_undrawn: list[int] = []
    for (undrawn, a_to_move, before), (_, _, after) in zip(exact, exact[1:]):
        if before is None or after is None:
            continue
        if a_to_move and after < before:
            a_errors += before - after
            a_error_undrawn.append(undrawn)
        if not a_to_move and after > before:
            b_errors += after - before
    return {
        "index": game["index"], "a_first": game["a_first"], "a_margin": final,
        "entry_undrawn": solved[0][0], "entry_margin": solved[0][2],
        "a_error_boxes": a_errors, "b_error_boxes": b_errors,
        "a_error_moves": len(a_error_undrawn),
        "a_error_undrawn": " ".join(map(str, a_error_undrawn)),
        "skipped": sum(e[2] is None for e in exact),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True, help="JSON lines from arena --save_games")
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--max-undrawn", type=int, default=34)
    p.add_argument("--node-budget", type=int, default=20_000_000)
    p.add_argument("--table-entries", type=int, default=1 << 22)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    cols = args.cols or args.rows
    path = Path(args.games)
    games = [json.loads(line) for line in path.read_text().splitlines()]
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        out = list(pool.map(lambda g: replay(g, args.rows, cols, args.max_undrawn,
                                             args.node_budget, args.table_entries), games))
    print(f"{len(out)} games from {path.name}, solved from {args.max_undrawn} undrawn edges "
          f"in {time.time() - start:.0f}s ({sum(o['skipped'] for o in out)} positions over "
          f"the {args.node_budget:,}-node budget)\n")

    entry = np.array([o["entry_margin"] for o in out])
    final = np.array([o["a_margin"] for o in out])
    a_err = np.array([o["a_error_boxes"] for o in out])
    b_err = np.array([o["b_error_boxes"] for o in out])
    print("A's result | games | A winning at entry | mean entry | A error boxes | "
          "B error boxes | mean final")
    for name, mask in (("won", final > 0), ("lost", final < 0), ("all", final == final)):
        n = int(mask.sum())
        if n:
            print(f"{name} | {n} | {float(np.mean(entry[mask] > 0)):.2f} | "
                  f"{float(np.mean(entry[mask])):+.2f} | {float(np.mean(a_err[mask])):.2f} | "
                  f"{float(np.mean(b_err[mask])):.2f} | {float(np.mean(final[mask])):+.2f}")
    lost = final < 0
    print(f"\nA's losses decided above {args.max_undrawn} undrawn (losing at entry): "
          f"{int((lost & (entry < 0)).sum())}; lost inside the reach from a winning entry: "
          f"{int((lost & (entry > 0)).sum())}")
    print(f"A's wins from a losing entry (B's errors inside the reach): "
          f"{int(((final > 0) & (entry < 0)).sum())}")
    undrawn = [int(u) for o in out for u in o["a_error_undrawn"].split()]
    if undrawn:
        counts = {u: undrawn.count(u) for u in sorted(set(undrawn), reverse=True)}
        print("A's error moves by undrawn edges: "
              + ", ".join(f"{u}: {n}" for u, n in counts.items()))

    csv_path = EXP_DIR / "data" / f"arena_exact-{path.stem}-u{args.max_undrawn}.csv"
    csv_path.parent.mkdir(exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
