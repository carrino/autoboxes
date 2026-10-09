"""Score checkpoints against exact values on late positions from the run's own games.

Every sampled position with --min-undrawn..--max-undrawn undrawn edges (default 8..24) is
solved exactly by the C++ endgame solver (itself verified against the brute-force oracle in
tests/test_boxes_cpp_solver.py), which gives the set of optimal edges and the exact final
margin for the side to move. Only positions where the search decides are sampled: those
with a forced capture are played without a search and would count as optimal for free.
Per checkpoint this reports, on the same sampled positions:

  policy_optimal  the raw net's argmax over legal edges is an optimal edge
  search_optimal  the MCTS agent's move (forced-move collapse, temperature 0) is optimal
  value_sign      the net's P(win) > 0.5 agrees with the sign of the oracle's final margin
  margin_mae      |E[margin] - oracle final margin|

and the move rates of the baseline agents on the same positions as a scale. Positions come
from every NPZ game under $GAME_DATA_DIR/experiments/<this folder>/<tag>/ (bootstrap and all
self-play iterations), so every checkpoint is scored on one fixed set. Writes
data/oracle_eval-<tag>.csv and prints the table. Usage:

  uv run oracle_eval.py --tag 5x5-solver [--iterations 0 1 2] [--max-undrawn 28]
"""
# ruff: noqa: N806
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import torch

import alpha_go.boxes.agents  # noqa: F401  (registers boxes-* agents)
from alpha_go.agents.base import get_agent
from alpha_go.boxes.encode import encode_batch
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import (
    BoxesLeafEvaluator,
    BoxesMCTSAgent,
    add_search_flags,
    load_boxes_net,
    pick_device,
    search_flags,
)
from alpha_go.boxes.rules import geometry

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")).resolve()


def late_positions(game_dir: Path, rows: int, cols: int, min_undrawn: int, max_undrawn: int,
                   n: int, rng: random.Random) -> list[Any]:
    """Replay every game (C++ boards, which the agents expect) and sample n distinct
    search-decided positions with min_undrawn..max_undrawn undrawn edges."""
    boards: dict[tuple[int, int, int], Any] = {}
    for path in sorted(game_dir.rglob("*.npz")):
        game = np.load(path)
        board = alpha_go_cpp.BoxesBoard(rows, cols)
        for row, col in game["moves"]:
            if row < 0:
                break
            undrawn = board.num_edges() - board.move_count()
            decided = not alpha_go_cpp.BoxesSearchState(board).prefix()
            if min_undrawn <= undrawn <= max_undrawn and decided:
                boards.setdefault((board.edges(), board.to_play(), board.margin()), board.copy())
            board.play(int(row), int(col))
    assert boards, f"no positions with {min_undrawn}..{max_undrawn} undrawn edges under {game_dir}"
    keys = sorted(boards)
    rng.shuffle(keys)
    return [boards[k] for k in keys[:n]]


@torch.no_grad()
def raw_net(model: BoxesNet, device: torch.device, boards: list[Any]
            ) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Argmax legal edge, P(win) and E[margin] for every board in one forward pass."""
    planes_BKHW = torch.from_numpy(encode_batch(boards)).to(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        policy_BE, margin_BM = model(planes_BKHW)
    logits_BE = policy_BE.float().cpu().numpy()
    argmax = []
    for i, b in enumerate(boards):
        legal = b.get_legal_moves_flat()
        argmax.append(int(legal[int(np.argmax(logits_BE[i, legal]))]))
    win = model.win_prob(margin_BM).cpu().numpy()
    return argmax, win, model.expected_margin(margin_BM).cpu().numpy()


def agent_moves(agent: Any, boards: list[Any], seed: int) -> list[int]:
    moves = []
    for i, b in enumerate(boards):
        agent.start_game(b.rows())
        row, col = agent.select_move(b.copy(), seed + i)
        moves.append(b.edge_index(row, col))
    return moves


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=None,
                   help="checkpoint subdir and data subdir; default <rows>x<cols>")
    p.add_argument("--rows", type=int, default=None,
                   help="board rows; default from a tag like 5x5-solver, else 3")
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--iterations", type=int, nargs="*", default=None,
                   help="checkpoint iterations to score; default every iter*.pt found")
    p.add_argument("--min-undrawn", type=int, default=8)
    p.add_argument("--max-undrawn", type=int, default=24)
    p.add_argument("--num-positions", type=int, default=300)
    p.add_argument("--num_simulations", type=int, default=100)
    p.add_argument("--baselines", default="boxes-greedy,boxes-ab-d4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    add_search_flags(p)
    args = p.parse_args()
    mcts_flags, evaluator_flags = search_flags(args)

    device = pick_device("cpu" if args.cpu else None)
    assert args.cpu or device.type == "cuda", "CUDA not available; pass --cpu to run on CPU"
    size = re.match(r"(\d+)x(\d+)", args.tag or "")
    rows = args.rows or (int(size.group(1)) if size else 3)
    cols = args.cols or (int(size.group(2)) if size and not args.rows else rows)
    tag = args.tag or f"{rows}x{cols}"
    ckpt_dir = EXP_DIR / "checkpoints" / tag
    iterations = args.iterations
    if iterations is None:
        iterations = sorted(int(f.stem[4:]) for f in ckpt_dir.glob("iter*.pt"))
    assert iterations, f"no checkpoints under {ckpt_dir}"

    rng = random.Random(args.seed)
    t0 = time.time()
    boards = late_positions(GAME_DATA_DIR / "experiments" / EXP_NAME / tag, rows, cols,
                            args.min_undrawn, args.max_undrawn, args.num_positions, rng)
    solver = alpha_go_cpp.BoxesSolver(rows, cols, 1 << 22)
    geo = geometry(rows, cols)

    def child_value(mask: int, e: int) -> int:
        gained = sum(1 for b in geo.edge_boxes[e]
                     if all(((mask | 1 << e) >> s) & 1 for s in geo.box_edges[b]))
        rest = solver.value(mask | 1 << e)
        return gained + rest if gained else -rest

    values = [solver.value(b.edges()) for b in boards]  # remaining margin, side to move
    best = [{e for e in b.get_legal_moves_flat() if child_value(b.edges(), e) == v}
            for b, v in zip(boards, values)]
    final = np.array([b.margin() + v for b, v in zip(boards, values)])
    print(f"{len(boards)} positions with {args.min_undrawn}..{args.max_undrawn} undrawn edges "
          f"(mean undrawn {np.mean([b.num_edges() - b.move_count() for b in boards]):.1f}), "
          f"solved in {time.time() - t0:.1f}s; optimal-move share of a random legal move: "
          f"{np.mean([len(s) / len(b.get_legal_moves_flat()) for s, b in zip(best, boards)]):.3f}")

    def optimal_rate(moves: list[int]) -> float:
        return float(np.mean([m in s for m, s in zip(moves, best)]))

    rows_out: list[dict[str, object]] = []
    for name in [n for n in args.baselines.split(",") if n]:
        t1 = time.time()
        rate = optimal_rate(agent_moves(get_agent(name), boards, args.seed))
        rows_out.append({"agent": name, "iteration": "", "search_optimal": round(rate, 4)})
        print(f"{name:>14}: move optimal {rate:.3f} ({time.time() - t1:.0f}s)")
    for it in iterations:
        t1 = time.time()
        model = load_boxes_net(ckpt_dir / f"iter{it}.pt", device)
        argmax, win, expected = raw_net(model, device, boards)
        agent = BoxesMCTSAgent(BoxesLeafEvaluator(model, device, **evaluator_flags),
                               temperature=0.0, num_simulations=args.num_simulations,
                               **mcts_flags)
        decided = final != 0
        row = {
            "agent": f"iter{it}", "iteration": it,
            "policy_optimal": round(optimal_rate(argmax), 4),
            "search_optimal": round(optimal_rate(agent_moves(agent, boards, args.seed)), 4),
            "value_sign": round(float(np.mean((win[decided] > 0.5) == (final[decided] > 0))), 4),
            "margin_mae": round(float(np.mean(np.abs(expected - final))), 3),
        }
        rows_out.append(row)
        print(f"{row['agent']:>14}: policy optimal {row['policy_optimal']:.3f}  search optimal "
              f"{row['search_optimal']:.3f}  value sign {row['value_sign']:.3f}  margin MAE "
              f"{row['margin_mae']:.2f} ({time.time() - t1:.0f}s)")

    out = EXP_DIR / "data" / f"oracle_eval-{tag}.csv"
    out.parent.mkdir(exist_ok=True)
    fields = ["agent", "iteration", "policy_optimal", "search_optimal", "value_sign", "margin_mae",
              "num_positions", "min_undrawn", "max_undrawn", "num_simulations", "search_flags"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({**row, "num_positions": len(boards),
                             "min_undrawn": args.min_undrawn, "max_undrawn": args.max_undrawn,
                             "num_simulations": args.num_simulations,
                             "search_flags": json.dumps({**mcts_flags, **evaluator_flags})})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
