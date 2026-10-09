"""Autoresearch edit target: MCTS search hyperparameters scored against the exact oracle.

A frozen 3x3 checkpoint searches a fixed set of late positions (6..14 undrawn edges, sampled
from the 3x3-cpu loop's own games) and the metric is the share of oracle-optimal moves it
plays (`search_optimal`, higher is better): exact, no opponent, far less variance than a
win rate. Only VAR is edited between runs; CKPT, the positions and the seed stay fixed so
runs are comparable. Positions with a forced capture are played without search, as in a
game, so only positions where the search decides (quiet, or the take-all / keep-control
choice) are sampled.

  uv run eval_search.py [--var '{"c_puct": 1.0}']   # prints ===RESULT=== {search_optimal, ...}

run_loop.py drives the keep / discard loop by passing one change at a time through --var,
so VAR below is the current best and every run is reproducible from runs.jsonl.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import alpha_go_cpp
import numpy as np
import torch

from alpha_go.boxes.nn_agent import BoxesLeafEvaluator, BoxesMCTSAgent, load_boxes_net
from alpha_go.boxes.oracle import Oracle

LOOP = Path(__file__).resolve().parents[1] / "2026-10-08_14-42-boxes-3x3-loop"
sys.path.insert(0, str(LOOP))
from oracle_eval import agent_moves, late_positions  # noqa: E402

CKPT = LOOP / "checkpoints" / "3x3-cpu" / "iter5.pt"  # the CPU run's champion
GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root"))
GAMES = GAME_DATA_DIR / "experiments" / LOOP.name / "3x3-cpu"
N_POSITIONS = 600
MIN_UNDRAWN, MAX_UNDRAWN = 6, 14
SEED = 0

# --- Variant (the only thing edited between runs) ---
VAR = dict(
    num_simulations=100, c_puct=1.5, leaf_batch_size=16,
    policy_temperature=1.0, margin_utility_lambda=0.0, margin_utility_k=6.0,
    solver_max_undrawn=0, solver_node_budget=20_000, merge_equivalent=False,
)
EVALUATOR_KEYS = ("policy_temperature", "margin_utility_lambda", "margin_utility_k")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--var", default="{}", help="JSON overrides of VAR for this run")
    VAR.update(json.loads(p.parse_args().var))
    torch.set_num_threads(2)
    device = torch.device("cpu")
    rng = random.Random(SEED)
    t0 = time.time()
    candidates = late_positions(GAMES, 3, 3, MIN_UNDRAWN, MAX_UNDRAWN, 4 * N_POSITIONS, rng)
    boards = [b for b in candidates if not alpha_go_cpp.BoxesSearchState(b).prefix()][:N_POSITIONS]
    assert len(boards) == N_POSITIONS, len(boards)
    oracle = Oracle(3, 3)
    values = [oracle.value(b.edges()) for b in boards]
    best = [{e for e in b.get_legal_moves_flat() if oracle.child_value(b.edges(), e) == v}
            for b, v in zip(boards, values)]
    final = np.array([b.margin() + v for b, v in zip(boards, values)])
    print(f"{len(boards)} positions solved in {time.time() - t0:.1f}s; random-move optimal share "
          f"{np.mean([len(s) / len(b.get_legal_moves_flat()) for s, b in zip(best, boards)]):.3f}")

    model = load_boxes_net(CKPT, device)
    evaluator = BoxesLeafEvaluator(model, device, **{k: VAR[k] for k in EVALUATOR_KEYS})
    mcts = {k: v for k, v in VAR.items() if k not in EVALUATOR_KEYS}
    agent = BoxesMCTSAgent(evaluator, temperature=0.0, **mcts)

    t1 = time.time()
    moves, q_signs = [], []
    for i, b in enumerate(boards):
        moves.extend(agent_moves(agent, [b], SEED + i))
        result = agent.last_search_result
        # result.Q is from the opponent's perspective (player_at_parent); 1 - Q is the mover's.
        q_signs.append(None if result is None else (1.0 - result.Q > 0.5) == (final[i] > 0))
    elapsed = time.time() - t1
    searched = [q for q in q_signs if q is not None]
    out = {
        "search_optimal": round(float(np.mean([m in s for m, s in zip(moves, best)])), 4),
        "root_q_sign": round(float(np.mean(searched)), 4) if searched else None,
        "searched_positions": len(searched), "positions": len(boards),
        "sec_per_move": round(elapsed / len(boards), 4), "elapsed_seconds": round(elapsed),
        "peak_vram_mb": 0, "checkpoint": str(CKPT), "var": VAR,
    }
    print("===RESULT===")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
