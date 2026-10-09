"""Searched self-play for one iteration: iter N checkpoint on both sides.

Registers two BoxesMCTSAgent classes (distinct names so the registry keeps black and
white apart), optionally sharing one PlaneBatchedEngine so every game thread's leaves are
batched into the same GPU forwards, then calls alpha_go.self_play with --collect-metrics.
"""
from __future__ import annotations

import argparse
import sys

import torch

from alpha_go.boxes.inference import PlaneBatchedEngine
from alpha_go.boxes.nn_agent import load_boxes_net, pick_device, register_boxes_mcts_agent

C_PUCT = 1.5
LEAF_BATCH_SIZE = 16
DIRICHLET_ALPHA = 0.5  # ~10 / branching factor on 3x3; noise keeps openings diverse
DIRICHLET_WEIGHT = 0.25


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--num_games", type=int, default=200)
    p.add_argument("--num_simulations", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature_cutoff", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64, help="shared engine batch size")
    p.add_argument("--save-name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solver_max_undrawn", type=int, default=0,
                   help="exact endgame solver at <= N undrawn edges (0 = off)")
    p.add_argument("--solver_node_budget", type=int, default=20_000)
    p.add_argument("--cpu", action="store_true")
    args, remaining = p.parse_known_args()

    device = pick_device("cpu" if args.cpu else None)
    assert args.cpu or device.type == "cuda", "CUDA not available; pass --cpu to run on CPU"
    model = load_boxes_net(args.checkpoint, device)
    engine = PlaneBatchedEngine(model, device, batch_size=args.batch_size, batch_timeout_ms=1.0)
    engine.start()
    mcts = dict(
        num_simulations=args.num_simulations, c_puct=C_PUCT, temperature=args.temperature,
        temperature_cutoff=args.temperature_cutoff, add_noise=True,
        noise_alpha=DIRICHLET_ALPHA, noise_weight=DIRICHLET_WEIGHT, leaf_batch_size=LEAF_BATCH_SIZE,
        solver_max_undrawn=args.solver_max_undrawn, solver_node_budget=args.solver_node_budget,
    )
    black = register_boxes_mcts_agent("sp-black", args.checkpoint, args.rows, args.cols,
                                      engine=engine, **mcts)
    white = register_boxes_mcts_agent("sp-white", args.checkpoint, args.rows, args.cols,
                                      engine=engine, **mcts)
    print(f"self-play: {args.num_games} games, {args.num_simulations} sims, device={device}, "
          f"torch={torch.__version__}", flush=True)
    from alpha_go.self_play import main as self_play_main
    sys.argv = [
        "self_play", "--game", "boxes", "--board_size", str(args.rows),
        *(["--board-cols", str(args.cols)] if args.cols else []),
        "--black", black, "--white", white, "--num_games", str(args.num_games),
        "--num_workers", str(args.num_workers), "--save-name", args.save_name,
        "--seed", str(args.seed), "--collect-metrics", "--black-is-teacher", "--white-is-teacher",
        *remaining,
    ]
    self_play_main()
    print(f"engine: {engine.total_requests} leaves in {engine.total_batches} batches "
          f"(avg {engine.total_requests / max(1, engine.total_batches):.1f})", flush=True)
    engine.stop()


if __name__ == "__main__":
    main()
