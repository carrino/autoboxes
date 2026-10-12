"""Reproducible Task 4 sampling and CPU search measurements (see README.md)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import torch

from alpha_go.boxes.nn_agent import (
    BoxesLeafEvaluator,
    BoxesMCTSAgent,
    BoxesSearchResult,
    BoxesSolvedResult,
    add_search_flags,
    load_boxes_net,
    optimal_edges,
    search_flags,
)
from alpha_go.boxes.rules import geometry

HERE = Path(__file__).resolve().parent
LOOP = HERE.parent / "2026-10-08_14-42-boxes-3x3-loop"


def sample(args: argparse.Namespace) -> None:
    """Match oracle_eval.late_positions, saving move prefixes instead of pickled boards."""
    source = Path(os.environ["GAME_DATA_DIR"]) / "experiments" / LOOP.name / args.tag
    positions: dict[tuple[int, int, int], list[int]] = {}
    paths = sorted(source.rglob("*.npz"))
    assert paths, source
    for path in paths:
        with np.load(path) as game:
            prefix = game["start_moves"] if "start_moves" in game else np.zeros((0, 2), dtype=int)
            moves = np.concatenate([prefix, game["moves"]])
        board = alpha_go_cpp.BoxesBoard(5, 5)
        history: list[int] = []
        for row, col in moves:
            if row < 0:
                continue
            left = board.num_edges() - board.move_count()
            if 29 <= left <= 34 and not alpha_go_cpp.BoxesSearchState(board).prefix():
                key = (board.edges(), board.to_play(), board.margin())
                positions.setdefault(key, history.copy())
            edge = board.edge_index(int(row), int(col))
            assert board.play_edge(edge)
            history.append(edge)
    keys = sorted(positions)
    random.Random(args.seed).shuffle(keys)
    selected = keys[:args.num_positions]
    assert len(selected) == args.num_positions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "tag": args.tag, "seed": args.seed, "games": len(paths),
        "available_positions": len(keys), "keys": selected,
        "moves": [positions[key] for key in selected],
    }) + "\n")
    print(f"Sampled {len(selected)} / {len(keys)} positions from {len(paths)} {args.tag} games")


def evaluate(args: argparse.Namespace) -> None:
    """One fresh agent per run, with the same position order and seeds as oracle_eval."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data = json.loads(args.positions.read_text())
    boards = []
    for moves in data["moves"]:
        board = alpha_go_cpp.BoxesBoard(5, 5)
        for edge in moves:
            assert board.play_edge(edge)
        boards.append(board)
    if args.limit:
        boards = boards[:args.limit]
    start = time.perf_counter()
    cache = args.positions.with_name("oracle_labels-" + args.positions.stem + ".json")
    fingerprint = hashlib.sha256(args.positions.read_bytes()).hexdigest()
    if data["tag"] == "5x5-chains" and cache.exists():
        labels = json.loads(cache.read_text())
        assert labels["positions_sha256"] == fingerprint
        best = [set(edges) for edges in labels["optimal"][:len(boards)]]
    else:
        oracle = alpha_go_cpp.BoxesSolver(5, 5, 1 << 22)
        geo = geometry(5, 5)
        # Clean evaluation labels stay in memory and never become a training dataset.
        best = []
        for i, board in enumerate(boards):
            best.append(set(optimal_edges(oracle, board, geo)))
            if (i + 1) % 50 == 0:
                print(f"Oracle: {i + 1}/{len(boards)}", flush=True)
        if data["tag"] == "5x5-chains" and not args.limit:
            cache.write_text(json.dumps({"positions_sha256": fingerprint,
                                         "optimal": [sorted(edges) for edges in best]}) + "\n")
        del oracle
    oracle_seconds = time.perf_counter() - start
    print(f"Oracle ready in {oracle_seconds:.1f}s", flush=True)
    device = torch.device("cpu")
    checkpoint = LOOP / "checkpoints" / "5x5-chains" / f"iter{args.iteration}.pt"
    model = load_boxes_net(checkpoint, device)
    mcts_flags, evaluator_flags = search_flags(args)
    agent = BoxesMCTSAgent(
        BoxesLeafEvaluator(model, device, **evaluator_flags), temperature=0.0,
        num_simulations=400, solver_max_undrawn=30, solver_node_budget=100000,
        **mcts_flags,
    )
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for i, (board, optimal) in enumerate(zip(boards, best)):
        t0 = time.perf_counter()
        agent.start_game(5)
        row, col = agent.select_move(board.copy(), data["seed"] + i)
        edge = board.edge_index(row, col)
        result = agent.last_search_result
        records.append({
            "position": i, "mask": board.edges(), "undrawn": 60 - board.move_count(),
            "move": edge, "optimal": int(edge in optimal),
            "seconds": time.perf_counter() - t0,
            "root_solved": int(isinstance(result, BoxesSolvedResult)),
            "root_proven": int(isinstance(result, BoxesSearchResult)
                               and result.tree.is_root_proven()),
        })
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - start
            print(f"{args.name}: {i + 1}/{len(boards)} in {elapsed:.1f}s", flush=True)
    seconds = time.perf_counter() - start
    output = HERE / "data" / args.name
    output.parent.mkdir(exist_ok=True)
    with output.with_suffix(".csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "name": args.name, "iteration": args.iteration, "positions_tag": data["tag"],
        "num_positions": len(boards),
        "search_optimal": sum(r["optimal"] for r in records) / len(boards),
        "correct": sum(r["optimal"] for r in records), "seconds": seconds,
        "oracle_seconds": oracle_seconds, "search_flags": {**mcts_flags, **evaluator_flags},
        "num_simulations": 400, "solver_max_undrawn": 30, "solver_node_budget": 100000,
        "seed": data["seed"], "torch_threads": 1,
        "positions_sha256": hashlib.sha256(args.positions.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "command": sys.argv,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main() -> None:
    """Sample development/held-out positions or evaluate one logged search configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    sampling = subs.add_parser("sample")
    sampling.add_argument("--tag", choices=("5x5-chains", "5x5-mid"), required=True)
    sampling.add_argument("--seed", type=int, default=0)
    sampling.add_argument("--num-positions", type=int, default=300)
    sampling.add_argument("--output", type=Path, required=True)
    sampling.set_defaults(function=sample)
    evaluation = subs.add_parser("evaluate")
    evaluation.add_argument("--positions", type=Path, required=True)
    evaluation.add_argument("--iteration", type=int, choices=(15, 54), default=54)
    evaluation.add_argument("--name", required=True)
    evaluation.add_argument("--limit", type=int, default=0, help="development timing pilot only")
    add_search_flags(evaluation)
    evaluation.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
