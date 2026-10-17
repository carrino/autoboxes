"""Audit real arena roots: solver cost and raw versus proof-filtered teaching policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import alpha_go_cpp as cpp
import torch

from alpha_go.boxes.nn_agent import (
    BoxesLeafEvaluator,
    BoxesMCTSAgent,
    BoxesSearchResult,
    load_boxes_net,
)


def sample(games: Path, output: Path) -> None:
    """Freeze eight quiet roots per band from deterministically shuffled arena prefixes."""
    records = [json.loads(line) for line in games.read_text().splitlines()]
    selected = []
    for undrawn in [32, 34, 35, 36, 38, 40]:
        candidates = []
        for game in records:
            board = cpp.BoxesBoard(5, 5)
            prefix = []
            for row, col in game["moves"][:60 - undrawn]:
                edge = board.edge_index(row, col)
                assert board.play_edge(edge)
                prefix.append(edge)
            if cpp.BoxesSearchState(board).prefix():
                continue
            record = dict(game=game["index"], family=game["index"] // 2,
                          undrawn=undrawn, prefix=prefix)
            serialized = json.dumps([20261029, record], sort_keys=True).encode()
            key = hashlib.sha256(serialized).hexdigest()
            candidates.append((key, record))
        families = set()
        for _, record in sorted(candidates):
            if record["family"] in families:
                continue
            families.add(record["family"])
            selected.append(record)
            if len(families) == 8:
                break
        assert len(families) == 8
    output.write_text("".join(json.dumps(dict(id=i, **r)) + "\n" for i, r in enumerate(selected)))


class TimedEvaluator(BoxesLeafEvaluator):
    seconds: float = 0.0
    leaves: int = 0

    def batch_evaluate(self, boards: list[Any]) -> Any:
        started = time.perf_counter()
        result = super().batch_evaluate(boards)
        self.seconds += time.perf_counter() - started
        self.leaves += len(boards)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--solver", type=int, default=34)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.games:
        sample(args.games, args.positions)
        return
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = load_boxes_net(args.checkpoint, device)
    positions = [json.loads(line) for line in args.positions.read_text().splitlines()]
    if args.limit:
        positions = positions[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for position in positions:
            board = cpp.BoxesBoard(5, 5)
            for edge in position["prefix"]:
                assert board.play_edge(edge)
            evaluator = TimedEvaluator(model, device, margin_utility_lambda=0.5)
            agent = BoxesMCTSAgent(evaluator, num_simulations=400, c_puct=1.5,
                                  leaf_batch_size=16, temperature=0.0, add_noise=False,
                                  solver_max_undrawn=args.solver, solver_node_budget=50000,
                                  prove_terminals=True, shape_terminals=True)
            started = time.perf_counter()
            move = agent.select_move(board, 20261029 + position["id"])
            seconds = time.perf_counter() - started
            result = agent.last_search_result
            raw = result.tree.get_child_visit_counts() if result is not None else {}
            total = sum(raw.values())
            search = isinstance(result, BoxesSearchResult)
            policy = result.tree.get_action_probabilities(1.0) if search else {
                e: n / total for e, n in raw.items()}
            removed = sum(n for e, n in raw.items() if policy.get(e, 0) == 0)
            record = dict(**position, solver=args.solver, seconds=seconds,
                          evaluator_seconds=evaluator.seconds, evaluator_leaves=evaluator.leaves,
                          chosen=board.edge_index(*move), searched=search,
                          root_proven=bool(search and result.tree.is_root_proven()),
                          discarded_target_mass=removed / total if total else 0,
                          visits=raw, selection_policy=policy)
            if hasattr(agent.solver, "profile"):
                record["solver_profile"] = agent.solver.profile()
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"solver{args.solver}: root {position['id']}, {position['undrawn']} left, "
                  f"{seconds:.2f}s, removed target mass {record['discarded_target_mass']:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
