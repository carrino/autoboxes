"""One-thread CPU timing and decision trace on two roots per undrawn band."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import alpha_go_cpp as cpp
import torch

from alpha_go.boxes.nn_agent import (
    BoxesLeafEvaluator,
    BoxesMCTSAgent,
    BoxesSearchResult,
    load_boxes_net,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = load_boxes_net(args.checkpoint, device)
    source = (Path(__file__).resolve().parent.parent /
              "2026-10-17_10-17-boxes-search-target-audit/data/positions.jsonl")
    records = [json.loads(line) for line in source.read_text().splitlines()]
    with args.output.open("w") as out:
        for position in records:
            if position["id"] % 8 >= 2:
                continue
            board = cpp.BoxesBoard(5, 5)
            for edge in position["prefix"]:
                assert board.play_edge(edge)
            evaluator = BoxesLeafEvaluator(model, device, margin_utility_lambda=0.5)
            agent = BoxesMCTSAgent(
                evaluator, num_simulations=400, c_puct=1.5, leaf_batch_size=16,
                temperature=0, add_noise=False, solver_max_undrawn=34,
                solver_node_budget=50000, prove_terminals=True, shape_terminals=True,
            )
            start = time.process_time()
            move = agent.select_move(board, 20261029 + position["id"])
            seconds = time.process_time() - start
            result = agent.last_search_result
            assert result is not None
            visits = result.tree.get_child_visit_counts()
            searched = isinstance(result, BoxesSearchResult)
            policy = result.tree.get_action_probabilities(1.0) if searched else {
                edge: count / sum(visits.values()) for edge, count in visits.items()}
            row = dict(id=position["id"], undrawn=position["undrawn"], cpu_seconds=seconds,
                       chosen=board.edge_index(*move), visits=visits, policy=policy,
                       root_proven=bool(searched and result.tree.is_root_proven()))
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(position["id"], seconds, flush=True)


if __name__ == "__main__":
    main()
