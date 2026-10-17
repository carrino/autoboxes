"""Measure bounded probes discarded by select_move on forced capture prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import alpha_go_cpp as cpp
import torch

from alpha_go.boxes.nn_agent import BoxesMCTSAgent


class NoInference:
    margin_utility_lambda = 0.5
    margin_utility_k = 6.0

    def batch_evaluate(self, boards):
        raise AssertionError("A forced capture must not reach neural search")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    BoxesMCTSAgent._solver_announced = True
    candidates = []
    for line in args.games.read_text().splitlines():
        game = json.loads(line)
        board = cpp.BoxesBoard(5, 5)
        prefix = []
        for row, col in game["moves"]:
            undrawn = 60 - board.move_count()
            state = cpp.BoxesSearchState(board)
            if 24 <= undrawn <= 38 and state.prefix() and not state.is_game_over():
                record = dict(game=game["index"], family=game["index"] // 2,
                              prefix=prefix[:], forced=state.prefix(),
                              undrawn=undrawn, after_undrawn=60-state.move_count())
                key = hashlib.sha256(json.dumps([20261030, record], sort_keys=True).encode())
                candidates.append((key.hexdigest(), record))
            edge = board.edge_index(row, col)
            prefix.append(edge)
            assert board.play_edge(edge)
    chosen, families = [], set()
    for _, record in sorted(candidates):
        if record["family"] not in families:
            families.add(record["family"])
            chosen.append(record)
        if len(chosen) == 24:
            break
    rows = []
    for record in chosen:
        board = cpp.BoxesBoard(5, 5)
        for edge in record["prefix"]:
            assert board.play_edge(edge)
        agent = BoxesMCTSAgent(NoInference(), solver_max_undrawn=34,
                              solver_node_budget=50_000, prove_terminals=True,
                              shape_terminals=True)
        steps = []
        for edge in record["forced"]:
            before = agent.solver.profile() if agent.solver is not None else [0] * 7
            started = time.perf_counter()
            move = agent.select_move(board, 20261030)
            elapsed = time.perf_counter() - started
            assert move == board.row_col(edge)
            assert agent.last_search_result is None
            after = agent.solver.profile()
            steps.append(dict(edge=edge, seconds=elapsed,
                              solver_profile=[b-a for a, b in zip(before, after)]))
            assert board.play_edge(edge)
        rows.append(dict(**record, steps=steps))
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    steps = [s for row in rows for s in row["steps"]]
    print(json.dumps(dict(candidates=len(candidates), families=len(families),
                         sequences=len(rows), forced_moves=sum(len(r["steps"]) for r in rows),
                         seconds=sum(s["seconds"] for r in rows for s in r["steps"]),
                         bounded_calls=sum(s["solver_profile"][0] for s in steps),
                         bounded_nodes=sum(s["solver_profile"][1] for s in steps),
                         aborted_calls=sum(s["solver_profile"][2] for s in steps)),
                     indent=2))


if __name__ == "__main__":
    main()
