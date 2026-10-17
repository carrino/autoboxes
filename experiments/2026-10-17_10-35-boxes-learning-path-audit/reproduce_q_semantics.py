"""Reproduce shaped search Q accepted as a win probability by ordinary replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import BoxesLeafEvaluator, BoxesMCTSAgent
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = BoxesNet(2, 2, channels=8, n_blocks=1, value_hidden=8)
    with torch.no_grad():
        # Every NN evaluation has P(win)=.75 and E[margin]=2. The shaped
        # utility .75 + .5*tanh(2/6) is still inside the probability range.
        model.value_fc2.bias.fill_(-30)
        model.value_fc2.bias[3] = np.log(0.25)  # margin -1
        model.value_fc2.bias[7] = np.log(0.75)  # margin +3
    results = {}
    for shaped in (False, True):
        folder = args.output / ("shaped" if shaped else "plain")
        folder.mkdir()

        def agent() -> BoxesMCTSAgent:
            evaluator = BoxesLeafEvaluator(model, torch.device("cpu"),
                                            margin_utility_lambda=0.5 if shaped else 0)
            return BoxesMCTSAgent(evaluator, num_simulations=3, leaf_batch_size=0,
                                  temperature=0, forced_collapse=False,
                                  prove_terminals=False, shape_terminals=shaped)

        record = play_game(agent(), agent(), board_size=2, game=get_game("boxes"),
                           seed=20261030, collect_metrics=True)
        path = save_game_data(record, folder, 0, "semantics")
        dataset = BoxesDataset([folder], min_undrawn=12)
        assert len(dataset) == 1
        row = dataset[0]
        # These are exactly the existing trainer's two q_mix guards.
        metadata_accepts = all(str(g.get("root_value_semantics", "")) != "shaped_utility"
                               for g in dataset.games)
        range_accepts = 0 <= float(row["root_value"]) <= 1
        results[folder.name] = dict(npz=path.name, termination=record.termination,
                                    rows=record.num_moves,
                                    saved_semantics=dataset.games[0].get("root_value_semantics"),
                                    first_root_value=float(row["root_value"]),
                                    metadata_guard_accepts=metadata_accepts,
                                    range_guard_accepts=range_accepts)
        assert metadata_accepts and range_accepts
    assert abs(results["plain"]["first_root_value"]
               - results["shaped"]["first_root_value"]) > 0.05
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
