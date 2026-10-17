"""The policy taught by self-play must respect the proofs used to select its move."""
from __future__ import annotations

from pathlib import Path

import alpha_go_cpp as cpp
import numpy as np
import pytest
import torch

from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import BoxesLeafEvaluator, BoxesMCTSAgent, BoxesSearchResult
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


def capture_board():
    board = cpp.BoxesBoard(1, 2)
    for edge in [0, 1, 2, 3, 4]:
        assert board.play_edge(edge)
    return board


def uniform(board):
    edges = board.get_legal_moves_flat()
    return {e: 1.0 / len(edges) for e in edges}, 0.5


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("shaped", [False, True])
def test_proven_losing_move_has_no_teaching_mass(batched: bool, shaped: bool) -> None:
    board = capture_board()
    config = cpp.MCTSConfig()
    config.prove_terminals = True
    config.terminal_margin_lambda = 0.5 if shaped else 0.0
    config.c_puct = 1.0
    tree = cpp.BoxesMCTSTree(board, config)
    if batched:
        tree.run_simulations_batched(50, 8, lambda bs: [uniform(b) for b in bs])
    else:
        tree.run_simulations(50, uniform)
    result = BoxesSearchResult(tree)
    raw = tree.get_child_visit_counts()
    assert raw[6] > 0, "Regression needs actual exploration of the losing move"
    assert tree.select_action(0.0) == 5
    assert result.training_visit_counts() == {5: raw[5], 6: 0}
    assert tree.get_child_visit_counts() == raw


@pytest.mark.parametrize("prove", [False, True])
def test_unproven_search_keeps_visit_weights(prove: bool) -> None:
    config = cpp.MCTSConfig()
    config.prove_terminals = prove
    tree = cpp.BoxesMCTSTree(cpp.BoxesBoard(3, 3), config)
    tree.run_simulations_batched(32, 8, lambda bs: [uniform(b) for b in bs])
    assert not tree.is_root_proven()
    assert BoxesSearchResult(tree).training_visit_counts() == tree.get_child_visit_counts()


def test_proofs_survive_game_record_npz_and_dataset(tmp_path: Path) -> None:
    board = capture_board()
    prefix = [board.row_col(e) for e in [0, 1, 2, 3, 4]]
    net = BoxesNet(1, 2, channels=8, n_blocks=1, value_hidden=8)
    def agent():
        return BoxesMCTSAgent(BoxesLeafEvaluator(net, torch.device("cpu")),
                              num_simulations=50, c_puct=1.0, leaf_batch_size=8,
                              forced_collapse=False, prove_terminals=True, temperature=0.0)
    record = play_game(agent(), agent(), board_size=1, board_cols=2,
                       start_moves=prefix, game=get_game("boxes"), collect_metrics=True)
    assert record.moves[0] == board.row_col(5)
    path = save_game_data(record, tmp_path, 0, "proof-policy")
    with np.load(path) as saved:
        assert saved["mcts_visits"][0, 6] == 0
        assert saved["mcts_raw_visits"][0, 6] > 0
        assert saved["policy_target_semantics"] == "proof_filtered_visits_v1"
    item = BoxesDataset([tmp_path])[0]
    assert item["policy"][5] == 1 and item["policy"][6] == 0
    py = BoxesBoard(1, 2)
    for e in [0, 1, 2, 3, 4]:
        py.play_edge(e)
    assert Oracle(1, 2).best_edges(py) == [5]


def test_temperature_zero_does_not_turn_an_unproven_target_into_one_hot() -> None:
    tree = cpp.BoxesMCTSTree(cpp.BoxesBoard(3, 3), cpp.MCTSConfig())
    tree.run_simulations_batched(64, 8, lambda bs: [uniform(b) for b in bs])
    assert sum(p > 0 for p in tree.get_action_probabilities(0).values()) == 1
    assert sum(n > 0 for n in BoxesSearchResult(tree).training_visit_counts().values()) > 1
