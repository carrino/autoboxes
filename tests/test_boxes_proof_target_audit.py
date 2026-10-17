"""Fail before training if a fresh collection does not contain proof-filter targets."""
# ruff: noqa: N806 (dimension-suffixed arrays)
from __future__ import annotations

from pathlib import Path

import alpha_go_cpp as cpp
import numpy as np
import pytest
import torch

from alpha_go.boxes.curriculum import digest
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import BoxesLeafEvaluator, BoxesMCTSAgent
from alpha_go.boxes.proof_target_audit import SEMANTICS, audit_collection
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


def collect_game(directory: Path, *, solved: bool = False) -> Path:
    torch.manual_seed(17)
    net = BoxesNet(1, 2, channels=8, n_blocks=1, value_hidden=8)
    board = cpp.BoxesBoard(1, 2)
    prefix = [] if solved else [board.row_col(edge) for edge in [0, 1, 2, 3, 4]]

    def agent():
        return BoxesMCTSAgent(BoxesLeafEvaluator(net, torch.device("cpu")),
                              num_simulations=50, c_puct=1.0, leaf_batch_size=8,
                              forced_collapse=solved, solver_max_undrawn=7 if solved else 0,
                              prove_terminals=True, temperature=0.0)

    record = play_game(agent(), agent(), board_size=1, board_cols=2,
                       start_moves=prefix, game=get_game("boxes"), collect_metrics=True)
    return save_game_data(record, directory, 0, "proof-audit")


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    return collect_game(tmp_path)


def test_actual_game_records_removed_losing_mass_without_mutation(collection: Path) -> None:
    before = digest(collection)
    with np.load(collection) as game:
        assert game["mcts_visits"][0, 6] == 0
        assert game["mcts_raw_visits"][0, 6] > 0
    report = audit_collection(collection.parent, 1)
    assert report["status"] == "complete" and report["games"] == 1
    assert report["totals"]["changed_positions"] == 1
    assert 0 < report["totals"]["mean_raw_mass_removed"] < 1
    assert report["bands"]["0..23"] == report["totals"]
    assert report["files_sha256"] == {str(collection.resolve()): before}
    assert digest(collection) == before


def test_solved_root_targets_and_unsearched_forced_captures_are_valid(tmp_path: Path) -> None:
    collect_game(tmp_path, solved=True)
    report = audit_collection(tmp_path, 1)
    assert report["totals"]["searched_positions"] > 0
    assert report["totals"]["unsearched_positions"] > 0
    assert report["totals"]["changed_positions"] == 0
    assert report["totals"]["mean_raw_mass_removed"] == 0


@pytest.mark.parametrize("semantics", [None, "raw_visits", [SEMANTICS]])
def test_missing_stale_or_nonscalar_semantics_rejected(collection: Path, semantics) -> None:
    with np.load(collection) as saved:
        game = dict(saved)
    game.pop("policy_target_semantics")
    if semantics is not None:
        game["policy_target_semantics"] = np.asarray(semantics)
    np.savez_compressed(collection, **game)
    with pytest.raises(ValueError, match="policy_target_semantics"):
        audit_collection(collection.parent, 1)


@pytest.mark.parametrize("problem", ["missing", "shape", "nan", "negative", "partial",
                                      "added", "erased"])
def test_invalid_visit_arrays_fail_before_training(collection: Path, problem: str) -> None:
    with np.load(collection) as saved:
        game = dict(saved)
    if problem == "missing":
        del game["mcts_raw_visits"]
    elif problem == "shape":
        game["mcts_raw_visits"] = game["mcts_raw_visits"][:, :-1]
    elif problem == "nan":
        game["mcts_raw_visits"] = game["mcts_raw_visits"].astype(float)
        game["mcts_raw_visits"][0, 0] = np.nan
    elif problem == "negative":
        game["mcts_raw_visits"][0, 0] = -1
    elif problem == "partial":
        game["mcts_visits"] = game["mcts_visits"].astype(float)
        game["mcts_visits"][0, 5] /= 2
    elif problem == "added":
        game["mcts_visits"][0, 0] = 1
    else:
        game["mcts_visits"][0] = 0
    np.savez_compressed(collection, **game)
    with pytest.raises(ValueError):
        audit_collection(collection.parent, 1)


def test_count_mismatch_and_unfinished_files_rejected(collection: Path) -> None:
    with pytest.raises(ValueError, match="Expected 2 finished games, found 1"):
        audit_collection(collection.parent, 2)
    temporary = collection.with_name("unfinished.tmp.npz")
    temporary.write_bytes(b"unfinished")
    with pytest.raises(ValueError, match="Unfinished"):
        audit_collection(collection.parent, 1)


def test_band_accounting_includes_unchanged_searched_rows(tmp_path: Path) -> None:
    undrawn = [24, 30, 31, 34, 35, 36, 37, 38, 39, 60]
    boards_NHW = np.zeros((len(undrawn), 11, 11), dtype=np.int8)
    board = cpp.BoxesBoard(5, 5)
    for index, remaining in enumerate(undrawn):
        for edge in range(60 - remaining):
            row, col = board.row_col(edge)
            boards_NHW[index, row, col] = 1
    raw_NE = np.zeros((len(undrawn), 60), dtype=np.int16)
    raw_NE[:, 58:] = 2
    filtered_NE = raw_NE.copy()
    filtered_NE[::2, 59] = 0
    np.savez_compressed(tmp_path / "game.npz", boards=boards_NHW,
                        policy_target_semantics=SEMANTICS, mcts_raw_visits=raw_NE,
                        mcts_visits=filtered_NE)
    report = audit_collection(tmp_path, 1)
    for name in ("24..30", "31..34", "35..36", "37..38", "39..60"):
        assert report["bands"][name] == dict(positions=2, searched_positions=2,
                                             changed_positions=1, unsearched_positions=0,
                                             mean_raw_mass_removed=0.25)
