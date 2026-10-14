"""Exact relabelling of stored games matches the brute-force oracle and feeds the dataset."""
from __future__ import annotations

import random
from pathlib import Path

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import pytest

from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.relabel import EXACT_UNKNOWN, exact_margins_p1, relabel_file
from alpha_go.boxes.rules import geometry
from alpha_go.self_play import parse_score_from_result


def random_game(rows: int, cols: int, seed: int, start: int = 0) -> dict[str, np.ndarray | str]:
    """A random full game in the stored NPZ layout, its first `start` moves as start_moves."""
    rng = random.Random(seed)
    board = alpha_go_cpp.BoxesBoard(rows, cols)
    geo = geometry(rows, cols)
    moves, to_play, boards = [], [], []
    while not board.is_game_over():
        edge = rng.choice(board.get_legal_moves_flat())
        moves.append(geo.edge_rc[edge])
        to_play.append(int(board.player()))
        boards.append(np.asarray(board.to_numpy(), dtype=np.int8))
        board.play_edge(edge)
    score = round(board.score())
    result = "Draw" if score == 0 else f"{'B' if score > 0 else 'W'}+{abs(score)}.0"
    return {"moves": np.array(moves[start:], np.int16),
            "to_play": np.array(to_play[start:], np.int8), "boards": np.array(boards[start:]),
            "start_moves": np.array(moves[:start], np.int16), "result": result}


class TestExactMargins:
    @pytest.mark.parametrize("rows,cols,start", [(2, 2, 0), (2, 3, 0), (2, 3, 5)])
    def test_matches_oracle_and_outcome(self, rows: int, cols: int, start: int) -> None:
        oracle = Oracle(rows, cols)
        solver = alpha_go_cpp.BoxesSolver(rows, cols, 1 << 12)
        for seed in range(6):
            game = random_game(rows, cols, seed, start)
            reach = oracle.geo.num_edges - 3
            exact = exact_margins_p1(game, rows, cols, reach, 10**7, solver)
            board = alpha_go_cpp.BoxesBoard(rows, cols)
            for row, col in game["start_moves"]:
                board.play(int(row), int(col))
            for k, (row, col) in enumerate(game["moves"]):
                undrawn = board.num_edges() - board.move_count()
                if undrawn > reach:
                    assert exact[k] == EXACT_UNKNOWN
                else:
                    value = oracle.value(board.edges())
                    expected = round(board.score()) + (value if board.player() == 0 else -value)
                    assert exact[k] == expected
                board.play(int(row), int(col))
            # The last position is one move from the end: its exact margin is the outcome.
            assert exact[-1] == parse_score_from_result(str(game["result"]))

    def test_budget_leaves_unknown(self) -> None:
        solver = alpha_go_cpp.BoxesSolver(2, 3, 1 << 12)
        game = random_game(2, 3, 0)
        exact = exact_margins_p1(game, 2, 3, 17, 20, solver)
        assert exact[0] == EXACT_UNKNOWN and exact[-1] != EXACT_UNKNOWN


class TestDatasetUsesExact:
    def test_relabelled_file_changes_margins(self, tmp_path: Path) -> None:
        game = random_game(2, 3, 1)
        path = tmp_path / "game.npz"
        np.savez_compressed(path, **game, num_moves=len(game["moves"]), game="boxes")
        before = BoxesDataset([tmp_path])
        outcome = [int(before[i]["margin"]) for i in range(len(before))]
        exact = relabel_file(path, 2, 3, 12, 10**7, alpha_go_cpp.BoxesSolver(2, 3, 1 << 12))
        after = BoxesDataset([tmp_path])
        assert "exact_margin_p1" in after.games[0]
        margins = [int(after[i]["margin"]) for i in range(len(after))]
        for i, margin in enumerate(margins):
            sign = 1 if int(game["to_play"][i]) == 0 else -1
            # Above the reach the target is still the outcome; within it, the exact margin.
            assert margin == (outcome[i] if exact[i] == EXACT_UNKNOWN else sign * int(exact[i]))
        assert exact[0] == EXACT_UNKNOWN and exact[-1] != EXACT_UNKNOWN
        assert margins[-1] == outcome[-1] and margins != outcome
        assert after.num_exact() == int((exact != EXACT_UNKNOWN).sum())
        control = BoxesDataset([tmp_path], use_exact=False)
        assert [int(control[i]["margin"]) for i in range(len(control))] == outcome
        assert control.num_exact() == 0
