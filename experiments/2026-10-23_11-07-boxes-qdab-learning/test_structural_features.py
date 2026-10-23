"""Semantic checks for the isolated structural descriptor prototype."""
from __future__ import annotations

import numpy as np
import pytest
from structural_features import feature_names, from_board, structural_features

from alpha_go.boxes.rules import BoxesBoard
from alpha_go.boxes.symmetry import edge_permutation, transforms


def values(board: BoxesBoard) -> dict[str, float]:
    return dict(zip(feature_names(board.geo), from_board(board).tolist()))


def draw_border(board: BoxesBoard) -> None:
    for edge, adjacent in enumerate(board.geo.edge_boxes):
        if len(adjacent) == 1:
            board.play_edge(edge)


def test_empty_board_has_safe_edges_and_junctions_not_chains() -> None:
    board = BoxesBoard(5)
    result = values(board)
    assert len(result) == 123
    assert result["degree_4"] == result["safe_edges"] == result["has_junction"] == 1
    assert result["quiet_closed_endgame"] == 0
    assert sum(result[name] for name in result if "_length_" in name) == 0


def test_loop_and_opened_loop_are_distinguished() -> None:
    board = BoxesBoard(2)
    draw_border(board)
    assert values(board)["loop_length_4"] == .25
    assert values(board)["quiet_closed_endgame"] == 1
    board.play_edge(board.geo.box_edges[0][3])
    result = values(board)
    assert result["loop_length_4"] == 0
    assert result["opened_path_length_4"] == result["endpoints_OO"] == .25
    assert result["capture_edges"] > 0
    assert result["quiet_closed_endgame"] == 0


def test_long_path_keeps_its_full_length() -> None:
    board = BoxesBoard(1, 5)
    draw_border(board)
    result = values(board)
    assert result["opened_path_length_5"] == pytest.approx(.2)
    assert result["opened_path_length_3"] == 0


def test_double_capture_and_score_perspective() -> None:
    board = BoxesBoard(1, 2)
    draw_border(board)
    result = values(board)
    assert result["double_capture_edges"] == pytest.approx(1 / board.num_edges())
    shared = next(edge for edge, boxes in enumerate(board.geo.edge_boxes) if len(boxes) == 2)
    board.play_edge(shared)
    result = values(board)
    assert result["mover_score"] == result["score_margin"] == 1
    assert result["opponent_score"] == result["unclaimed_boxes"] == 0
    assert result["quiet_closed_endgame"] == 0


def test_d4_invariance_and_player_swap_on_legal_histories() -> None:
    rng = np.random.default_rng(20261023)
    for _ in range(20):
        board = BoxesBoard(3)
        edges = rng.permutation(board.num_edges()).tolist()
        for edge in edges[:int(rng.integers(board.num_edges() + 1))]:
            board.play_edge(edge)
        mover = board.player()
        a, b = board.boxes[mover], board.boxes[1 - mover]
        baseline = from_board(board)
        for k in transforms(3, 3):
            permutation = edge_permutation(board.geo, k)
            transformed = sum(1 << int(permutation[edge]) for edge in range(board.num_edges())
                              if board.edges >> edge & 1)
            np.testing.assert_array_equal(
                baseline, structural_features(transformed, board.geo, a, b))
        swapped = dict(zip(feature_names(board.geo),
                           structural_features(board.edges, board.geo, b, a)))
        original = values(board)
        for name in original:
            if name == "mover_score":
                assert swapped[name] == original["opponent_score"]
            elif name == "opponent_score":
                assert swapped[name] == original["mover_score"]
            elif name == "score_margin":
                assert swapped[name] == -original[name]
            else:
                assert swapped[name] == original[name]


def test_score_mismatch_rejected() -> None:
    board = BoxesBoard(2)
    with pytest.raises(ValueError, match="Scores"):
        structural_features(board.edges, board.geo, 1, 0)
