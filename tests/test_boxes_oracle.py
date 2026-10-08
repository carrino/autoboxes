"""Tests for the exhaustive Boxes oracle, cross-checked against an independent search."""
from __future__ import annotations

import random

import pytest

from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.boxes.symmetry import edge_permutation, transforms


def negamax_on_boards(board: BoxesBoard, memo: dict[int, int]) -> int:
    """Independent reference: same quantity as Oracle.value but walks BoxesBoard copies
    and reads captures off the board's box counts instead of bitmask arithmetic."""
    if board.is_game_over():
        return 0
    if board.edges in memo:
        return memo[board.edges]
    best = -board.geo.num_boxes
    for e in board.get_legal_moves_flat():
        child = board.copy()
        child.play_edge(e)
        gained = sum(child.boxes) - sum(board.boxes)
        rest = negamax_on_boards(child, memo)
        best = max(best, gained + rest if gained else -rest)
    memo[board.edges] = best
    return best


class TestKnownValues:
    def test_one_by_one_first_player_loses_the_box(self) -> None:
        assert Oracle(1, 1).value(0) == -1

    def test_double_capture_position(self) -> None:
        board = BoxesBoard(1, 2)
        for e in (0, 1, 2, 3, 4, 6):
            board.play_edge(e)
        oracle = Oracle(1, 2)
        assert oracle.value(board.edges) == 2 and oracle.best_edges(board) == [5]
        assert oracle.final_margin(board) == 2

    def test_take_then_take_again_beats_declining(self) -> None:
        # 1x2 after edges 0,1,2,3,4: PLAYER_2 to move. Edge 5 captures box A and keeps the
        # turn, then edge 6 captures box B: +2. Edge 6 first hands PLAYER_1 both boxes: -2.
        board = BoxesBoard(1, 2)
        for e in (0, 1, 2, 3, 4):
            board.play_edge(e)
        oracle = Oracle(1, 2)
        assert board.to_play() == 2
        assert oracle.child_value(board.edges, 5) == 2
        assert oracle.child_value(board.edges, 6) == -2
        assert oracle.best_edges(board) == [5]


class TestAgainstIndependentSearch:
    @pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (2, 2)])
    def test_every_reachable_position(self, rows: int, cols: int) -> None:
        oracle = Oracle(rows, cols)
        memo: dict[int, int] = {}
        rng = random.Random(7)
        for _ in range(25):
            board = BoxesBoard(rows, cols)
            while not board.is_game_over():
                assert oracle.value(board.edges) == negamax_on_boards(board, memo)
                board.play_edge(rng.choice(board.get_legal_moves_flat()))

    def test_two_by_three_matches_on_late_positions(self) -> None:
        oracle = Oracle(2, 3)
        memo: dict[int, int] = {}
        rng = random.Random(11)
        for _ in range(10):
            board = BoxesBoard(2, 3)
            for _ in range(rng.randint(9, 14)):
                board.play_edge(rng.choice(board.get_legal_moves_flat()))
            assert oracle.value(board.edges) == negamax_on_boards(board, memo)


class TestSymmetryInvariance:
    @pytest.mark.parametrize("rows,cols", [(1, 2), (2, 2)])
    def test_value_is_invariant(self, rows: int, cols: int) -> None:
        oracle = Oracle(rows, cols)
        rng = random.Random(3)
        for _ in range(20):
            moves = rng.sample(range(oracle.geo.num_edges), rng.randint(0, oracle.geo.num_edges))
            board = BoxesBoard(rows, cols)
            for e in moves:
                board.play_edge(e)
            for k in transforms(rows, cols):
                perm = edge_permutation(oracle.geo, k)
                mirror = BoxesBoard(rows, cols)
                for e in moves:
                    mirror.play_edge(int(perm[e]))
                assert oracle.value(mirror.edges) == oracle.value(board.edges)
