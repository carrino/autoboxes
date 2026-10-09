"""Tests for the exact endgame solver (Python reference) against the brute-force oracle."""
from __future__ import annotations

import random

import pytest

from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.boxes.solver import Solver, loony_value


def random_position(rows: int, cols: int, undrawn: int, rng: random.Random) -> BoxesBoard:
    board = BoxesBoard(rows, cols)
    while board.num_edges() - board.move_count() > undrawn:
        board.play_edge(rng.choice(board.get_legal_moves_flat()))
    return board


class TestLoonyValue:
    def test_single_component_is_simply_taken(self) -> None:
        for size in range(1, 7):
            assert loony_value((size,), ()) == -size
        assert loony_value((), (4,)) == -4 and loony_value((), (6,)) == -6

    def test_hand_values(self) -> None:
        # Two 3-chains: the controller takes 1, gives 2, then takes the other 3: opener -2.
        assert loony_value((3, 3), ()) == -2
        # Three 3-chains: Berlekamp's controlled value 3 * (3 - 4) + 4 = 1 for the controller.
        assert loony_value((3, 3, 3), ()) == -1
        # Two 4-loops: keeping control costs 4, so the controller takes all: even.
        assert loony_value((), (4, 4)) == 0
        # A 1-chain beside a long chain: open the 1-chain first (the taker loses control).
        assert loony_value((1, 5), ()) == max(-1 - loony_value((5,), ()), -5 + 4 - 1) == 4
        # 2-chains cannot be double-dealt (hard-hearted handout): same as a 1-chain plus one.
        assert loony_value((2, 5), ()) == 3

    def test_order_independent_and_memoised(self) -> None:
        assert loony_value((3, 5, 1), (4,)) == loony_value((1, 3, 5), (4,))
        assert loony_value((), ()) == 0


class TestSolver:
    @pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (2, 2)])
    def test_every_position_matches_oracle(self, rows: int, cols: int) -> None:
        oracle, solver = Oracle(rows, cols), Solver(rows, cols)
        for mask in range(1 << oracle.geo.num_edges):
            assert solver.value(mask) == oracle.value(mask), bin(mask)

    def test_two_by_three_matches_oracle(self) -> None:
        oracle, solver = Oracle(2, 3), Solver(2, 3)
        rng = random.Random(1)
        for _ in range(400):
            mask = rng.getrandbits(oracle.geo.num_edges)
            assert solver.value(mask) == oracle.value(mask), bin(mask)
        assert solver.value(0) == oracle.value(0)  # the empty 2x3 board, fully solved

    @pytest.mark.parametrize("rows,cols,undrawn,n",
                             [(3, 3, 14, 60), (5, 5, 10, 30), (2, 4, 12, 40)])
    def test_late_positions_match_oracle(self, rows: int, cols: int, undrawn: int, n: int) -> None:
        oracle, solver = Oracle(rows, cols), Solver(rows, cols)
        rng = random.Random(rows * 10 + cols)
        for _ in range(n):
            board = random_position(rows, cols, rng.randint(2, undrawn), rng)
            assert solver.remaining(board) == oracle.value(board.edges), board.render()
            assert solver.final_margin(board) == oracle.final_margin(board)

    def test_every_reduction_is_exact(self) -> None:
        # The loony leaf, the chain/loop equivalence and the canonical key each leave the
        # value unchanged relative to the bare search.
        bare = Solver(3, 3, use_leaf=False, use_equivalence=False, use_symmetry=False)
        variants = [Solver(3, 3), Solver(3, 3, use_leaf=False), Solver(3, 3, use_equivalence=False),
                    Solver(3, 3, use_symmetry=False)]
        rng = random.Random(7)
        for _ in range(60):
            board = random_position(3, 3, rng.randint(2, 13), rng)
            expected = bare.remaining(board)
            for s in variants:
                assert s.remaining(board) == expected, board.render()

    def test_pure_chain_position_has_the_loony_value(self) -> None:
        # 1x3 with every horizontal edge drawn: each box keeps its two vertical sides, so
        # the row is one 3-chain from ground to ground. The mover must open it: -3.
        solver = Solver(1, 3)
        board = BoxesBoard(1, 3)
        for e in range(6):  # h(0, c) and h(1, c)
            board.play_edge(e)
        assert solver.remaining(board) == loony_value((3,), ()) == -3
        assert solver.final_margin(board) == -3 and Oracle(1, 3).final_margin(board) == -3
        assert BoxesBoard(1, 3).margin() == 0 and solver.remaining(BoxesBoard(1, 3)) == -1

    def test_best_edge_is_optimal(self) -> None:
        oracle, solver = Oracle(2, 3), Solver(2, 3)
        rng = random.Random(3)
        for _ in range(80):
            board = random_position(2, 3, rng.randint(1, 10), rng)
            assert solver.best_edge(board) in oracle.best_edges(board), board.render()
