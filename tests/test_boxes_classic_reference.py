"""Mathematical checks of the classical evaluation before its C++ implementation."""
from __future__ import annotations

import itertools
import random

import pytest

from alpha_go.boxes.chains import degrees
from alpha_go.boxes.classic import SCALE, controlled_value, evaluate, remaining_estimate
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.solver import loony_value
from alpha_go.boxes.symmetry import edge_permutation, transforms


class TestControlledValue:
    def test_terminal_bonuses(self) -> None:
        assert controlled_value((), ()) == 0
        assert controlled_value((3,), ()) == 3
        assert controlled_value((), (4,)) == 4
        assert controlled_value((3,), (4,)) == 1
        assert controlled_value((4,), (4,)) == 0
        assert loony_value((3,), (4,)) == -1
        # A negative controlled value is not the controller's actual winning margin.
        assert controlled_value((3, 3, 3, 3, 3), ()) == -1
        assert loony_value((3, 3, 3, 3, 3), ()) == -1

    def test_controlled_value_theorem_against_exact_recurrence(self) -> None:
        """For a union of long chains/even loops, cv >= 2 implies exact value = -cv."""
        for count in range(7):
            for chain in itertools.combinations_with_replacement(range(3, 9), count):
                for nloops in range(4):
                    for loop in itertools.combinations_with_replacement((4, 6, 8), nloops):
                        if sum(chain) + sum(loop) > 25:
                            continue
                        cv = controlled_value(chain, loop)
                        if cv >= 2:
                            assert loony_value(chain, loop) == -cv, (chain, loop, cv)


class TestEvaluation:
    @pytest.mark.parametrize("rows,cols", [(1, 2), (2, 2)])
    def test_simple_endings_match_brute_force(self, rows: int, cols: int) -> None:
        geo, oracle = geometry(rows, cols), Oracle(rows, cols)
        for mask in range(1 << geo.num_edges):
            if max(degrees(mask, geo)) <= 2:
                assert remaining_estimate(mask, geo) == SCALE * oracle.value(mask)

    @pytest.mark.parametrize("rows,cols", [(2, 3), (3, 3), (5, 5)])
    def test_symmetry_bounds_and_score_offset(self, rows: int, cols: int) -> None:
        rng = random.Random(3100 + rows + cols)
        geo = geometry(rows, cols)
        for _ in range(50):
            board = BoxesBoard(rows, cols)
            for _ in range(rng.randrange(geo.num_edges + 1)):
                board.play_edge(rng.choice(board.get_legal_moves_flat()))
            value = remaining_estimate(board.edges, geo)
            remaining = sum(d > 0 for d in degrees(board.edges, geo))
            assert abs(value) <= SCALE * remaining
            assert evaluate(board) == board.margin() + value / SCALE
            for transform in transforms(rows, cols):
                perm = edge_permutation(geo, transform)
                image = sum(1 << int(perm[e]) for e in range(geo.num_edges)
                            if (board.edges >> e) & 1)
                assert remaining_estimate(image, geo) == value

    def test_capture_keeps_the_perspective(self) -> None:
        board = BoxesBoard(1, 1)
        for edge in range(3):
            board.play_edge(edge)
        assert evaluate(board) == 1.0
        board.play_edge(3)
        assert evaluate(board) == 1.0
