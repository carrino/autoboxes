"""Tests for Boxes lattice symmetries and the induced edge permutations."""
from __future__ import annotations

import random

import numpy as np
import pytest

from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.symmetry import apply, edge_permutation, inverse, transforms


def _replay(rows: int, cols: int, edges: list[int]) -> BoxesBoard:
    board = BoxesBoard(rows, cols)
    for e in edges:
        assert board.play_edge(e)
    return board


class TestGroup:
    def test_transform_counts(self) -> None:
        assert transforms(3, 3) == list(range(8))
        assert transforms(2, 3) == [0, 2, 4, 6]

    @pytest.mark.parametrize("k", range(8))
    def test_inverse_round_trip(self, k: int) -> None:
        grid = np.arange(5 * 7).reshape(5, 7) if k in (0, 2, 4, 6) else np.arange(49).reshape(7, 7)
        assert np.array_equal(apply(apply(grid, k), inverse(k)), grid)

    def test_shape_preserving_transforms_keep_shape(self) -> None:
        grid = np.zeros((5, 7))
        for k in transforms(2, 3):
            assert apply(grid, k).shape == (5, 7)
        assert apply(grid, 1).shape == (7, 5)

    def test_batched_axes(self) -> None:
        planes = np.random.default_rng(0).random((2, 3, 7, 7))
        assert np.array_equal(apply(planes, 5)[1, 2], apply(planes[1, 2], 5))


class TestEdgePermutation:
    @pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (2, 3), (3, 3), (5, 5)])
    def test_is_a_permutation(self, rows: int, cols: int) -> None:
        geo = geometry(rows, cols)
        for k in transforms(rows, cols):
            perm = edge_permutation(geo, k)
            assert sorted(perm.tolist()) == list(range(geo.num_edges))

    def test_identity_and_inverse(self) -> None:
        geo = geometry(3, 3)
        assert np.array_equal(edge_permutation(geo, 0), np.arange(geo.num_edges))
        for k in transforms(3, 3):
            perm, back = edge_permutation(geo, k), edge_permutation(geo, inverse(k))
            assert np.array_equal(back[perm], np.arange(geo.num_edges))

    def test_quarter_turn_swaps_edge_orientation(self) -> None:
        geo = geometry(3, 3)
        perm = edge_permutation(geo, 1)
        horizontal = (geo.rows + 1) * geo.cols
        for e in range(geo.num_edges):
            assert (e < horizontal) != (perm[e] < horizontal)


class TestBoardEquivariance:
    @pytest.mark.parametrize("rows,cols", [(2, 2), (2, 3), (3, 3)])
    def test_transformed_game_is_transformed_board(self, rows: int, cols: int) -> None:
        rng = random.Random(rows * 10 + cols)
        geo = geometry(rows, cols)
        for _ in range(10):
            moves = rng.sample(range(geo.num_edges), rng.randint(1, geo.num_edges))
            board = _replay(rows, cols, moves)
            legal = np.array([board.is_legal_edge(e) for e in range(geo.num_edges)])
            for k in transforms(rows, cols):
                perm = edge_permutation(geo, k)
                mirror = _replay(rows, cols, [int(perm[e]) for e in moves])
                assert np.array_equal(apply(board.to_numpy(), k), mirror.to_numpy())
                assert mirror.boxes == board.boxes and mirror.to_play() == board.to_play()
                legal_sym = np.empty_like(legal)
                legal_sym[perm] = legal
                assert np.array_equal(
                    legal_sym, [mirror.is_legal_edge(e) for e in range(geo.num_edges)]
                )
