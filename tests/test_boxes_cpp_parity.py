"""C++ `alpha_go_cpp.BoxesBoard` must match the Python reference under random play."""
from __future__ import annotations

import random

import alpha_go_cpp
import numpy as np
import pytest

from alpha_go.boxes.rules import BoxesBoard, geometry, perft

SIZES = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (5, 5)]


def assert_same_state(py: BoxesBoard, cpp: alpha_go_cpp.BoxesBoard) -> None:
    assert cpp.to_play() == py.to_play() and cpp.player() == py.player()
    assert cpp.move_count() == py.move_count()
    assert [cpp.boxes(0), cpp.boxes(1)] == py.boxes
    assert cpp.edges() == py.edges
    assert cpp.get_legal_moves_flat() == py.get_legal_moves_flat()
    assert cpp.is_game_over() == py.is_game_over()
    assert cpp.score() == py.score() and cpp.margin() == py.margin()
    assert cpp.get_winner() == py.get_winner()
    assert [cpp.owner(b) for b in range(py.geo.num_boxes)] == py.owner
    assert [cpp.sides(b) for b in range(py.geo.num_boxes)] == py.sides
    assert np.array_equal(cpp.to_numpy(), py.to_numpy())
    assert cpp.render() == py.render()


class TestGeometryParity:
    @pytest.mark.parametrize("rows,cols", SIZES)
    def test_tables_match(self, rows: int, cols: int) -> None:
        geo = geometry(rows, cols)
        cpp = alpha_go_cpp.BoxesBoard(rows, cols)
        assert (cpp.rows(), cpp.cols(), cpp.num_edges(), cpp.num_boxes()) == (
            rows, cols, geo.num_edges, geo.num_boxes,
        )
        for e in range(geo.num_edges):
            assert tuple(cpp.row_col(e)) == geo.edge_rc[e]
        height, width = geo.lattice_shape
        for r in range(height):
            for c in range(width):
                assert cpp.edge_index(r, c) == int(geo.lattice_edge[r, c])

    def test_square_default_and_edge_limit(self) -> None:
        assert alpha_go_cpp.BoxesBoard(3).cols() == 3
        with pytest.raises(ValueError):
            alpha_go_cpp.BoxesBoard(6, 6)


class TestRandomPlayParity:
    @pytest.mark.parametrize("rows,cols", SIZES)
    def test_games_match_move_by_move(self, rows: int, cols: int) -> None:
        rng = random.Random(1000 * rows + cols)
        for _ in range(30):
            py, cpp = BoxesBoard(rows, cols), alpha_go_cpp.BoxesBoard(rows, cols)
            assert_same_state(py, cpp)
            while not py.is_game_over():
                edge = rng.choice(py.get_legal_moves_flat())
                assert py.play_edge(edge) and cpp.play_edge(edge)
                assert not cpp.play_edge(edge)
                assert_same_state(py, cpp)

    def test_lattice_play_and_copy(self) -> None:
        cpp = alpha_go_cpp.BoxesBoard(2, 3)
        assert cpp.is_legal(0, 1) and not cpp.is_legal(0, 0) and not cpp.is_legal(-1, 0)
        assert cpp.play(0, 1) and not cpp.play(0, 1)
        other = cpp.copy()
        assert other.play(0, 3) and not cpp.is_legal_edge(other.edge_index(0, 1))
        assert cpp.is_legal(0, 3)


class TestPerftParity:
    @pytest.mark.parametrize(
        "rows,cols,depth",
        [(1, 1, 4), (1, 2, 4), (1, 2, 7), (2, 2, 4), (2, 2, 5), (2, 3, 4), (3, 3, 3)],
    )
    def test_counts_match(self, rows: int, cols: int, depth: int) -> None:
        expected = perft(BoxesBoard(rows, cols), depth)
        cpp = alpha_go_cpp.BoxesBoard(rows, cols)
        assert tuple(alpha_go_cpp.boxes_perft(cpp, depth)) == expected
