"""Tests for the pure-Python Dots and Boxes reference rules."""
from __future__ import annotations

import random

import numpy as np
import pytest

from alpha_go.boxes.rules import (
    PLAYER_1,
    PLAYER_2,
    BoxesBoard,
    BoxesState,
    geometry,
    perft,
)


class TestGeometry:
    @pytest.mark.parametrize(
        "rows,cols,edges", [(1, 1, 4), (1, 2, 7), (2, 2, 12), (2, 3, 17), (3, 3, 24), (5, 5, 60)]
    )
    def test_edge_count(self, rows: int, cols: int, edges: int) -> None:
        assert geometry(rows, cols).num_edges == edges

    def test_boxes_and_edges_are_consistent(self) -> None:
        geo = geometry(3, 4)
        assert geo.num_boxes == 12
        for sides in geo.box_edges:
            assert len(set(sides)) == 4
        counts = [0] * geo.num_edges
        for sides in geo.box_edges:
            for e in sides:
                counts[e] += 1
        assert all(1 <= n <= 2 for n in counts)
        assert sum(1 for n in counts if n == 2) == 2 * 3 * 4 - 3 - 4  # interior edges

    def test_lattice_coordinates(self) -> None:
        geo = geometry(2, 3)
        for e, (r, c) in enumerate(geo.edge_rc):
            assert (r % 2, c % 2) in ((0, 1), (1, 0))
            assert geo.lattice_edge[r, c] == e
        assert geo.edge_rc[geo.box_edges[4][0]] == (2 * 1, 2 * 1 + 1)  # box (1,1) top edge
        assert geo.edge_rc[geo.box_edges[4][2]] == (2 * 1 + 1, 2 * 1)  # box (1,1) left edge


class TestRules:
    def test_initial_state(self) -> None:
        board = BoxesBoard(3)
        assert board.to_play() == PLAYER_1 and board.player() == 0
        assert len(board.get_legal_moves_flat()) == 24
        assert not board.is_game_over() and board.score() == 0.0

    def test_non_capturing_move_switches_player(self) -> None:
        board = BoxesBoard(2)
        assert board.play_edge(0)
        assert board.to_play() == PLAYER_2
        assert not board.is_legal_edge(0)
        assert not board.play_edge(0)
        assert board.move_count() == 1

    def test_capture_gives_extra_move(self) -> None:
        board = BoxesBoard(2, 2)
        top, bottom, left, right = board.geo.box_edges[0]
        board.play_edge(top)  # P1
        board.play_edge(bottom)  # P2
        board.play_edge(left)  # P1
        board.play_edge(board.geo.box_edges[3][0])  # P2 elsewhere
        assert board.to_play() == PLAYER_1
        board.play_edge(right)  # P1 completes box 0
        assert board.owner[0] == PLAYER_1 and board.boxes == [1, 0]
        assert board.to_play() == PLAYER_1  # must move again
        assert board.margin() == 1
        board.play_edge(board.geo.box_edges[3][1])  # P1 draws elsewhere, no capture
        assert board.to_play() == PLAYER_2
        assert board.margin() == -1

    def test_double_capture_counts_two_and_ends_game(self) -> None:
        board = BoxesBoard(1, 2)
        # Edges: h(0,0)=0 h(0,1)=1 h(1,0)=2 h(1,1)=3 v(0,0)=4 v(0,1)=5 v(0,2)=6.
        for e in (0, 1, 2, 3, 4, 6):
            board.play_edge(e)
        assert board.to_play() == PLAYER_1 and board.boxes == [0, 0]
        board.play_edge(5)  # shared middle edge completes both boxes
        assert board.boxes == [2, 0]
        assert board.is_game_over()
        assert board.get_winner() == PLAYER_1 and board.score() == 2.0
        assert board.to_play() == PLAYER_1  # the capturer keeps the turn even at the end

    def test_one_by_one_second_player_always_captures(self) -> None:
        board = BoxesBoard(1)
        for e in (0, 1, 2, 3):
            board.play_edge(e)
        assert board.is_game_over() and board.get_winner() == PLAYER_2
        assert board.margin() == 1  # PLAYER_2 is to move and is ahead

    def test_lattice_play_and_numpy(self) -> None:
        board = BoxesBoard(2, 3)
        assert board.is_legal(0, 1) and not board.is_legal(0, 0) and not board.is_legal(1, 1)
        assert not board.is_legal(-1, 0) and not board.is_legal(0, 99)
        assert board.play(0, 1)
        grid = board.to_numpy()
        assert grid.shape == (5, 7) and grid.dtype == np.int8
        assert grid[0, 1] == 1 and grid.sum() == 1
        assert board.row_col(board.edge_index(0, 1)) == (0, 1)

    def test_render(self) -> None:
        board = BoxesBoard(1, 2)
        for e in (0, 1, 2, 3, 4, 6, 5):
            board.play_edge(e)
        assert board.render() == ".-.-.\n|1|1|\n.-.-."

    def test_copy_is_independent(self) -> None:
        board = BoxesBoard(2)
        other = board.copy()
        other.play_edge(0)
        assert board.is_legal_edge(0) and not other.is_legal_edge(0)


class TestPerft:
    @pytest.mark.parametrize(
        "rows,cols,depth,expected",
        [
            # Counted by hand: a box completed on move k has its 4 sides in 4! orders; the
            # 4th move is PLAYER_2's and the 5th is PLAYER_1's unless a capture intervened.
            (1, 1, 4, (24, 0, 24)),
            (1, 1, 5, (24, 0, 24)),
            (1, 2, 3, (210, 0, 0)),
            (1, 2, 4, (840, 0, 48)),
            (2, 2, 4, (11880, 0, 96)),
            (2, 2, 5, (95040, 3072, 768)),  # 4 boxes * 4 last sides * 8 extras * 24 orders
            # Full 1x2 games: regression value from this reference (2 boxes on every leaf).
            (1, 2, 7, (5040, 2592, 7488)),
        ],
    )
    def test_golden(self, rows: int, cols: int, depth: int, expected: tuple[int, int, int]) -> None:
        assert perft(BoxesBoard(rows, cols), depth) == expected


class TestRandomPlayInvariants:
    @pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (5, 5)])
    def test_invariants_hold_through_random_games(self, rows: int, cols: int) -> None:
        rng = random.Random(rows * 100 + cols)
        for _ in range(20):
            board = BoxesBoard(rows, cols)
            captures = 0
            while not board.is_game_over():
                before = sum(board.boxes)
                mover = board.to_play()
                edge = rng.choice(board.get_legal_moves_flat())
                assert board.play_edge(edge)
                gained = sum(board.boxes) - before
                captures += gained
                assert (board.to_play() == mover) == (gained > 0)
                assert bin(board.edges).count("1") == board.move_count()
                assert sum(1 for s in board.sides if s == 4) == sum(board.boxes)
            assert board.move_count() == board.num_edges()
            assert sum(board.boxes) == rows * cols == captures
            assert board.get_winner() == (
                PLAYER_1 if board.score() > 0 else PLAYER_2 if board.score() < 0 else 0
            )


class TestBoxesState:
    def test_protocol_round_trip(self) -> None:
        state = BoxesState.new_game(1, 2)
        assert state.current_player() == 0 and len(state.get_legal_actions()) == 7
        for e in (0, 1, 2, 3, 4, 6):
            state = state.apply_action(e)
        assert state.current_player() == 0 and not state.is_terminal()
        final = state.apply_action(5)
        assert final.is_terminal()
        assert final.get_reward(0) == 1.0 and final.get_reward(1) == 0.0
        assert not state.is_terminal()  # apply_action does not mutate
        assert final.clone().board is not final.board
