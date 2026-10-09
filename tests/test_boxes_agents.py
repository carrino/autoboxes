"""Tests for the built-in Boxes baselines."""
from __future__ import annotations

import random

import alpha_go_cpp
import pytest

from alpha_go.agents import get_agent
from alpha_go.boxes import agents as _agents  # noqa: F401  (registers boxes-* agents)
from alpha_go.boxes.agents import CAPTURE, LOONY, SAFE, AlphaBeta, classify, greedy_haul
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard, geometry


def cpp_from_moves(rows: int, cols: int, moves: list[int]) -> alpha_go_cpp.BoxesBoard:
    board = alpha_go_cpp.BoxesBoard(rows, cols)
    for e in moves:
        assert board.play_edge(e)
    return board


def play_out(first: str, second: str, rows: int, cols: int, seed: int) -> int:
    """Play one game between two registered agents; return the final margin for `first`."""
    board = alpha_go_cpp.BoxesBoard(rows, cols)
    agents = {1: get_agent(first), 2: get_agent(second)}
    while not board.is_game_over():
        row, col = agents[board.to_play()].select_move(board, seed + board.move_count())
        assert board.play(row, col)
    return int(board.score())


class TestClassification:
    def test_classes(self) -> None:
        geo = geometry(2, 2)
        board = cpp_from_moves(2, 2, [0, 2])  # box 0 has top and bottom
        assert classify(board, geo, geo.box_edges[0][2]) == LOONY  # third side
        assert classify(board, geo, geo.box_edges[3][0]) == SAFE
        board.play_edge(geo.box_edges[0][2])
        assert classify(board, geo, geo.box_edges[0][3]) == CAPTURE

    def test_greedy_haul_counts_a_chain(self) -> None:
        # 1x2 after 0,1,2,3,4: PLAYER_2 to move can take both boxes (5 then 6).
        board = cpp_from_moves(1, 2, [0, 1, 2, 3, 4])
        assert greedy_haul(board, geometry(1, 2)) == 2
        assert greedy_haul(alpha_go_cpp.BoxesBoard(2, 2), geometry(2, 2)) == 0


class TestAgents:
    def test_random_is_legal_and_seeded(self) -> None:
        agent = get_agent("boxes-random")
        board = alpha_go_cpp.BoxesBoard(3)
        move = agent.select_move(board, 7)
        assert board.is_legal(*move) and move == agent.select_move(board, 7)

    def test_greedy_takes_capture_and_avoids_third_side(self) -> None:
        agent = get_agent("boxes-greedy")
        board = cpp_from_moves(1, 2, [0, 1, 2, 3, 4])
        assert board.edge_index(*agent.select_move(board, 0)) == 5  # captures box A
        geo = geometry(3, 3)
        board = cpp_from_moves(3, 3, [0, 2])  # box 0 has two sides; its other sides are loony
        for seed in range(10):
            edge = board.edge_index(*agent.select_move(board, seed))
            assert classify(board, geo, edge) == SAFE

    def test_alpha_beta_matches_oracle_when_depth_covers_the_game(self) -> None:
        oracle = Oracle(2, 2)
        searcher = AlphaBeta(depth=12)
        rng = random.Random(3)
        for _ in range(20):
            py = BoxesBoard(2, 2)
            moves: list[int] = []
            for _ in range(rng.randint(0, 9)):
                e = rng.choice(py.get_legal_moves_flat())
                py.play_edge(e)
                moves.append(e)
            cpp = cpp_from_moves(2, 2, moves)
            edge = searcher.best_edge(cpp, random.Random(0))
            assert edge in oracle.best_edges(py), (py.render(), edge)

    def test_alpha_beta_beats_greedy_and_random(self) -> None:
        wins = sum(play_out("boxes-ab-d4", "boxes-greedy", 3, 3, seed) > 0 for seed in range(6))
        wins += sum(play_out("boxes-greedy", "boxes-ab-d4", 3, 3, seed) < 0 for seed in range(6))
        assert wins >= 9
        assert all(play_out("boxes-greedy", "boxes-random", 3, 3, seed) > 0 for seed in range(4))

    @pytest.mark.parametrize("name", ["boxes-ab-d2", "boxes-ab-d4", "boxes-ab-d6"])
    def test_registered_depths(self, name: str) -> None:
        agent = get_agent(name)
        board = alpha_go_cpp.BoxesBoard(2, 3)
        assert board.is_legal(*agent.select_move(board, 1))
        assert agent.searcher.depth == int(name[-1])

    @pytest.mark.parametrize("name", ["boxes-greedy-s24", "boxes-ab-d2-s24"])
    def test_solver_backed_baselines(self, name: str) -> None:
        # Perfect from 24 undrawn edges: on 2x3 that is the whole game, so moving first it
        # scores at least the game value and moving second it concedes at most that value
        # against alpha-beta depth 4. The solver phase accepts Python boards as well.
        agent = get_agent(name)
        cpp = alpha_go_cpp.BoxesBoard(3, 3)
        assert cpp.is_legal(*agent.select_move(cpp, 1))
        assert BoxesBoard(2, 3).is_legal(*agent.select_move(BoxesBoard(2, 3), 1))
        value = alpha_go_cpp.BoxesSolver(2, 3).value(0)
        assert all(play_out(name, "boxes-ab-d4", 2, 3, seed) >= value for seed in range(3))
        assert all(play_out("boxes-ab-d4", name, 2, 3, seed) <= value for seed in range(3))
