"""C++ BoxesSearchState must match the Python collapse, and the searched agent must use it."""
from __future__ import annotations

import random

import alpha_go_cpp
import numpy as np
import pytest
import torch

from alpha_go.agents import get_agent
from alpha_go.boxes import agents as _boxes_agents  # noqa: F401
from alpha_go.boxes.forced import BoxesSearchState as PyState
from alpha_go.boxes.forced import collapse
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import BoxesLeafEvaluator, BoxesMCTSAgent
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.gameplay import play_game
from alpha_go.games import get_game

CPU = torch.device("cpu")


def random_pair(rows: int, cols: int, rng: random.Random) -> tuple[BoxesBoard, alpha_go_cpp.BoxesBoard]:
    py, cpp = BoxesBoard(rows, cols), alpha_go_cpp.BoxesBoard(rows, cols)
    for _ in range(rng.randint(0, py.num_edges() - 1)):
        e = rng.choice(py.get_legal_moves_flat())
        py.play_edge(e)
        cpp.play_edge(e)
    return py, cpp


class TestParity:
    @pytest.mark.parametrize("rows,cols", [(1, 3), (2, 2), (2, 3), (3, 3), (5, 5)])
    def test_collapse_matches_python(self, rows: int, cols: int) -> None:
        rng = random.Random(rows * 31 + cols)
        decisions = 0
        for _ in range(60):
            py, cpp = random_pair(rows, cols, rng)
            c = collapse(py)
            state = alpha_go_cpp.BoxesSearchState(cpp)
            assert list(state.prefix()) == c.prefix, py.render()
            assert state.has_decision() == (c.decision is not None)
            if c.decision is not None:
                decisions += 1
                assert list(state.decision_take()) == c.decision.take
                assert state.decision_control() == c.decision.control
                assert list(state.get_legal_moves_flat()) == c.decision.actions
            else:
                assert list(state.get_legal_moves_flat()) == c.board.get_legal_moves_flat()
            assert np.array_equal(state.to_numpy(), c.board.to_numpy())
            assert state.to_play() == c.board.to_play() and state.margin() == c.board.margin()
        assert decisions > 0 or rows * cols <= 3

    def test_apply_matches_python(self) -> None:
        rng = random.Random(5)
        for _ in range(30):
            py, cpp = random_pair(2, 3, rng)
            py_state = PyState.from_board(py)
            state = alpha_go_cpp.BoxesSearchState(cpp)
            for a in py_state.get_legal_actions()[:3]:
                child = py_state.apply_action(a)
                moved = state.copy()
                moved.apply(a)
                assert np.array_equal(moved.to_numpy(), child.to_numpy())
                assert list(moved.get_legal_moves_flat()) == child.get_legal_actions()

    def test_search_state_tree_has_oracle_value(self) -> None:
        """Negamax over C++ search states equals the oracle on 2x2 (collapse is value-preserving)."""
        oracle = Oracle(2, 2)
        memo: dict[tuple[int, int], int] = {}

        def value(state: alpha_go_cpp.BoxesSearchState) -> int:
            if state.is_game_over():
                return 0
            key = (state.edges(), state.player())
            if key in memo:
                return memo[key]
            best = -99
            me = state.player()
            for a in state.get_legal_moves_flat():
                child = state.copy()
                child.apply(a)
                gained = child.boxes(me) - state.boxes(me)
                lost = child.boxes(1 - me) - state.boxes(1 - me)
                rest = value(child)
                best = max(best, gained - lost + (rest if child.player() == me else -rest))
            memo[key] = best
            return best

        rng = random.Random(8)
        for _ in range(30):
            py, cpp = random_pair(2, 2, rng)
            state = alpha_go_cpp.BoxesSearchState(cpp)
            prefix_gain = state.boxes(py.player()) - py.boxes[py.player()]
            assert prefix_gain + value(state) == oracle.value(py.edges), py.render()


class TestAgentWithCollapse:
    def test_forced_prefix_is_played_without_search(self) -> None:
        net = BoxesNet(1, 2, channels=8, n_blocks=1, value_hidden=8)
        agent = BoxesMCTSAgent(BoxesLeafEvaluator(net, CPU), num_simulations=8)
        board = alpha_go_cpp.BoxesBoard(1, 2)
        for e in (0, 1, 2, 3, 4, 6):
            board.play_edge(e)
        assert board.edge_index(*agent.select_move(board, 0)) == 5  # hard-hearted: just take
        assert agent.last_search_result is None

    def test_decision_is_searched_and_capture_line_wins(self) -> None:
        net = BoxesNet(1, 2, channels=8, n_blocks=1, value_hidden=8)
        agent = BoxesMCTSAgent(BoxesLeafEvaluator(net, CPU), num_simulations=64, temperature=0.0)
        board = alpha_go_cpp.BoxesBoard(1, 2)
        for e in (0, 1, 2, 3, 4):
            board.play_edge(e)
        assert board.edge_index(*agent.select_move(board, 0)) == 5
        assert agent.last_search_result is not None
        assert set(agent.last_search_result.tree.get_child_visit_counts()) == {5, 6}

    def test_full_game_with_metrics(self) -> None:
        record = play_game(
            get_agent("boxes-mcts-untrained-3x3"), get_agent("boxes-greedy"), board_size=3,
            seed=1, max_moves=24, game=get_game("boxes"), collect_metrics=True,
        )
        assert record.num_moves == 24
        searched = [m for m in record.move_metrics if m.visit_counts is not None]
        forced = [m for m in record.move_metrics[::2] if m.visit_counts is None]
        assert searched and all(m.visit_counts.sum() == 32 for m in searched)
        assert len(searched) + len(forced) == 12
