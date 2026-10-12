"""Exact-leaf utility and proof semantics against the pure-Python margin oracle."""
from __future__ import annotations

import random
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import pytest

from alpha_go.boxes.nn_agent import BoxesSolvedResult
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.boxes.utility import exact_value


def uniform(board: Any) -> tuple[dict[int, float], float]:
    legal = board.get_legal_moves_flat()
    return {e: 1.0 / len(legal) for e in legal}, 0.5


def run(tree: Any, batched: bool, sims: int) -> None:
    if batched:
        tree.run_simulations_batched(sims, 4, lambda boards: [uniform(b) for b in boards])
    else:
        tree.run_simulations(sims, uniform)


def config(lam: float, prove: bool = True) -> Any:
    cfg = alpha_go_cpp.MCTSConfig()
    cfg.c_puct = 8.0  # enough exploration to settle every small-board child
    cfg.prove_terminals = prove
    cfg.terminal_margin_lambda = lam
    cfg.terminal_margin_k = 6.0
    return cfg


class TestTerminalUtility:
    @pytest.mark.parametrize("batched", [False, True])
    @pytest.mark.parametrize("prove", [False, True])
    @pytest.mark.parametrize("collapsed", [False, True])
    def test_natural_terminal_matches_reference(
        self, batched: bool, prove: bool, collapsed: bool,
    ) -> None:
        board = alpha_go_cpp.BoxesBoard(1, 2)
        for edge in range(board.num_edges()):
            board.play_edge(edge)
        state = alpha_go_cpp.BoxesSearchState(board) if collapsed else board
        factory = alpha_go_cpp.BoxesSearchMCTSTree if collapsed else alpha_go_cpp.BoxesMCTSTree
        tree = factory(state, config(2.0, prove))
        run(tree, batched, 8)
        assert tree.get_root_q_value() == pytest.approx(exact_value(-board.margin(), 2.0, 6.0))

    @pytest.mark.parametrize("batched", [False, True])
    @pytest.mark.parametrize("solver_on", [False, True])
    def test_minimax_margin_and_actions_match_python_oracle(
        self, batched: bool, solver_on: bool,
    ) -> None:
        rng = random.Random(740)
        oracle = Oracle(2, 3)
        solver = alpha_go_cpp.BoxesSolver(2, 3, 1 << 12)
        losing_roots = 0
        for trial in range(41):
            cfg = config(2.0)
            cfg.c_puct = 1000.0  # exhaust the tree, including dominated subtrees
            py, cpp = BoxesBoard(2, 3), alpha_go_cpp.BoxesBoard(2, 3)
            losing = [7, 14, 1, 6, 5, 9, 11, 15, 12, 10, 2, 4, 16]
            for i in range(py.num_edges() - 4):
                edge = losing[i] if trial == 40 else rng.choice(py.get_legal_moves_flat())
                py.play_edge(edge)
                cpp.play_edge(edge)
            if solver_on:
                state = alpha_go_cpp.BoxesSearchState(cpp, solver, 3, 100000, False)
                if state.is_game_over():
                    continue
                # Forced captures can change the board before the root is searched.
                for edge in state.prefix():
                    py.play_edge(edge)
                tree = alpha_go_cpp.BoxesSearchMCTSTree(state, cfg)
            else:
                tree = alpha_go_cpp.BoxesMCTSTree(cpp, cfg)
            outcomes = {
                e: py.margin() + oracle.child_value(py.edges, e)
                for e in py.get_legal_moves_flat()
            }
            best = max(outcomes.values())
            losing_roots += best < 0
            run(tree, batched, 4000)
            assert tree.is_root_proven(), py.render()
            assert tree.get_root_proven_value() == pytest.approx(exact_value(-best, 2.0, 6.0))
            probs = tree.get_action_probabilities(1.0)
            assert all(outcomes[e] == best for e, p in probs.items() if p > 0), (outcomes, probs)
        if not solver_on:
            assert losing_roots > 0  # negative utilities must mask to the least bad margin

    def test_one_winning_child_does_not_prove_best_margin(self) -> None:
        rng = random.Random(944)
        solver = alpha_go_cpp.BoxesSolver(2, 3, 1 << 12)
        for _ in range(200):
            board = alpha_go_cpp.BoxesBoard(2, 3)
            for _ in range(board.num_edges() - 5):
                board.play_edge(rng.choice(board.get_legal_moves_flat()))
            state = alpha_go_cpp.BoxesSearchState(board, solver, 4, 100000, False)
            if state.is_game_over() or state.prefix():
                continue
            values = {}
            for edge in state.get_legal_moves_flat():
                child = board.copy()
                child.play_edge(edge)
                margin = solver.final_margin(child)
                values[edge] = margin if child.player() == board.player() else -margin
            wins = [e for e, value in values.items() if 0 < value < max(values.values())]
            if not wins:
                continue
            first = wins[0]
            tree = alpha_go_cpp.BoxesSearchMCTSTree(state, config(1.0))
            tree.run_simulations(2, lambda b: (
                {e: 0.99 if e == first else 0.01 / (len(values) - 1) for e in values}, 0.5,
            ))
            assert tree.get_child_proven_values()[first] > 1.0
            assert not tree.is_root_proven()
            return
        pytest.fail("fixture did not find two distinct winning margins")

    def test_direct_solver_result_uses_the_same_scale(self) -> None:
        for margin in range(-25, 26):
            result = BoxesSolvedResult([2, 3], margin, 0.5, 6.0)
            assert result.Q == pytest.approx(exact_value(-margin, 0.5, 6.0))
            assert result.get_child_q_values() == {e: exact_value(margin, 0.5, 6.0) for e in [2, 3]}

    @pytest.mark.parametrize("batched", [False, True])
    def test_go_ignores_boxes_utility(self, batched: bool) -> None:
        board = alpha_go_cpp.GoBoard(3)
        plain = alpha_go_cpp.MCTSTree(board, config(0.0))
        shaped = alpha_go_cpp.MCTSTree(board, config(2.0))
        run(plain, batched, 100)
        run(shaped, batched, 100)
        assert shaped.get_child_visit_counts() == plain.get_child_visit_counts()
        assert shaped.get_child_q_values() == plain.get_child_q_values()
        assert shaped.get_root_q_value() == plain.get_root_q_value()
