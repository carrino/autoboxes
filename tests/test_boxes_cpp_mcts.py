"""C++ `BoxesMCTSTree`: sign handling after captures, oracle agreement, Python parity."""
from __future__ import annotations

import random

import alpha_go_cpp
import numpy as np
import pytest

from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard, BoxesState
from alpha_go.mcts import MCTSConfig, run_mcts


def uniform(board: alpha_go_cpp.BoxesBoard) -> tuple[dict[int, float], float]:
    moves = board.get_legal_moves_flat()
    return {m: 1.0 / len(moves) for m in moves}, 0.5


def uniform_batched(boards: list[alpha_go_cpp.BoxesBoard]) -> list[tuple[dict[int, float], float]]:
    return [uniform(b) for b in boards]


def config(prove: bool = False) -> alpha_go_cpp.MCTSConfig:
    cfg = alpha_go_cpp.MCTSConfig()
    cfg.c_puct = 1.0
    cfg.dirichlet_alpha = 0.0
    cfg.prove_terminals = prove
    return cfg


def search(
    board: alpha_go_cpp.BoxesBoard, sims: int, batched: bool, prove: bool = False
) -> alpha_go_cpp.BoxesMCTSTree:
    tree = alpha_go_cpp.BoxesMCTSTree(board, config(prove))
    if batched:
        tree.run_simulations_batched(sims, 8, uniform_batched)
    else:
        tree.run_simulations(sims, uniform)
    return tree


def capture_position() -> alpha_go_cpp.BoxesBoard:
    board = alpha_go_cpp.BoxesBoard(1, 2)
    for e in (0, 1, 2, 3, 4):
        board.play_edge(e)
    return board


def random_late_position(rng: random.Random, rows: int, cols: int, left: int) -> list[int]:
    board = BoxesBoard(rows, cols)
    moves: list[int] = []
    while board.num_edges() - board.move_count() > left:
        edge = rng.choice(board.get_legal_moves_flat())
        board.play_edge(edge)
        moves.append(edge)
    return moves


class TestSignAfterCapture:
    @pytest.mark.parametrize("batched", [False, True])
    def test_capture_line_is_a_win_for_the_capturer(self, batched: bool) -> None:
        tree = search(capture_position(), 200, batched)
        q = tree.get_child_q_values()
        # Edge 5 captures a box and keeps the turn; the forced edge 6 wins 2-0 for PLAYER_2.
        assert q[5] > 0.9
        # Edge 6 hands both boxes to PLAYER_1: near 0 for PLAYER_2.
        assert q[6] < 0.3  # leaf batching averages a few more 0.5 leaf estimates into it
        assert tree.select_action(0.0) == 5
        assert tree.get_child_visit_counts()[5] > tree.get_child_visit_counts()[6]

    def test_tree_size_and_visits(self) -> None:
        tree = search(capture_position(), 50, False)
        assert tree.get_root_visit_count() == 50
        assert tree.tree_size() == 5  # root, two children, one forced grandchild each


class TestAgainstOracle:
    @pytest.mark.parametrize("batched", [False, True])
    def test_best_outcome_on_late_positions(self, batched: bool) -> None:
        oracle = Oracle(2, 2)
        rng = random.Random(9 + int(batched))
        for _ in range(30):
            moves = random_late_position(rng, 2, 2, rng.randint(2, 4))
            py, cpp = BoxesBoard(2, 2), alpha_go_cpp.BoxesBoard(2, 2)
            for e in moves:
                py.play_edge(e)
                cpp.play_edge(e)
            chosen = search(cpp, 400, batched).select_action(0.0)
            outcomes = {
                e: np.sign(py.margin() + oracle.child_value(py.edges, e))
                for e in py.get_legal_moves_flat()
            }
            assert outcomes[chosen] == max(outcomes.values()), (py.render(), chosen, outcomes)


class TestProofPropagation:
    """`prove_terminals`: exact leaves back up by minimax and settle the root."""

    def test_flag_is_off_by_default(self) -> None:
        assert alpha_go_cpp.MCTSConfig().prove_terminals is False
        tree = search(capture_position(), 50, False)
        assert not tree.is_root_proven() and tree.get_child_proven_values() == {}

    @pytest.mark.parametrize("batched", [False, True])
    def test_capture_line_is_proven(self, batched: bool) -> None:
        tree = search(capture_position(), 50, batched, prove=True)
        assert tree.get_child_proven_values()[5] == 1.0 and tree.get_child_q_values()[5] == 1.0
        # Proven the moment the winning line is: a loss for the opponent, who "moved" into it.
        assert tree.is_root_proven() and tree.get_root_proven_value() == 0.0
        probs = tree.get_action_probabilities(1.0)
        assert probs[5] == 1.0 and probs.get(6, 0.0) == 0.0
        assert tree.select_action(0.0) == 5

    def test_solver_leaves_prove_the_root(self) -> None:
        # 3x3 with nine undrawn edges and the solver at seven: every line reaches an exact leaf
        # within two plies, so the root is proven at the solver's own value.
        solver = alpha_go_cpp.BoxesSolver(3, 3, 1 << 16)
        rng = random.Random(23)
        checked = 0
        while checked < 8:
            cpp = alpha_go_cpp.BoxesBoard(3, 3)
            for e in random_late_position(rng, 3, 3, 9):
                cpp.play_edge(e)
            state = alpha_go_cpp.BoxesSearchState(cpp, solver, 7, 1 << 20, False)
            if state.solved():
                continue  # forced captures took it into the solver's zone
            tree = alpha_go_cpp.BoxesSearchMCTSTree(state, config(prove=True))
            tree.run_simulations_batched(400, 8, uniform_batched)
            expected = (np.sign(solver.final_margin(cpp)) + 1) / 2
            assert tree.is_root_proven(), cpp.render()
            assert max(tree.get_child_proven_values().values()) == expected, cpp.render()
            checked += 1

    @pytest.mark.parametrize("batched", [False, True])
    def test_exact_on_late_positions(self, batched: bool) -> None:
        # Two to four undrawn edges on 2x2: the tree is exhausted, the root's value is the
        # oracle's, and only optimal edges keep probability mass.
        oracle = Oracle(2, 2)
        rng = random.Random(17 + int(batched))
        for _ in range(30):
            moves = random_late_position(rng, 2, 2, rng.randint(2, 4))
            py, cpp = BoxesBoard(2, 2), alpha_go_cpp.BoxesBoard(2, 2)
            for e in moves:
                py.play_edge(e)
                cpp.play_edge(e)
            tree = search(cpp, 600, batched, prove=True)
            outcomes = {
                e: np.sign(py.margin() + oracle.child_value(py.edges, e))
                for e in py.get_legal_moves_flat()
            }
            best = max(outcomes.values())
            assert tree.is_root_proven(), py.render()
            assert max(tree.get_child_proven_values().values()) == (best + 1) / 2
            probs = tree.get_action_probabilities(1.0)
            assert all(outcomes[a] == best for a, p in probs.items() if p > 0), (py.render(), probs)

    def test_matches_python_reference(self) -> None:
        # Same positions, same uniform evaluator: both searches settle on the same exact value.
        rng = random.Random(29)
        cfg = MCTSConfig(c_puct=1.0, prove_terminals=True)
        for _ in range(20):
            moves = random_late_position(rng, 2, 3, rng.randint(2, 4))
            py, cpp = BoxesBoard(2, 3), alpha_go_cpp.BoxesBoard(2, 3)
            for e in moves:
                py.play_edge(e)
                cpp.play_edge(e)

            def py_uniform(state: BoxesState) -> tuple[dict[int, float], float]:
                actions = state.get_legal_actions()
                return {a: 1.0 / len(actions) for a in actions}, 0.5

            py_root = run_mcts(BoxesState(py), 600, cfg, py_uniform)
            tree = search(cpp, 600, False, prove=True)
            assert py_root.proven and tree.is_root_proven(), py.render()
            assert py_root.proven_value == tree.get_root_proven_value()
            py_values = {a: c.proven_value for a, c in py_root.children.items() if c.proven}
            cpp_values = tree.get_child_proven_values()
            assert all(py_values[a] == v for a, v in cpp_values.items() if a in py_values)


class TestPythonParity:
    def test_visit_distributions_are_close(self) -> None:
        """Same biased evaluator, same positions: Python and C++ trees agree.

        Biased priors (as in upstream's Go parity test) avoid exact PUCT ties, which the
        two implementations break in different orders.
        """

        def py_biased(state: BoxesState) -> tuple[dict[int, float], float]:
            w = {a: 1.0 / (a + 2) for a in state.get_legal_actions()}
            t = sum(w.values())
            return {a: x / t for a, x in w.items()}, 0.5

        def cpp_biased(board: alpha_go_cpp.BoxesBoard) -> tuple[dict[int, float], float]:
            w = {a: 1.0 / (a + 2) for a in board.get_legal_moves_flat()}
            t = sum(w.values())
            return {a: x / t for a, x in w.items()}, 0.5

        rng = random.Random(21)
        for _ in range(10):
            moves = random_late_position(rng, 2, 3, rng.randint(3, 6))
            py, cpp = BoxesBoard(2, 3), alpha_go_cpp.BoxesBoard(2, 3)
            for e in moves:
                py.play_edge(e)
                cpp.play_edge(e)
            py_root = run_mcts(BoxesState(py), 300, MCTSConfig(c_puct=1.0), py_biased)
            cpp_tree = alpha_go_cpp.BoxesMCTSTree(cpp, config())
            cpp_tree.run_simulations(300, cpp_biased)
            py_visits = {a: c.N for a, c in py_root.children.items()}
            cpp_visits = cpp_tree.get_child_visit_counts()
            total = sum(py_visits.values())
            assert total == sum(cpp_visits.values()) == 299
            tv = 0.5 * sum(
                abs(py_visits.get(a, 0) - cpp_visits.get(a, 0)) / total
                for a in set(py_visits) | set(cpp_visits)
            )
            assert tv < 0.2, (py.render(), tv)
            assert max(py_visits, key=py_visits.get) == max(cpp_visits, key=cpp_visits.get)
            assert abs(py_root.Q - cpp_tree.get_root_q_value()) < 0.1
