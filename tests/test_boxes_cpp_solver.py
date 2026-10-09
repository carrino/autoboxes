"""C++ `BoxesSolver`: parity with the Python reference and the oracle, node budgets, and the
solver-terminated search state inside the C++ MCTS."""
from __future__ import annotations

import random

import alpha_go_cpp
import pytest

from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.boxes.solver import Solver, loony_value


def random_cpp_position(rows: int, cols: int, undrawn: int, rng: random.Random
                        ) -> alpha_go_cpp.BoxesBoard:
    board = alpha_go_cpp.BoxesBoard(rows, cols)
    while board.num_edges() - board.move_count() > undrawn:
        board.play_edge(rng.choice(board.get_legal_moves_flat()))
    return board


def uniform_batched(states: list) -> list[tuple[dict[int, float], float]]:  # type: ignore[type-arg]
    out = []
    for s in states:
        moves = s.get_legal_moves_flat()
        out.append(({m: 1.0 / len(moves) for m in moves}, 0.5))
    return out


class TestCppSolver:
    @pytest.mark.parametrize("rows,cols", [(3, 3), (2, 3), (5, 5)])
    def test_canonical_key_matches_python(self, rows: int, cols: int) -> None:
        cpp, py = alpha_go_cpp.BoxesSolver(rows, cols, 1 << 10), Solver(rows, cols)
        rng = random.Random(rows + cols)
        for _ in range(200):
            mask = rng.getrandbits(py.geo.num_edges)
            assert cpp.canonical(mask) == py.canonical(mask)

    def test_loony_value_matches_python(self) -> None:
        cpp = alpha_go_cpp.BoxesSolver(3, 3, 1 << 10)
        rng = random.Random(5)
        for _ in range(150):
            chains = sorted(rng.randint(1, 6) for _ in range(rng.randint(0, 4)))
            loops = sorted(rng.choice([4, 6, 8]) for _ in range(rng.randint(0, 2)))
            assert cpp.loony_value(chains, loops) == loony_value(tuple(chains), tuple(loops))
        assert cpp.loony_value([3, 3, 3], []) == -1

    @pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (2, 2), (2, 3)])
    def test_every_position_matches_oracle(self, rows: int, cols: int) -> None:
        # Every mask of the board through one shared table: exercises the bounded entries.
        cpp, oracle = alpha_go_cpp.BoxesSolver(rows, cols, 1 << 16), Oracle(rows, cols)
        for mask in range(1 << oracle.geo.num_edges):
            assert cpp.value(mask) == oracle.value(mask), bin(mask)

    @pytest.mark.parametrize("rows,cols,undrawn", [(3, 3, 14), (5, 5, 12), (2, 4, 12)])
    def test_late_positions_match_python_solver(self, rows: int, cols: int, undrawn: int) -> None:
        cpp, py = alpha_go_cpp.BoxesSolver(rows, cols), Solver(rows, cols)
        rng = random.Random(undrawn)
        for _ in range(40):
            board = random_cpp_position(rows, cols, rng.randint(2, undrawn), rng)
            assert cpp.value(board.edges()) == py.value(board.edges()), board.render()
            assert cpp.final_margin(board) == board.margin() + py.value(board.edges())

    def test_principal_line_is_consistent_at_high_n(self) -> None:
        # No reference exists at 28 undrawn edges on 5x5, but the value must agree with the
        # line the solver itself plays: v(pos) = gain + v(after) on captures, -v(after) else.
        cpp = alpha_go_cpp.BoxesSolver(5, 5, 1 << 20)
        rng = random.Random(28)
        for _ in range(10):
            board = random_cpp_position(5, 5, 28, rng)
            while not board.is_game_over():
                value = cpp.value(board.edges())
                mover, before = board.player(), board.boxes(board.player())
                edge = cpp.best_edge(board)
                board.play_edge(edge)
                gain = board.boxes(mover) - before
                after = cpp.value(board.edges())
                assert value == (gain + after if board.player() == mover else -after)

    def test_node_budget(self) -> None:
        cpp = alpha_go_cpp.BoxesSolver(5, 5, 1 << 16)
        board = random_cpp_position(5, 5, 26, random.Random(9))
        assert cpp.value_within(board.edges(), 1) is None
        exact = cpp.value(board.edges())
        assert cpp.nodes() >= 1
        assert cpp.value_within(board.edges(), 10**9) == exact

    def test_best_edge_is_optimal(self) -> None:
        cpp, oracle = alpha_go_cpp.BoxesSolver(2, 3), Oracle(2, 3)
        rng = random.Random(3)
        for _ in range(60):
            board = random_cpp_position(2, 3, rng.randint(1, 10), rng)
            py = BoxesBoard(2, 3)
            for e in range(board.num_edges()):
                if (board.edges() >> e) & 1:
                    py.play_edge(e)
            assert cpp.best_edge(board) in oracle.best_edges(py), board.render()

    def test_table_footprint(self) -> None:
        cpp = alpha_go_cpp.BoxesSolver(5, 5, 1000)
        assert cpp.table_entries() == 1024 and cpp.table_bytes() == 1024 * 16


class TestSolvedSearchState:
    def test_late_state_is_terminal_with_the_exact_outcome(self) -> None:
        solver, oracle = alpha_go_cpp.BoxesSolver(3, 3), Oracle(3, 3)
        rng = random.Random(12)
        for _ in range(30):
            board = random_cpp_position(3, 3, rng.randint(1, 12), rng)
            state = alpha_go_cpp.BoxesSearchState(board, solver, 12, 10**7)
            assert state.is_game_over()
            if state.board().is_game_over():  # the forced captures finished the game
                continue
            assert state.solved()
            final = state.margin() + oracle.value(state.edges())
            assert state.solved_margin() == final
            expected = 1.0 if final > 0 else 0.5 if final == 0 else 0.0
            assert state.outcome(state.player()) == expected
            assert state.outcome(1 - state.player()) == 1.0 - expected
            plain = alpha_go_cpp.BoxesSearchState(board, solver, 0, 10**7)
            assert not plain.solved() and plain.is_game_over() == board.is_game_over()

    def test_children_inherit_the_solver(self) -> None:
        solver = alpha_go_cpp.BoxesSolver(3, 3)
        board = random_cpp_position(3, 3, 14, random.Random(2))
        state = alpha_go_cpp.BoxesSearchState(board, solver, 12, 10**7)
        assert not state.solved()
        child = state.copy()
        child.apply(state.get_legal_moves_flat()[0])
        assert child.solved() or child.is_game_over()

    def test_mcts_plays_perfectly_with_the_solver(self) -> None:
        # From 16 undrawn edges the search reaches solved states within two plies, so with
        # a uniform net the chosen move must have the oracle-optimal outcome.
        solver, oracle = alpha_go_cpp.BoxesSolver(3, 3), Oracle(3, 3)
        cfg = alpha_go_cpp.MCTSConfig()
        cfg.c_puct = 1.0
        cfg.dirichlet_alpha = 0.0
        rng = random.Random(16)
        for _ in range(12):
            board = random_cpp_position(3, 3, 16, rng)
            state = alpha_go_cpp.BoxesSearchState(board, solver, 14, 10**7)
            if state.prefix() or state.has_decision():
                continue
            tree = alpha_go_cpp.BoxesSearchMCTSTree(state, cfg)
            tree.run_simulations_batched(400, 8, uniform_batched)
            chosen = tree.select_action(0.0)
            outcomes = {
                e: (state.margin() + oracle.child_value(state.edges(), e) > 0)
                for e in state.get_legal_moves_flat()
            }
            assert outcomes[chosen] == max(outcomes.values()), (state.render(), chosen, outcomes)
