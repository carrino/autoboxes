"""C++ classical-engine parity, minimax, table reuse, and move deadline checks."""
from __future__ import annotations

import random
import time

import alpha_go_cpp  # type: ignore[import-not-found]
import pytest

import alpha_go.boxes.agents  # noqa: F401
from alpha_go.agents.base import get_agent
from alpha_go.boxes.classic import SCALE, remaining_estimate
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import geometry


class TestClassicEvaluation:
    @pytest.mark.parametrize("rows,cols", [(1, 1), (1, 3), (2, 3), (3, 3), (5, 5)])
    def test_matches_python_on_random_masks(self, rows: int, cols: int) -> None:
        geo = geometry(rows, cols)
        engine = alpha_go_cpp.BoxesClassic(rows, cols, 0.02, 0, 256, 256)
        rng = random.Random(17 * rows + cols)
        for _ in range(200):
            mask = rng.getrandbits(geo.num_edges)
            assert engine.evaluate(mask) == remaining_estimate(mask, geo) / SCALE
            assert engine.value(mask, 0) == engine.evaluate(mask)

    def test_score_margin_is_added(self) -> None:
        board = alpha_go_cpp.BoxesBoard(2, 3)
        engine = alpha_go_cpp.BoxesClassic(2, 3)
        rng = random.Random(8)
        while not board.is_game_over():
            board.play_edge(rng.choice(board.get_legal_moves_flat()))
            assert engine.final_margin(board) == board.margin() + engine.evaluate(board.edges())


class TestClassicSearch:
    @pytest.mark.parametrize("mask,expected", [(1001442305178077535, -5),
                                               (932763501780797775, 2)])
    def test_sliced_probes_finish_previously_heuristic_leaves(self, mask: int,
                                                            expected: int) -> None:
        # Development positions: a single 1024-node probe gave depth-one values
        # +2.12 and +4.57 respectively, instead of the exact remaining margins.
        solver = alpha_go_cpp.BoxesSolver(5, 5)
        assert solver.value(mask) == expected
        engine = alpha_go_cpp.BoxesClassic(5, 5)
        assert engine.value(mask, 1) == expected
        assert engine.solver_nodes() > 1024

    @pytest.mark.parametrize("entries", [2, 256])
    @pytest.mark.parametrize("rows,cols", [(1, 2), (2, 2), (2, 3)])
    def test_full_depth_matches_oracle_without_solver(self, entries: int,
                                                    rows: int, cols: int) -> None:
        oracle = Oracle(rows, cols)
        engine = alpha_go_cpp.BoxesClassic(rows, cols, 0.1, 0, entries, 256)
        rng = random.Random(23)
        masks = [rng.getrandbits(oracle.geo.num_edges) for _ in range(100)]
        if rows == 1:
            masks = list(range(1 << oracle.geo.num_edges))
        for mask in masks:
            assert engine.value(mask, oracle.geo.num_edges) == oracle.value(mask)

    @pytest.mark.parametrize("solver_reach", [0, 17])
    def test_selected_edge_is_optimal_on_small_boards(self, solver_reach: int) -> None:
        oracle = Oracle(2, 3)
        engine = alpha_go_cpp.BoxesClassic(2, 3, 0.1, solver_reach, 1 << 12, 1 << 12)
        rng = random.Random(91)
        for _ in range(30):
            board = alpha_go_cpp.BoxesBoard(2, 3)
            while board.move_count() < 8:
                board.play_edge(rng.choice(board.get_legal_moves_flat()))
            edge = engine.best_edge(board.edges(), 0)
            assert edge in board.get_legal_moves_flat()
            assert oracle.child_value(board.edges(), edge) == oracle.value(board.edges())

    def test_timeout_preserves_a_completed_iteration(self) -> None:
        board = alpha_go_cpp.BoxesBoard(5, 5)
        engine = alpha_go_cpp.BoxesClassic(5, 5, 0.03, 0)
        start = time.monotonic()
        edge = engine.best_edge(board.edges(), 54)
        elapsed = time.monotonic() - start
        depth, value = engine.completed_depth(), engine.last_value()
        assert elapsed < 0.5  # loose scheduling allowance, not a performance claim
        assert edge in board.get_legal_moves_flat() and depth >= 1
        assert engine.value(board.edges(), depth) == value

    def test_timeout_during_solver_probes_keeps_a_legal_move(self) -> None:
        mask = 928189155451622239  # quiet development position with 27 undrawn edges
        engine = alpha_go_cpp.BoxesClassic(5, 5, 0.03)
        start = time.monotonic()
        edge = engine.best_edge(mask, 0)
        elapsed = time.monotonic() - start
        assert 0 <= edge < 60 and not (mask >> edge) & 1
        assert engine.solver_nodes() > 0
        assert elapsed < 0.5  # allows scheduling delay; never run the full solve unchecked

    def test_zero_budget_and_terminal_board(self) -> None:
        engine = alpha_go_cpp.BoxesClassic(1, 1, 0.0)
        assert engine.best_edge(0, 12) in range(4)
        assert engine.completed_depth() == 0
        assert engine.best_edge(15, 12) == -1

    def test_agents_are_opt_in_and_rebuild_for_a_new_shape(self) -> None:
        for name in ("boxes-classic", "boxes-classic-fast", "boxes-classic-tiny"):
            agent = get_agent(name)
            for rows, cols in ((1, 1), (1, 2)):
                board = alpha_go_cpp.BoxesBoard(rows, cols)
                move = agent.select_move(board, 6)
                assert board.is_legal(*move)

    def test_bad_configuration_fails_clearly(self) -> None:
        with pytest.raises(ValueError):
            alpha_go_cpp.BoxesClassic(5, 5, -1.0)
        with pytest.raises(ValueError):
            alpha_go_cpp.BoxesClassic(5, 5, float("nan"))
        with pytest.raises(ValueError):
            alpha_go_cpp.BoxesClassic(5, 5, 0.1, -1)
