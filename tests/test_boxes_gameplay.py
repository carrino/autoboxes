"""Shared play loop, self-play CLI and arena driving Boxes."""
from __future__ import annotations

import sys
from pathlib import Path

import alpha_go_cpp
import numpy as np

from alpha_go import self_play
from alpha_go.agents import get_agent
from alpha_go.agents.base import register_agent
from alpha_go.boxes import agents as _boxes_agents
from alpha_go.boxes.arena import play_match, random_opening
from alpha_go.boxes.nn_agent import register_boxes_mcts_agent
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


class TestPlayGame:
    def test_boxes_game_record(self) -> None:
        record = play_game(
            get_agent("boxes-greedy"), get_agent("boxes-random"), board_size=3, seed=1,
            max_moves=24, game=get_game("boxes"),
        )
        assert record.game == "boxes" and record.num_moves == 24
        assert record.termination == "all_edges"
        assert len(record.boards) == len(record.to_play) == 24
        assert record.boards[0].shape == (7, 7)
        assert record.to_play[0] == 0 and set(record.to_play) <= {0, 1}
        assert record.black_move_count + record.white_move_count == 24
        assert record.winner in (1, 2) and record.result[0] in "BW"  # 9 boxes: no ties

    def test_rectangular_board(self) -> None:
        record = play_game(
            get_agent("boxes-random"), get_agent("boxes-random"), board_size=2, board_cols=3,
            seed=0, max_moves=17, game=get_game("boxes"),
        )
        assert record.num_moves == 17 and record.boards[0].shape == (5, 7)

    def test_go_record_unchanged_plus_to_play(self) -> None:
        record = play_game(
            get_agent("random"), get_agent("random"), board_size=9, seed=0, max_moves=20
        )
        assert record.game == "go" and record.termination in ("double_pass", "max_moves")
        assert record.to_play == [i % 2 for i in range(record.num_moves)]

    def test_npz_has_game_and_to_play(self, tmp_path: Path) -> None:
        record = play_game(
            get_agent("boxes-random"), get_agent("boxes-random"), board_size=2, seed=3,
            max_moves=12, game=get_game("boxes"),
        )
        path = save_game_data(record, tmp_path, 0, "test")
        data = np.load(path)
        assert str(data["game"]) == "boxes" and data["to_play"].dtype == np.int8
        assert data["to_play"].shape == (12,) and data["boards"].shape == (12, 5, 5)


class TestSolvedTermination:
    def test_game_ends_with_the_exact_outcome_once_solved(self) -> None:
        """2x2 with the solver covering the whole game: the first searched position is solved,
        so the game ends after one move with the oracle's final margin as its result."""
        name = register_boxes_mcts_agent("boxes-mcts-test-solved-stop", None, 2, device="cpu",
                                         net_kwargs=dict(channels=8, n_blocks=1),
                                         num_simulations=2, solver_max_undrawn=12,
                                         solver_node_budget=1 << 20)
        oracle = Oracle(2, 2)
        for seed in range(3):
            prefix = [(0, 1), (1, 0)] if seed else []
            record = play_game(get_agent(name), get_agent(name), board_size=2, seed=seed,
                               max_moves=12, game=get_game("boxes"), collect_metrics=True,
                               start_moves=prefix, stop_when_solved=True)
            py = BoxesBoard(2, 2)
            for r, c in prefix:
                py.play_edge(int(geometry(2, 2).lattice_edge[r, c]))
            final = oracle.final_margin(py)  # for the side to move at the branch point
            score = final if py.to_play() == 1 else -final
            assert record.termination == "solved" and record.num_moves == 1
            assert record.start_moves == prefix and len(record.boards) == 1
            assert record.result == f"{'B' if score > 0 else 'W'}+{abs(score):.1f}"
            assert record.move_metrics[0].visit_counts.sum() >= 1  # the uniform-optimal row

    def test_record_opening_with_an_unsearched_move_saves(self, tmp_path: Path) -> None:
        """A branched game can open with a forced capture (no search): the NPZ's action
        vectors are sized from the first searched move, not Go's board-size formula."""
        name = register_boxes_mcts_agent("boxes-mcts-test-forced-open", None, 3, device="cpu",
                                         net_kwargs=dict(channels=8, n_blocks=1),
                                         num_simulations=4)
        # Three sides of the top-left box drawn: the mover's first move is a forced capture.
        prefix = [(0, 1), (1, 0), (1, 2)]
        record = play_game(get_agent(name), get_agent("boxes-greedy"), board_size=3, seed=0,
                           max_moves=24, game=get_game("boxes"), collect_metrics=True,
                           start_moves=prefix)
        assert record.move_metrics[0].visit_counts is None
        data = np.load(save_game_data(record, tmp_path, 0, "t"))
        assert data["mcts_visits"].shape == (record.num_moves, 24)


class TestSelfPlayCLI:
    def test_boxes_games_saved(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(self_play, "GAME_DATA_DIR", tmp_path)
        monkeypatch.setattr(sys, "argv", [
            "self_play", "--game", "boxes", "--board_size", "3", "--black", "boxes-greedy",
            "--white", "boxes-random", "--num_games", "2", "--save-name", "dev", "--seed", "0",
        ])
        self_play.main()
        files = sorted((tmp_path / "dev").glob("*.npz"))
        assert len(files) == 2
        data = np.load(files[0])
        assert int(data["num_moves"]) == 24 and str(data["termination"]) == "all_edges"

    def test_branched_games_record_the_prefix_and_replay(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(self_play, "GAME_DATA_DIR", tmp_path)
        monkeypatch.setattr(sys, "argv", [
            "self_play", "--game", "boxes", "--board_size", "3", "--black", "boxes-greedy",
            "--white", "boxes-random", "--num_games", "2", "--save-name", "src", "--seed", "0",
        ])
        self_play.main()
        monkeypatch.setattr(sys, "argv", [
            "self_play", "--game", "boxes", "--board_size", "3", "--black", "boxes-greedy",
            "--white", "boxes-random", "--num_games", "3", "--save-name", "branched", "--seed", "7",
            "--start-positions", "src", "--start-undrawn-min", "10", "--start-undrawn-max", "14",
        ])
        self_play.main()
        sys.path.insert(0, str(Path("experiments/2026-10-08_14-42-boxes-3x3-loop").resolve()))
        from oracle_eval import game_moves
        for path in sorted((tmp_path / "branched").glob("*.npz")):
            data = np.load(path)
            prefix, moves = data["start_moves"], int(data["num_moves"])
            assert 10 <= len(prefix) <= 14 and len(prefix) + moves == 24  # no solver: to the end
            assert data["boards"].shape[0] == moves
            board = alpha_go_cpp.BoxesBoard(3, 3)
            for r, c in game_moves(data):
                board.play(r, c)
            assert board.is_game_over() and np.array_equal(
                data["boards"][0], self._after(prefix))

    @staticmethod
    def _after(prefix: np.ndarray) -> np.ndarray:
        board = alpha_go_cpp.BoxesBoard(3, 3)
        for r, c in prefix:
            board.play(int(r), int(c))
        return np.asarray(board.to_numpy())

    def test_default_max_moves_covers_every_edge(self) -> None:
        assert get_game("boxes").default_max_moves(5) == 60
        assert get_game("boxes").default_max_moves(2, 3) == 17
        assert get_game("go").default_max_moves(9) == 162


class TestArena:
    def test_match_alternates_and_reports(self) -> None:
        result = play_match(
            "boxes-greedy", "boxes-random", rows=3, num_games=4, seed=0, num_workers=2
        )
        assert result.games == 4 and result.a_wins + result.b_wins + result.draws == 4
        assert len(result.margins) == 4 and result.a_wins >= 3
        summary = result.summary()
        lo, hi = result.wilson()
        assert 0.0 <= lo <= result.a_win_rate <= hi <= 1.0
        assert summary["board"] == "3x3" and summary["a_sec_per_move"] >= 0

    def test_random_opening_is_legal_and_seeded(self) -> None:
        moves = random_opening(3, 3, 5, np.random.default_rng(3))
        assert moves == random_opening(3, 3, 5, np.random.default_rng(3))
        assert len(moves) == 5 and len(set(moves)) == 5
        board = alpha_go_cpp.BoxesBoard(3, 3)
        for r, c in moves:
            board.play(r, c)
        assert board.move_count() == 5

    def test_pairs_share_an_opening_and_swap_colours(self) -> None:
        seen: list[tuple[int, ...]] = []

        @register_agent("boxes-test-opening-probe")
        class _Probe(_boxes_agents.BoxesRandomAgent):
            def select_move(self, board: object, seed: int) -> tuple[int, int]:
                if board.move_count() == 3:  # type: ignore[attr-defined]
                    seen.append(tuple(np.asarray(board.to_numpy()).flatten().tolist()))  # type: ignore[attr-defined]
                return super().select_move(board, seed)

        probe = "boxes-test-opening-probe"
        result = play_match(probe, probe, rows=3, num_games=4, seed=5, num_workers=1,
                            opening_moves=3)
        # Three random edges never complete a box, so each game's first searched position is
        # its opening: both games of a pair see the same one and the two pairs differ.
        assert len(seen) == 4 and seen[0] == seen[1] and seen[2] == seen[3]
        assert seen[0] != seen[2]
        m = result.margins
        assert result.pair_margins == [m[0] + m[1], m[2] + m[3]]
        first, second = result.colour_win_rates()
        assert first == sum((x > 0) + 0.5 * (x == 0) for x in m[0::2]) / 2
        assert second == sum((x > 0) + 0.5 * (x == 0) for x in m[1::2]) / 2
        lo, hi = result.pair_wilson()
        assert 0.0 <= lo <= result.pair_score <= hi <= 1.0
        summary = result.summary()
        assert summary["a_pair_score"] == result.pair_score
        assert {"a_first_win_rate", "a_second_win_rate", "a_pair_score_ci95"} <= summary.keys()
