"""Shared play loop, self-play CLI and arena driving Boxes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from alpha_go import self_play
from alpha_go.agents import get_agent
from alpha_go.boxes import agents as _boxes_agents  # noqa: F401
from alpha_go.boxes.arena import play_match
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
