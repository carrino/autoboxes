"""Tests for the Game registry and the Go adapter."""
from __future__ import annotations

import alpha_go_cpp
import pytest

from alpha_go.games import get_game, list_games


class TestRegistry:
    def test_go_is_registered(self) -> None:
        assert "go" in list_games()
        assert get_game("go").name == "go"

    def test_unknown_game(self) -> None:
        with pytest.raises(ValueError):
            get_game("chess")


class TestGoGame:
    def test_new_board(self) -> None:
        board = get_game("go").new_board(9, 7.5)
        assert board.size() == 9
        assert board.komi() == 7.5

    def test_num_actions_and_pass_index(self) -> None:
        go = get_game("go")
        board = go.new_board(9, 7.5)
        assert go.num_actions(board) == 82
        assert go.action_index(board, alpha_go_cpp.PASS_ACTION) == 81
        assert go.action_index(board, 40) == 40

    def test_player_alternates(self) -> None:
        go = get_game("go")
        board = go.new_board(9, 7.5)
        assert go.player(board) == 0
        board.play(4, 4)
        assert go.player(board) == 1
        board.pass_move()
        assert go.player(board) == 0
