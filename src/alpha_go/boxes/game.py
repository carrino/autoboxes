"""Boxes as a `Game`: adapter over the C++ `alpha_go_cpp.BoxesBoard`."""
from __future__ import annotations

from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]

from alpha_go.games.base import register_game


@register_game
class BoxesGame:
    """Boxes facts for the shared play loop: actions are edge indices, no pass."""

    name = "boxes"

    def new_board(self, board_size: int, komi: float, board_cols: int | None = None) -> Any:
        return alpha_go_cpp.BoxesBoard(board_size, board_cols or 0)

    def num_actions(self, board: Any) -> int:
        return int(board.num_edges())

    def action_index(self, board: Any, action: int) -> int:
        return action

    def player(self, board: Any) -> int:
        return int(board.player())

    def terminal_label(self) -> str:
        return "all_edges"

    def default_max_moves(self, board_size: int, board_cols: int | None = None) -> int:
        cols = board_cols or board_size
        return (board_size + 1) * cols + board_size * (cols + 1)
