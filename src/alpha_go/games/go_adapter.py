"""Go as a `Game`: a thin adapter over the C++ `alpha_go_cpp.GoBoard`."""
from __future__ import annotations

from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]

from alpha_go.games.base import register_game


@register_game
class GoGame:
    """Go facts for the shared play loop: square board, flat actions plus a trailing pass."""

    name = "go"

    def new_board(self, board_size: int, komi: float, board_cols: int | None = None) -> Any:
        return alpha_go_cpp.GoBoard(board_size, komi)

    def num_actions(self, board: Any) -> int:
        return int(board.size()) ** 2 + 1

    def action_index(self, board: Any, action: int) -> int:
        if action == alpha_go_cpp.PASS_ACTION:
            return int(board.size()) ** 2
        return action

    def player(self, board: Any) -> int:
        return 0 if board.to_play() == alpha_go_cpp.GoBoard.BLACK else 1
