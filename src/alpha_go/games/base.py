"""Game protocol and registry.

A `Game` holds the few static facts the shared play loop and self-play CLI need to
drive a game without knowing its rules: how to make a board, how long the dense action
vector is, how an MCTS action key maps into it, and whose turn it is. Boards stay
duck-typed: `alpha_go.gameplay.play_game` calls `to_play()`, `is_game_over()`,
`is_legal(row, col)`, `play(row, col)`, `score()` and `to_numpy()` on whatever
`new_board` returns, so every game's board exposes the same surface as
`alpha_go_cpp.GoBoard`.
"""
from __future__ import annotations

from typing import Any, Protocol


class Game(Protocol):
    """Static facts about a game plus its board factory."""

    name: str

    def new_board(self, board_size: int, komi: float, board_cols: int | None = None) -> Any:
        """Create an empty board. Games without komi or rectangular boards ignore those."""
        ...

    def num_actions(self, board: Any) -> int:
        """Length of the dense action vector (policy logits, MCTS statistics)."""
        ...

    def action_index(self, board: Any, action: int) -> int:
        """Map a sparse MCTS action key to its index in the dense action vector."""
        ...

    def player(self, board: Any) -> int:
        """Index of the side to move: 0 moves first in a new game, 1 is the other side."""
        ...


_GAME_REGISTRY: dict[str, Game] = {}


def register_game(cls: type[Game]) -> type[Game]:
    """Class decorator: instantiate the game once and register it under `cls.name`."""
    _GAME_REGISTRY[cls.name] = cls()
    return cls


def get_game(name: str) -> Game:
    """Look up a registered game by name."""
    if name not in _GAME_REGISTRY:
        available = ", ".join(_GAME_REGISTRY.keys())
        raise ValueError(f"Unknown game: {name}. Available: {available}")
    return _GAME_REGISTRY[name]


def list_games() -> list[str]:
    """Names of all registered games."""
    return list(_GAME_REGISTRY.keys())
