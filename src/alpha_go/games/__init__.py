"""Game implementations. Importing this package registers every game."""
# Import games to trigger registration
from alpha_go.games import go_adapter as _go_adapter  # noqa: F401
from alpha_go.games.base import Game, get_game, list_games, register_game

__all__ = ["Game", "get_game", "list_games", "register_game"]
