"""Exact value targets for stored games: the solver's final margin from every late position.

A self-play game's value target is its outcome, which both sides can misplay between the
position and the end. Replaying the game against the exact solver from the first position
with at most `max_undrawn` undrawn edges gives every later position its exact final margin
under optimal play instead. One solver per game carries its table down the line, so the
first solve does nearly all the work. `BoxesDataset` prefers the `exact_margin_p1` array
(player-1 final margin, `EXACT_UNKNOWN` where the budget ran out or the position is above
the reach) over the game's result wherever it is present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
from numpy.typing import NDArray

EXACT_UNKNOWN = int(np.iinfo(np.int16).min)


def exact_margins_p1(game: dict[str, Any], rows: int, cols: int, max_undrawn: int,
                     node_budget: int, solver: Any) -> NDArray[np.int16]:
    """Player-1 exact final margin for each stored position of `game` within the solver's
    reach, `EXACT_UNKNOWN` elsewhere. `game` is a loaded NPZ (`moves`, `to_play`, and
    `start_moves` for a branched game)."""
    board = alpha_go_cpp.BoxesBoard(rows, cols)
    for row, col in np.asarray(game.get("start_moves", np.zeros((0, 2), np.int16))):
        board.play(int(row), int(col))
    moves = np.asarray(game["moves"])
    out = np.full(len(moves), EXACT_UNKNOWN, dtype=np.int16)
    for k, (row, col) in enumerate(moves):
        assert int(board.player()) == int(game["to_play"][k]), "stored game does not replay"
        if board.num_edges() - board.move_count() <= max_undrawn:
            value = solver.value_within(board.edges(), node_budget)
            if value is not None:
                out[k] = round(board.score()) + (value if board.player() == 0 else -value)
        board.play(int(row), int(col))
    return out


def relabel_file(path: Path, rows: int, cols: int, max_undrawn: int, node_budget: int,
                 solver: Any) -> NDArray[np.int16]:
    """Add (or replace) `exact_margin_p1` in the game NPZ at `path`, in place; returns it."""
    game = dict(np.load(path))
    exact = exact_margins_p1(game, rows, cols, max_undrawn, node_budget, solver)
    game["exact_margin_p1"] = exact
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **game)
    tmp.replace(path)
    return exact
