"""Exhaustive minimax oracle for tiny boards.

Because every undrawn edge is legal for whoever moves, the remaining-game value depends
only on the drawn-edge mask: `Oracle.value(mask)` is the best box margin the side to
move can still gain over the rest of the game under optimal play (negamax over masks,
memoised). The final margin from a position is `board.margin() + value`. Feasible up
to 2x3 (2^17 masks); used to validate the search and the fast solver.
"""
from __future__ import annotations

from functools import lru_cache

from alpha_go.boxes.rules import BoxesBoard, geometry


class Oracle:
    """Negamax over edge masks for one board size."""

    def __init__(self, rows: int, cols: int | None = None) -> None:
        self.geo = geometry(rows, rows if cols is None else cols)
        self.box_masks = [sum(1 << e for e in sides) for sides in self.geo.box_edges]
        self.full = (1 << self.geo.num_edges) - 1
        self.value = lru_cache(maxsize=None)(self._value)

    def captures(self, mask: int, edge: int) -> int:
        """Boxes completed by drawing `edge` on top of `mask`."""
        nxt = mask | (1 << edge)
        masks = self.box_masks
        return sum(1 for b in self.geo.edge_boxes[edge] if nxt & masks[b] == masks[b])

    def child_value(self, mask: int, edge: int) -> int:
        """Remaining margin for the side to move if it draws `edge` and then plays optimally."""
        gained = self.captures(mask, edge)
        nxt = mask | (1 << edge)
        return gained + self.value(nxt) if gained else -self.value(nxt)

    def _value(self, mask: int) -> int:
        if mask == self.full:
            return 0
        legal = [e for e in range(self.geo.num_edges) if not (mask >> e) & 1]
        return max(self.child_value(mask, e) for e in legal)

    def best_edges(self, board: BoxesBoard) -> list[int]:
        """Every edge whose line reaches the optimal remaining margin."""
        mask = board.edges
        best = self.value(mask)
        return [e for e in board.get_legal_moves_flat() if self.child_value(mask, e) == best]

    def final_margin(self, board: BoxesBoard) -> int:
        """Final box margin for the side to move under optimal play from here."""
        return board.margin() + self.value(board.edges)
