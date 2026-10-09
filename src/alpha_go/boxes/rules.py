"""Dots and Boxes rules: pure-Python reference implementation.

A board has R x C boxes on (R+1) x (C+1) dots. Actions are edge indices: horizontal
edge h(r, c) for r in [0, R], c in [0, C) has index r*C + c; vertical edge v(r, c) for
r in [0, R), c in [0, C] has index (R+1)*C + r*(C+1) + c. Every edge also has lattice
coordinates on the (2R+1) x (2C+1) grid that interleaves dots (even, even), horizontal
edges (even, odd), vertical edges (odd, even) and boxes (odd, odd); `to_numpy()`
returns that grid with drawn edges as 1 and captured boxes as their owner's code.

Completing the fourth side of a box captures it for the mover, who then moves again
(one edge can complete two boxes). The game ends when every edge is drawn. `score()`
is boxes(player 1) - boxes(player 2), mirroring `GoBoard.score()`; `margin()` is the
same difference for the side to move. Player codes mirror Go's BLACK/WHITE so the
shared play loop can key agents by `to_play()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.typing import NDArray

PLAYER_1 = 1  # moves first
PLAYER_2 = 2


@dataclass(frozen=True)
class BoxesGeometry:
    """Index tables for an R x C board, shared by every board of that size."""

    rows: int
    cols: int
    num_edges: int
    edge_rc: tuple[tuple[int, int], ...]  # edge -> lattice (row, col)
    edge_boxes: tuple[tuple[int, ...], ...]  # edge -> adjacent boxes (one or two)
    box_edges: tuple[tuple[int, int, int, int], ...]  # box -> (top, bottom, left, right)
    lattice_edge: NDArray[np.int32]  # (2R+1, 2C+1) -> edge index, -1 off-edge

    @property
    def num_boxes(self) -> int:
        return self.rows * self.cols

    @property
    def lattice_shape(self) -> tuple[int, int]:
        return (2 * self.rows + 1, 2 * self.cols + 1)


@cache
def geometry(rows: int, cols: int) -> BoxesGeometry:
    """Build (once per size) the edge/box/lattice index tables."""

    def h(r: int, c: int) -> int:
        return r * cols + c

    def v(r: int, c: int) -> int:
        return (rows + 1) * cols + r * (cols + 1) + c

    num_edges = (rows + 1) * cols + rows * (cols + 1)
    edge_rc = [(0, 0)] * num_edges
    for r in range(rows + 1):
        for c in range(cols):
            edge_rc[h(r, c)] = (2 * r, 2 * c + 1)
    for r in range(rows):
        for c in range(cols + 1):
            edge_rc[v(r, c)] = (2 * r + 1, 2 * c)
    box_edges = [
        (h(r, c), h(r + 1, c), v(r, c), v(r, c + 1)) for r in range(rows) for c in range(cols)
    ]
    edge_boxes: list[list[int]] = [[] for _ in range(num_edges)]
    for b, sides in enumerate(box_edges):
        for e in sides:
            edge_boxes[e].append(b)
    lattice_edge = np.full((2 * rows + 1, 2 * cols + 1), -1, dtype=np.int32)
    for e, (r, c) in enumerate(edge_rc):
        lattice_edge[r, c] = e
    lattice_edge.setflags(write=False)
    return BoxesGeometry(
        rows, cols, num_edges, tuple(edge_rc), tuple(tuple(x) for x in edge_boxes),
        tuple(box_edges), lattice_edge,
    )


class BoxesBoard:
    """Mutable Boxes position. Method names follow `alpha_go_cpp.GoBoard` where they carry over."""

    __slots__ = ("geo", "edges", "sides", "owner", "boxes", "_to_play", "_move_count")

    def __init__(self, rows: int, cols: int | None = None) -> None:
        self.geo = geometry(rows, rows if cols is None else cols)
        self.edges = 0  # bitmask of drawn edges
        self.sides = [0] * self.geo.num_boxes  # drawn sides per box
        self.owner = [0] * self.geo.num_boxes  # 0 unowned, else PLAYER_1 / PLAYER_2
        self.boxes = [0, 0]  # captured boxes per player index
        self._to_play = PLAYER_1
        self._move_count = 0

    def copy(self) -> BoxesBoard:
        new = BoxesBoard.__new__(BoxesBoard)
        new.geo = self.geo
        new.edges = self.edges
        new.sides = list(self.sides)
        new.owner = list(self.owner)
        new.boxes = list(self.boxes)
        new._to_play = self._to_play
        new._move_count = self._move_count
        return new

    # --- queries -------------------------------------------------------------
    def rows(self) -> int:
        return self.geo.rows

    def cols(self) -> int:
        return self.geo.cols

    def num_edges(self) -> int:
        return self.geo.num_edges

    def to_play(self) -> int:
        return self._to_play

    def player(self) -> int:
        """Index of the side to move: 0 for PLAYER_1, 1 for PLAYER_2."""
        return self._to_play - 1

    def move_count(self) -> int:
        return self._move_count

    def row_col(self, edge: int) -> tuple[int, int]:
        """Lattice coordinates of an edge."""
        return self.geo.edge_rc[edge]

    def edge_index(self, row: int, col: int) -> int:
        """Edge at lattice (row, col), or -1 if that cell is not an edge."""
        height, width = self.geo.lattice_shape
        if not (0 <= row < height and 0 <= col < width):
            return -1
        return int(self.geo.lattice_edge[row, col])

    def is_legal_edge(self, edge: int) -> bool:
        return 0 <= edge < self.geo.num_edges and not (self.edges >> edge) & 1

    def is_legal(self, row: int, col: int) -> bool:
        return self.is_legal_edge(self.edge_index(row, col))

    def get_legal_moves_flat(self) -> list[int]:
        """Undrawn edge indices."""
        return [e for e in range(self.geo.num_edges) if not (self.edges >> e) & 1]

    def is_game_over(self) -> bool:
        return self.edges == (1 << self.geo.num_edges) - 1

    def score(self) -> float:
        """Boxes of PLAYER_1 minus boxes of PLAYER_2."""
        return float(self.boxes[0] - self.boxes[1])

    def margin(self) -> int:
        """Box difference for the side to move."""
        me = self.player()
        return self.boxes[me] - self.boxes[1 - me]

    def get_winner(self) -> int:
        """PLAYER_1, PLAYER_2, or 0 for a tie."""
        diff = self.boxes[0] - self.boxes[1]
        return PLAYER_1 if diff > 0 else PLAYER_2 if diff < 0 else 0

    # --- moves ---------------------------------------------------------------
    def play_edge(self, edge: int) -> bool:
        """Draw an edge. Returns False if it is already drawn. Captures keep the turn."""
        if not self.is_legal_edge(edge):
            return False
        self.edges |= 1 << edge
        captured = 0
        for b in self.geo.edge_boxes[edge]:
            self.sides[b] += 1
            if self.sides[b] == 4:
                self.owner[b] = self._to_play
                captured += 1
        self.boxes[self.player()] += captured
        self._move_count += 1
        if captured == 0:
            self._to_play = PLAYER_2 if self._to_play == PLAYER_1 else PLAYER_1
        return True

    def play(self, row: int, col: int) -> bool:
        """Draw the edge at lattice (row, col). Returns False if illegal."""
        return self.play_edge(self.edge_index(row, col))

    # --- export --------------------------------------------------------------
    def to_numpy(self) -> NDArray[np.int8]:
        """Lattice grid: drawn edges 1, captured boxes their owner's code, else 0."""
        grid = np.zeros(self.geo.lattice_shape, dtype=np.int8)
        for e, (r, c) in enumerate(self.geo.edge_rc):
            grid[r, c] = (self.edges >> e) & 1
        for b, owner in enumerate(self.owner):
            grid[2 * (b // self.geo.cols) + 1, 2 * (b % self.geo.cols) + 1] = owner
        return grid

    def render(self) -> str:
        """ASCII picture: dots, drawn edges, owners as 1/2."""
        grid = self.to_numpy()
        glyphs = {(0, 0): ".", (0, 1): " ", (1, 0): " ", (1, 1): " "}
        lines = []
        for r in range(grid.shape[0]):
            cells = []
            for c in range(grid.shape[1]):
                kind = (r % 2, c % 2)
                value = int(grid[r, c])
                if kind == (0, 1) and value:
                    cells.append("-")
                elif kind == (1, 0) and value:
                    cells.append("|")
                elif kind == (1, 1) and value:
                    cells.append(str(value))
                else:
                    cells.append(glyphs[kind])
            lines.append("".join(cells))
        return "\n".join(lines)


class BoxesState:
    """`alpha_go.mcts.GameState` adapter: actions are edge indices, players are 0 / 1."""

    __slots__ = ("board",)

    def __init__(self, board: BoxesBoard) -> None:
        self.board = board

    @classmethod
    def new_game(cls, rows: int, cols: int | None = None) -> BoxesState:
        return cls(BoxesBoard(rows, cols))

    def get_legal_actions(self) -> list[int]:
        return self.board.get_legal_moves_flat()

    def apply_action(self, action: int) -> BoxesState:
        board = self.board.copy()
        board.play_edge(action)
        return BoxesState(board)

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def get_reward(self, player: int) -> float:
        mine, theirs = self.board.boxes[player], self.board.boxes[1 - player]
        return 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0

    def current_player(self) -> int:
        return self.board.player()

    def clone(self) -> BoxesState:
        return BoxesState(self.board.copy())


def perft(board: BoxesBoard, depth: int) -> tuple[int, int, int]:
    """Count move sequences of length `depth` (or shorter if the game ends first).

    Returns (sequences, boxes held by PLAYER_1 summed over leaves, same for PLAYER_2).
    Legal moves never depend on whose turn it is, so the sequence count alone is just
    a falling factorial; the box sums pin down the capture and extra-move logic.
    """
    if depth == 0 or board.is_game_over():
        return 1, board.boxes[0], board.boxes[1]
    totals = [0, 0, 0]
    for e in board.get_legal_moves_flat():
        child = board.copy()
        child.play_edge(e)
        for i, n in enumerate(perft(child, depth - 1)):
            totals[i] += n
    return totals[0], totals[1], totals[2]
