"""Built-in Boxes baselines: random, greedy, and depth-limited alpha-beta.

All agents play on the C++ `alpha_go_cpp.BoxesBoard` the shared play loop passes in and
return the chosen edge as lattice (row, col), like Go agents return board coordinates.
Edge classification: CAPTURE completes a box, LOONY gives a box its third side (hands the
opponent a capture), SAFE does neither.
"""
from __future__ import annotations

import random
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]

import alpha_go.boxes.nn_agent  # noqa: F401  (registers boxes-mcts-* alongside the baselines)
from alpha_go.agents.base import Agent, register_agent
from alpha_go.boxes.rules import BoxesGeometry, geometry

CAPTURE, SAFE, LOONY = 0, 1, 2


def lattice(board: Any, edge: int) -> tuple[int, int]:
    """Lattice (row, col) of an edge, as the shared play loop expects moves."""
    row, col = board.row_col(edge)
    return int(row), int(col)


def classify(board: Any, geo: BoxesGeometry, edge: int) -> int:
    """Edge class for the side to move: CAPTURE, SAFE or LOONY."""
    most = max(int(board.sides(b)) for b in geo.edge_boxes[edge]) + 1
    return CAPTURE if most == 4 else LOONY if most == 3 else SAFE


def captures(board: Any, geo: BoxesGeometry, edge: int) -> int:
    """Boxes completed by drawing `edge`."""
    return sum(1 for b in geo.edge_boxes[edge] if board.sides(b) == 3)


def greedy_edge(board: Any, geo: BoxesGeometry, rng: random.Random) -> int:
    """Take a capture (two-box captures first); else a random safe edge; else random."""
    legal: list[int] = board.get_legal_moves_flat()
    rng.shuffle(legal)
    taking = [e for e in legal if classify(board, geo, e) == CAPTURE]
    if taking:
        return max(taking, key=lambda e: captures(board, geo, e))
    safe = [e for e in legal if classify(board, geo, e) == SAFE]
    return safe[0] if safe else legal[0]


def greedy_haul(board: Any, geo: BoxesGeometry) -> int:
    """Boxes the side to move collects by capturing greedily until nothing is capturable."""
    board = board.copy()
    mover, before = board.player(), int(board.margin())
    while not board.is_game_over():
        taking = [e for e in board.get_legal_moves_flat() if classify(board, geo, e) == CAPTURE]
        if not taking:
            break
        board.play_edge(max(taking, key=lambda e: captures(board, geo, e)))
    margin = int(board.margin())
    return margin - before if board.player() == mover else -margin - before


@register_agent("boxes-random")
class BoxesRandomAgent(Agent):
    """Uniformly random legal edge."""

    def select_move(self, board: Any, seed: int) -> tuple[int, int]:
        return lattice(board, random.Random(seed).choice(board.get_legal_moves_flat()))


@register_agent("boxes-greedy")
class BoxesGreedyAgent(Agent):
    """Takes captures, otherwise avoids giving a third side, otherwise random."""

    def select_move(self, board: Any, seed: int) -> tuple[int, int]:
        geo = geometry(board.rows(), board.cols())
        return lattice(board, greedy_edge(board, geo, random.Random(seed)))


class AlphaBeta:
    """Depth-limited negamax on the remaining box margin with a transposition table.

    The value of a position is the margin the side to move can still gain. Capturing
    moves keep the turn and do not consume depth (they shrink the board, so the search
    terminates). At the depth limit the position is scored by the greedy haul available
    to the side to move. The table is keyed on the edge mask alone, which identifies the
    remaining game regardless of the score so far; `tt_entries` bounds its size. This is
    the Python reference for `alpha_go_cpp.BoxesAlphaBeta`, which the registered agents use.
    """

    EXACT, LOWER, UPPER = 0, 1, 2
    INF = 10**6

    def __init__(self, depth: int, tt_entries: int = 1_000_000) -> None:
        self.depth = depth
        self.tt_entries = tt_entries
        self.tt: dict[int, tuple[int, int, int]] = {}
        self.nodes = 0
        self.geo: BoxesGeometry | None = None

    def footprint_bytes(self) -> int:
        """Approximate bytes one full table costs (int key + 3-int tuple per entry)."""
        return self.tt_entries * 160

    def ordered_moves(self, board: Any, geo: BoxesGeometry, rng: random.Random) -> list[int]:
        legal: list[int] = board.get_legal_moves_flat()
        rng.shuffle(legal)
        return sorted(legal, key=lambda e: classify(board, geo, e))

    def search(self, board: Any, depth: int, alpha: int, beta: int, rng: random.Random) -> int:
        geo = self.geo
        assert geo is not None
        self.nodes += 1
        if board.is_game_over():
            return 0
        if depth == 0:
            return greedy_haul(board, geo)
        key = int(board.edges())
        hit = self.tt.get(key)
        if hit is not None and hit[0] == depth:  # exact depth only: the value is then
            # the depth-limited minimax value, independent of move order (and of the C++ port)
            stored_depth, value, flag = hit
            if flag == self.EXACT:
                return value
            if flag == self.LOWER:
                alpha = max(alpha, value)
            if flag == self.UPPER:
                beta = min(beta, value)
            if alpha >= beta:
                return value
        alpha_orig = alpha
        best = -self.INF
        for edge in self.ordered_moves(board, geo, rng):
            child = board.copy()
            child.play_edge(edge)
            if child.player() == board.player():
                gained = int(child.margin()) - int(board.margin())
                value = gained + self.search(child, depth, alpha - gained, beta - gained, rng)
            else:
                value = -self.search(child, depth - 1, -beta, -alpha, rng)
            best = max(best, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        flag = self.UPPER if best <= alpha_orig else self.LOWER if best >= beta else self.EXACT
        if len(self.tt) < self.tt_entries:
            self.tt[key] = (depth, best, flag)
        return best

    def best_edge(self, board: Any, rng: random.Random) -> int:
        """Edge with the highest searched value; ties broken by `rng` through move order."""
        geo = geometry(board.rows(), board.cols())
        if self.geo is not geo:
            self.geo, self.tt = geo, {}
        best_edge, best_value = -1, -self.INF
        for edge in self.ordered_moves(board, geo, rng):
            child = board.copy()
            child.play_edge(edge)
            if child.player() == board.player():
                gained = int(child.margin()) - int(board.margin())
                value = gained + self.search(child, self.depth, -self.INF, self.INF, rng)
            else:
                value = -self.search(child, self.depth - 1, -self.INF, self.INF, rng)
            if value > best_value:
                best_edge, best_value = edge, value
        return best_edge


class BoxesAlphaBetaAgent(Agent):
    """Alpha-beta baseline on the C++ searcher (`alpha_go_cpp.BoxesAlphaBeta`, the port of
    `AlphaBeta` above, which stays as its reference); subclasses fix the depth."""

    depth = 4

    def __init__(self) -> None:
        self.searcher: Any = None  # built on first use (needs the board size)

    def select_move(self, board: Any, seed: int) -> tuple[int, int]:
        if self.searcher is None:
            self.searcher = alpha_go_cpp.BoxesAlphaBeta(int(board.rows()), int(board.cols()))
        mask = int(board.edges() if callable(board.edges) else board.edges)
        return lattice(board, int(self.searcher.best_edge(mask, self.depth, seed)))


for _depth in (2, 4, 6):
    _cls = type(f"BoxesAlphaBetaD{_depth}", (BoxesAlphaBetaAgent,), {"depth": _depth})
    register_agent(f"boxes-ab-d{_depth}")(_cls)


class BoxesSolverBackedAgent(Agent):
    """Fast baseline with a perfect endgame: greedy (depth 0) or alpha-beta until `undrawn`
    edges remain, then the exact C++ solver plays the rest. Registered as
    boxes-greedy-s<N> and boxes-ab-d<depth>-s<N>."""

    undrawn = 24
    depth = 0

    def __init__(self) -> None:
        self.searcher = AlphaBeta(self.depth) if self.depth else None
        self.solver: Any = None

    def select_move(self, board: Any, seed: int) -> tuple[int, int]:
        if int(board.num_edges()) - int(board.move_count()) <= self.undrawn:
            if self.solver is None:
                self.solver = alpha_go_cpp.BoxesSolver(int(board.rows()), int(board.cols()),
                                                       1 << 20)
            mask = int(board.edges() if callable(board.edges) else board.edges)
            return lattice(board, int(self.solver.best_edge_mask(mask)))
        rng = random.Random(seed)
        if self.searcher is None:
            return lattice(board, greedy_edge(board, geometry(board.rows(), board.cols()), rng))
        return lattice(board, self.searcher.best_edge(board, rng))


for _depth in (0, 2):
    _name = f"boxes-{'greedy' if _depth == 0 else f'ab-d{_depth}'}-s24"
    _cls = type(f"BoxesSolverBacked{_depth}", (BoxesSolverBackedAgent,),
                {"depth": _depth, "undrawn": 24})
    register_agent(_name)(_cls)
