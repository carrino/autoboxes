"""Forced-move collapse for the search (PLAN.md §2.5).

At a position where the side to move can capture, the collapse plays the captures that
never need thought: every opened chain or loop is taken in full except the *last* one,
which is taken down to a 2-chain remainder (4 for a loop). If that remainder admits the
double-dealing move (chain `A-e1-B-e2-X`: keep control by drawing `e2`; loop
`A1-B1-B2-A2`: draw the middle edge), the search sees exactly two macro-actions,
TAKE ALL and KEEP CONTROL, each identified by its first edge; otherwise the remainder is
taken too. With nothing to capture the actions are the plain undrawn edges.

`BoxesSearchState` wraps a board that sits at such a decision point and exposes both the
`alpha_go.mcts.GameState` protocol (actions are the macro-actions' first edges) and the
board surface the evaluators use (`to_numpy`, `to_play`, `get_legal_moves_flat`), so
priors come from the policy head's logit for the first edge and visit counts are credited
to it in training targets. The real game still plays one edge per turn: an agent plays
the auto-captured prefix edge by edge (no search needed) and searches only at decision
points. Works on the Python and the C++ board.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha_go.boxes.chains import OPEN, Component, components, degrees
from alpha_go.boxes.rules import BoxesGeometry, geometry


def edge_mask(board: Any) -> int:
    return int(board.edges() if callable(board.edges) else board.edges)


def box_geometry(board: Any) -> BoxesGeometry:
    return geometry(int(board.rows()), int(board.cols()))


def shared_edge(geo: BoxesGeometry, a: int, b: int) -> int:
    return next(e for e in geo.box_edges[a] if b in geo.edge_boxes[e])


def open_end_first(comp: Component, deg: list[int]) -> list[int]:
    """Boxes of an opened component ordered so capturing proceeds from an open end."""
    return comp.boxes if deg[comp.boxes[0]] == 1 else list(reversed(comp.boxes))


def take_sequence(mask: int, geo: BoxesGeometry, boxes: list[int]) -> list[int]:
    """Edges that capture `boxes` in order from the open end (one edge may take two)."""
    edges: list[int] = []
    for box in boxes:
        undrawn = [e for e in geo.box_edges[box] if not (mask >> e) & 1]
        if undrawn:  # otherwise the previous edge completed this box as well
            edges.append(undrawn[0])
            mask |= 1 << undrawn[0]
    return edges


def control_edge(mask: int, geo: BoxesGeometry, comp: Component, deg: list[int]) -> int | None:
    """Double-dealing edge of a remainder, or None if the remainder must simply be taken."""
    boxes = open_end_first(comp, deg)
    both_open = comp.ends == (OPEN, OPEN)
    if both_open and comp.size == 4:
        return shared_edge(geo, boxes[1], boxes[2])
    if not both_open and comp.size == 2:
        far = [e for e in geo.box_edges[boxes[1]] if not (mask >> e) & 1]
        link = shared_edge(geo, boxes[0], boxes[1])
        return next(e for e in far if e != link)
    return None


def remainder_size(comp: Component) -> int:
    return 4 if comp.ends == (OPEN, OPEN) else 2


@dataclass
class Decision:
    """Pending take-all / keep-control choice on the last opened component."""

    take: list[int]  # edges that capture the remainder
    control: int  # the double-dealing edge

    @property
    def actions(self) -> list[int]:
        return [self.take[0], self.control]


@dataclass
class Collapsed:
    board: Any
    prefix: list[int] = field(default_factory=list)  # auto-played edges (same mover)
    decision: Decision | None = None


def collapse(board: Any) -> Collapsed:
    """Auto-capture for the side to move until nothing is forced; see module docstring."""
    geo = box_geometry(board)
    board = board.copy()
    prefix: list[int] = []
    while not board.is_game_over():
        mask = edge_mask(board)
        deg = degrees(mask, geo)
        opened = [c for c in components(mask, geo) if c.opened]
        if not opened:
            return Collapsed(board, prefix)
        # Keep for last the component that offers control at its remainder (cheapest
        # sacrifice first: a chain gives 2, a loop 4); everything else is taken in full.
        with_control = [c for c in opened if c.size >= remainder_size(c)
                        and control_edge(mask, geo, c, deg) is not None
                        or c.size > remainder_size(c)]
        with_control.sort(key=remainder_size)
        last = with_control[0] if with_control else None
        for comp in opened:
            if comp is last:
                continue
            for e in take_sequence(mask, geo, open_end_first(comp, deg)):
                board.play_edge(e)
                prefix.append(e)
            break  # positions changed: recompute components
        else:
            assert last is not None
            boxes = open_end_first(last, deg)
            surplus = last.size - remainder_size(last)
            if surplus > 0:
                for e in take_sequence(mask, geo, boxes[:surplus]):
                    board.play_edge(e)
                    prefix.append(e)
                continue
            control = control_edge(mask, geo, last, deg)
            if control is None:
                for e in take_sequence(mask, geo, boxes):
                    board.play_edge(e)
                    prefix.append(e)
                continue
            return Collapsed(board, prefix, Decision(take_sequence(mask, geo, boxes), control))
    return Collapsed(board, prefix)


class BoxesSearchState:
    """A collapsed position: MCTS protocol + evaluator surface (see module docstring)."""

    __slots__ = ("board", "decision")

    def __init__(self, board: Any, decision: Decision | None) -> None:
        self.board = board
        self.decision = decision

    @classmethod
    def from_board(cls, board: Any) -> BoxesSearchState:
        c = collapse(board)
        return cls(c.board, c.decision)

    # --- MCTS protocol ---
    def get_legal_actions(self) -> list[int]:
        if self.decision is not None:
            return self.decision.actions
        return list(self.board.get_legal_moves_flat())

    def apply_action(self, action: int) -> BoxesSearchState:
        board = self.board.copy()
        edges = [action]
        if self.decision is not None and action == self.decision.take[0]:
            edges = self.decision.take
        for e in edges:
            assert board.play_edge(e)
        return BoxesSearchState.from_board(board)

    def is_terminal(self) -> bool:
        return bool(self.board.is_game_over())

    def get_reward(self, player: int) -> float:
        return float(self.board.outcome(player)) if hasattr(self.board, "outcome") else (
            1.0 if self.board.boxes[player] > self.board.boxes[1 - player] else
            0.5 if self.board.boxes[player] == self.board.boxes[1 - player] else 0.0)

    def current_player(self) -> int:
        return int(self.board.player())

    def clone(self) -> BoxesSearchState:
        return BoxesSearchState(self.board.copy(), self.decision)

    # --- evaluator surface (same names as the boards) ---
    def get_legal_moves_flat(self) -> list[int]:
        return self.get_legal_actions()

    def to_numpy(self) -> Any:
        return self.board.to_numpy()

    def to_play(self) -> int:
        return int(self.board.to_play())

    def player(self) -> int:
        return int(self.board.player())

    def rows(self) -> int:
        return int(self.board.rows())

    def cols(self) -> int:
        return int(self.board.cols())

    def is_game_over(self) -> bool:
        return self.is_terminal()

    def margin(self) -> int:
        return int(self.board.margin())
