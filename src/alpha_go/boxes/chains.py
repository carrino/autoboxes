"""Chain and loop analysis on the strings-and-coins dual.

Boxes are coins, undrawn edges are strings; a border edge is a string to the ground
(outside). A box's degree is its number of undrawn sides. Boxes of degree 1 or 2 form
*components*: paths (chains) whose ends are the ground, a junction (a box of degree >= 3)
or an *open* end (a box of degree 1, capturable right now), and cycles (loops) of
degree-2 boxes. Everything here works from the edge mask and the geometry tables, so it
serves the Python and C++ boards alike and is reused by the forced-move collapse and by
the Phase 2 features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha_go.boxes.rules import BoxesGeometry, geometry

GROUND = -1


@dataclass
class Component:
    boxes: list[int]  # in path order (any order for a loop)
    is_loop: bool
    ends: tuple[int, int]  # per end: GROUND, a junction box index, or OPEN (deg-1 end)

    @property
    def opened(self) -> bool:
        return OPEN in self.ends

    @property
    def size(self) -> int:
        return len(self.boxes)


OPEN = -2


def degrees(mask: int, geo: BoxesGeometry) -> list[int]:
    """Undrawn sides per box."""
    return [sum(1 for e in sides if not (mask >> e) & 1) for sides in geo.box_edges]


def undrawn_neighbour(mask: int, geo: BoxesGeometry, box: int, edge: int) -> int:
    """The box across `edge` from `box`, or GROUND for a border edge."""
    pair = geo.edge_boxes[edge]
    return pair[1] if pair[0] == box and len(pair) == 2 else (pair[0] if pair[0] != box else GROUND)


def components(mask: int, geo: BoxesGeometry) -> list[Component]:
    """Chains and loops of the current position (boxes of degree 1 or 2)."""
    deg = degrees(mask, geo)
    seen = [False] * geo.num_boxes
    out: list[Component] = []

    def walk(start: int, via: int) -> tuple[list[int], int]:
        """Follow degree-2 boxes from `start` across `via`; return (boxes, end marker)."""
        path: list[int] = []
        box, edge = start, via
        while True:
            nxt = undrawn_neighbour(mask, geo, box, edge)
            if nxt == GROUND:
                return path, GROUND
            if deg[nxt] >= 3:
                return path, nxt
            if seen[nxt]:
                return path, OPEN if deg[nxt] == 1 else nxt
            seen[nxt] = True
            path.append(nxt)
            others = [e for e in geo.box_edges[nxt] if not (mask >> e) & 1 and e != edge]
            if not others:  # degree-1 box: an open end
                return path, OPEN
            box, edge = nxt, others[0]

    for b in range(geo.num_boxes):
        if seen[b] or deg[b] not in (1, 2):
            continue
        seen[b] = True
        edges = [e for e in geo.box_edges[b] if not (mask >> e) & 1]
        left: list[int] = []
        left_end = OPEN
        if deg[b] == 2:
            left, left_end = walk(b, edges[0])
            if left_end == b:  # walked all the way around: a loop
                out.append(Component([b] + left, True, (b, b)))
                continue
        right, right_end = walk(b, edges[-1])
        out.append(Component(list(reversed(left)) + [b] + right, False, (left_end, right_end)))
    return out


def analyse(board: Any) -> list[Component]:
    """Components of a Python or C++ BoxesBoard."""
    return components(int(board.edges()) if callable(board.edges) else int(board.edges),
                      geometry(board.rows(), board.cols()))
