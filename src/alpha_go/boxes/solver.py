"""Exact endgame solver, Python reference (PLAN.md §4.1).

`Solver.value(mask)` is the remaining box margin for the side to move under optimal play,
a function of the edge mask alone (the margin so far is additive and never changes the
optimal line). It is a plain negamax made tractable by four exact reductions, each of
which is a theorem of strings-and-coins play rather than a heuristic:

* captures are forced through `forced.collapse_mask`: every opened chain or loop is
  taken in full except a remainder that offers the double-dealing choice, which becomes
  the only two moves (take all / keep control);
* the transposition table is keyed on the symmetry-canonical mask (minimum over the
  lattice transforms);
* in a quiet position the undrawn edges of one *independent* chain (both ends at the
  ground) or loop are equivalent, so one representative is searched; the representative
  of a 2-chain is its middle edge (the hard-hearted handout weakly dominates);
* a quiet position whose boxes are all in independent chains or loops (no junction) is a
  simple loony endgame, valued exactly by `loony_value` over the multiset of sizes.

This file is the correctness reference for the C++ solver; it is deliberately a plain
negamax with an exact table and no alpha-beta.
"""
from __future__ import annotations

from functools import cache
from typing import Any

from alpha_go.boxes.chains import GROUND, Component, components, degrees
from alpha_go.boxes.forced import collapse_mask, edge_mask
from alpha_go.boxes.rules import BoxesGeometry, geometry
from alpha_go.boxes.symmetry import edge_permutation, transforms


@cache
def loony_value(chains: tuple[int, ...], loops: tuple[int, ...]) -> int:
    """Exact remaining margin for the side to move in a simple loony endgame.

    The mover must open a component; the opponent (in control) either takes it all and
    moves next, or takes all but 2 (4 for a loop) and double-deals so the mover takes those
    and must open again. Chains of 1 and 2 cannot be double-dealt (2-chains are handed out
    hard-heartedly), so the opponent takes them and loses control.
    """
    best = None
    for i, size in enumerate(chains):
        if i and chains[i - 1] == size:
            continue
        rest = loony_value(chains[:i] + chains[i + 1:], loops)
        take_all = -size - rest
        value = take_all if size <= 2 else min(take_all, 4 - size + rest)
        best = value if best is None else max(best, value)
    for i, size in enumerate(loops):
        if i and loops[i - 1] == size:
            continue
        rest = loony_value(chains, loops[:i] + loops[i + 1:])
        value = min(-size - rest, 8 - size + rest)
        best = value if best is None else max(best, value)
    return 0 if best is None else best


def independent(comp: Component) -> bool:
    return comp.is_loop or comp.ends == (GROUND, GROUND)


def component_edges(mask: int, geo: BoxesGeometry, comp: Component) -> list[int]:
    """Undrawn edges of a component (its links, including those to the ground)."""
    return sorted({e for b in comp.boxes for e in geo.box_edges[b] if not (mask >> e) & 1})


def completed(before: int, after: int, geo: BoxesGeometry) -> int:
    """Boxes completed between two masks."""
    return sum(
        1 for sides in geo.box_edges
        if all((after >> e) & 1 for e in sides) and not all((before >> e) & 1 for e in sides)
    )


class Solver:
    """Exact remaining margin; see module docstring. `use_leaf` / `use_equivalence` /
    `use_symmetry` exist so tests can compare the reductions against the bare search."""

    def __init__(self, rows: int, cols: int | None = None, use_leaf: bool = True,
                 use_equivalence: bool = True, use_symmetry: bool = True) -> None:
        self.geo = geometry(rows, rows if cols is None else cols)
        self.full = (1 << self.geo.num_edges) - 1
        self.use_leaf, self.use_equivalence = use_leaf, use_equivalence
        ks = transforms(self.geo.rows, self.geo.cols) if use_symmetry else [0]
        self.perms = [edge_permutation(self.geo, k) for k in ks]
        self.table: dict[int, int] = {}

    def canonical(self, mask: int) -> int:
        bits = [e for e in range(self.geo.num_edges) if (mask >> e) & 1]
        return min(sum(1 << int(perm[e]) for e in bits) for perm in self.perms)

    def value(self, mask: int) -> int:
        """Remaining margin for the side to move, from any position (captures pending or not)."""
        key = self.canonical(mask)
        if key not in self.table:
            self.table[key] = self._value(mask)
        return self.table[key]

    def _value(self, mask: int) -> int:
        geo = self.geo
        quiet, prefix, decision = collapse_mask(mask, geo)
        gained = completed(mask, quiet, geo)
        if quiet == self.full:
            return gained
        if decision is not None:
            after_take = quiet
            for e in decision.take:
                after_take |= 1 << e
            take = completed(quiet, after_take, geo) + self.value(after_take)
            control = -self.value(quiet | (1 << decision.control))
            return gained + max(take, control)
        comps = components(quiet, geo)
        deg = degrees(quiet, geo)
        if self.use_leaf and max(deg) <= 2 and all(independent(c) for c in comps):
            chains = tuple(sorted(c.size for c in comps if not c.is_loop))
            loops = tuple(sorted(c.size for c in comps if c.is_loop))
            return gained + loony_value(chains, loops)
        return gained + max(-self.value(quiet | (1 << e)) for e in self.moves(quiet, comps))

    def moves(self, quiet: int, comps: list[Component]) -> list[int]:
        """Undrawn edges of a quiet position, one representative per independent component."""
        geo = self.geo
        legal = [e for e in range(geo.num_edges) if not (quiet >> e) & 1]
        if not self.use_equivalence:
            return legal
        drop: set[int] = set()
        for comp in comps:
            if not independent(comp):
                continue
            edges = component_edges(quiet, geo, comp)
            keep = edges[0]
            if not comp.is_loop and comp.size == 2:  # hard-hearted handout: the middle edge
                keep = next(e for e in edges if len(geo.edge_boxes[e]) == 2
                            and set(geo.edge_boxes[e]) == set(comp.boxes))
            drop.update(e for e in edges if e != keep)
        return [e for e in legal if e not in drop]

    # --- board-level helpers -------------------------------------------------
    def remaining(self, board: Any) -> int:
        return self.value(edge_mask(board))

    def final_margin(self, board: Any) -> int:
        """Final margin for the side to move under optimal play from `board`."""
        return int(board.margin()) + self.remaining(board)

    def best_edge(self, board: Any) -> int:
        """An optimal edge for the side to move (forced captures first)."""
        mask = edge_mask(board)
        quiet, prefix, decision = collapse_mask(mask, self.geo)
        if prefix:
            return prefix[0]
        if decision is not None:
            after_take = quiet
            for e in decision.take:
                after_take |= 1 << e
            take = completed(quiet, after_take, self.geo) + self.value(after_take)
            control = -self.value(quiet | (1 << decision.control))
            return decision.take[0] if take >= control else decision.control
        legal = [e for e in range(self.geo.num_edges) if not (quiet >> e) & 1]
        return max(legal, key=lambda e: -self.value(quiet | (1 << e)))
