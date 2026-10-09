"""Board symmetries: the dihedral group acting on the Boxes lattice.

A transform `k` in 0..7 is `rot90^(k % 4)` applied after an optional flip of the last
axis (`k >= 4`). It acts on any array whose last two axes are the lattice (a board grid,
a stack of feature planes, a batch of planes). Because every edge is a lattice cell,
applying the same transform to the grid of edge indices tells where each edge lands,
which yields the action permutation without a hand-written table. Square boards have all
8 transforms; rectangular boards keep the 4 that preserve the shape.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from alpha_go.boxes.rules import BoxesGeometry


def transforms(rows: int, cols: int) -> list[int]:
    """Transform ids valid for an R x C board: 8 if square, else the 4 shape-preserving ones."""
    return list(range(8)) if rows == cols else [0, 2, 4, 6]


def apply(array: NDArray[Any], k: int) -> NDArray[Any]:
    """Apply transform `k` to the last two axes of `array` (returns a view or copy)."""
    out = np.flip(array, axis=-1) if k >= 4 else array
    return np.rot90(out, k % 4, axes=(-2, -1))


def inverse(k: int) -> int:
    """Transform id that undoes `k` (flip-then-rotate transforms are involutions)."""
    return k if k >= 4 else (4 - k) % 4


def edge_permutation(geo: BoxesGeometry, k: int) -> NDArray[np.int64]:
    """`perm[e]` is the index of the edge that edge `e` becomes under transform `k`.

    A per-edge vector `p` (policy, visit counts, legal mask) transforms as
    `p_sym = np.empty_like(p); p_sym[perm] = p`.
    """
    moved = apply(geo.lattice_edge, k)
    perm = np.empty(geo.num_edges, dtype=np.int64)
    for r, c in zip(*np.nonzero(moved >= 0)):
        perm[moved[r, c]] = geo.lattice_edge[r, c]
    return perm
