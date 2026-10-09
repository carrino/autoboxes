"""Feature planes for the Boxes net on the (2R+1) x (2C+1) lattice.

Planes (K = 11), all float32, from the point of view of the side to move:

| plane | content                                        |
|-------|------------------------------------------------|
| 0     | edge drawn                                     |
| 1     | box captured by the side to move               |
| 2     | box captured by the opponent                   |
| 3-7   | box side count one-hot 0..4 (at box cells)     |
| 8     | is-edge-cell (constant)                        |
| 9     | is-box-cell (constant)                         |
| 10    | score margin for the side to move / (R*C)      |

The "chains" feature set (PLAN.md §4.2) appends 10 planes from the chain / loop analysis of
`alpha_go.boxes.chains`, so the net is handed the structure that decides the midgame instead
of having to count it from raw edges:

| plane | content                                                   |
|-------|-----------------------------------------------------------|
| 11    | box in a chain of 1 box                                   |
| 12    | box in a chain of 2 boxes                                 |
| 13    | box in a chain of 3+ boxes (a long chain)                 |
| 14    | box in a loop                                             |
| 15    | box in an opened component (capturable now)               |
| 16    | undrawn edge that is safe (gives no box its third side)   |
| 17    | number of long chains / 4 (constant)                      |
| 18    | number of loops / 4 (constant)                            |
| 19    | safe undrawn edges / E (constant)                         |
| 20    | parity of the number of long chains (constant)            |

Everything is a function of the lattice grid (`board.to_numpy()`: drawn edges 1, captured
boxes their owner's code) and the side to move, so saved games can be encoded without a
board object. Policy logits are produced per lattice cell and gathered at the edge cells
with `edge_flat_index`; see `alpha_go.boxes.symmetry` for the matching permutations.

Dimension key: K planes, H = 2R+1, W = 2C+1, E edges.
"""
from __future__ import annotations

import functools
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
from numpy.typing import NDArray

from alpha_go.boxes.chains import components, degrees
from alpha_go.boxes.rules import BoxesGeometry, geometry

NUM_PLANES = 11
NUM_CHAIN_PLANES = 10
FEATURE_SETS = {"basic": NUM_PLANES, "chains": NUM_PLANES + NUM_CHAIN_PLANES}
cached_geometry = functools.cache(geometry)


def num_planes(features: str) -> int:
    """Input planes of a feature set ("basic" or "chains")."""
    return FEATURE_SETS[features]


def lattice_rows_cols(grid_shape: tuple[int, ...]) -> tuple[int, int]:
    """Box rows and columns of a lattice grid."""
    return (grid_shape[-2] - 1) // 2, (grid_shape[-1] - 1) // 2


def edge_flat_index(rows: int, cols: int) -> NDArray[np.int64]:
    """Flat lattice index (row * W + col) of every edge, in edge order."""
    geo = geometry(rows, cols)
    width = 2 * cols + 1
    return np.array([r * width + c for r, c in geo.edge_rc], dtype=np.int64)


def chain_planes(mask: int, geo: BoxesGeometry) -> NDArray[np.float32]:
    """Planes 11..20 of the "chains" feature set from an edge mask (see the module docstring)."""
    planes = np.zeros((NUM_CHAIN_PLANES, 2 * geo.rows + 1, 2 * geo.cols + 1), dtype=np.float32)
    deg = degrees(mask, geo)
    comps = components(mask, geo)
    for comp in comps:
        kind = 3 if comp.is_loop else min(comp.size, 3) - 1  # chain of 1, 2, 3+ boxes; loop
        for b in comp.boxes:
            r, c = 2 * (b // geo.cols) + 1, 2 * (b % geo.cols) + 1
            planes[kind, r, c] = 1.0
            planes[4, r, c] = float(comp.opened)
    safe = [e for e in range(geo.num_edges)
            if not mask >> e & 1 and all(deg[b] >= 3 for b in geo.edge_boxes[e])]
    for e in safe:
        planes[5][geo.edge_rc[e]] = 1.0
    long_chains = sum(1 for comp in comps if not comp.is_loop and comp.size >= 3)
    planes[6] = long_chains / 4
    planes[7] = sum(1 for comp in comps if comp.is_loop) / 4
    planes[8] = len(safe) / geo.num_edges
    planes[9] = long_chains % 2
    return planes


def encode_grid(grid: NDArray[np.int8], to_play: int,
                features: str = "basic") -> NDArray[np.float32]:
    """Planes (K, H, W) for one lattice grid with `to_play` in {1, 2}."""
    height, width = grid.shape
    rows, cols = lattice_rows_cols(grid.shape)
    planes = np.zeros((NUM_PLANES, height, width), dtype=np.float32)
    edge_cells = np.zeros((height, width), dtype=bool)
    edge_cells[0::2, 1::2] = True
    edge_cells[1::2, 0::2] = True
    box_cells = np.zeros((height, width), dtype=bool)
    box_cells[1::2, 1::2] = True
    drawn = (grid == 1) & edge_cells
    boxes = grid[1::2, 1::2]
    planes[0] = drawn
    planes[1, 1::2, 1::2] = boxes == to_play
    planes[2, 1::2, 1::2] = boxes == 3 - to_play
    sides = (
        drawn[0:-2:2, 1::2].astype(np.int8) + drawn[2::2, 1::2]
        + drawn[1::2, 0:-2:2] + drawn[1::2, 2::2]
    )
    for n in range(5):
        planes[3 + n, 1::2, 1::2] = sides == n
    planes[8] = edge_cells
    planes[9] = box_cells
    margin = int((boxes == to_play).sum()) - int((boxes == 3 - to_play).sum())
    planes[10] = margin / (rows * cols)
    if features == "basic":
        return planes
    geo = cached_geometry(rows, cols)
    mask = sum(1 << e for e, rc in enumerate(geo.edge_rc) if drawn[rc])
    return np.concatenate([planes, chain_planes(mask, geo)])


def encode(board: Any, features: str = "basic") -> NDArray[np.float32]:
    """Planes for a Python or C++ BoxesBoard."""
    return encode_grid(board.to_numpy(), int(board.to_play()), features)


def encode_batch(boards: list[Any], features: str = "basic") -> NDArray[np.float32]:
    """Planes (B, K, H, W) for same-size boards; C++ boards and search states are encoded
    in C++ (`alpha_go_cpp.encode_planes`, parity-tested against `encode`)."""
    if isinstance(boards[0], (alpha_go_cpp.BoxesBoard, alpha_go_cpp.BoxesSearchState)):
        return np.asarray(alpha_go_cpp.encode_planes(boards, features == "chains"),
                          dtype=np.float32)
    return np.stack([encode(b, features) for b in boards])
