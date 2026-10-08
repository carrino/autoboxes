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

Everything is a function of the lattice grid (`board.to_numpy()`: drawn edges 1, captured
boxes their owner's code) and the side to move, so saved games can be encoded without a
board object. Policy logits are produced per lattice cell and gathered at the edge cells
with `edge_flat_index`; see `alpha_go.boxes.symmetry` for the matching permutations.

Dimension key: K planes, H = 2R+1, W = 2C+1, E edges.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from alpha_go.boxes.rules import geometry

NUM_PLANES = 11


def lattice_rows_cols(grid_shape: tuple[int, ...]) -> tuple[int, int]:
    """Box rows and columns of a lattice grid."""
    return (grid_shape[-2] - 1) // 2, (grid_shape[-1] - 1) // 2


def edge_flat_index(rows: int, cols: int) -> NDArray[np.int64]:
    """Flat lattice index (row * W + col) of every edge, in edge order."""
    geo = geometry(rows, cols)
    width = 2 * cols + 1
    return np.array([r * width + c for r, c in geo.edge_rc], dtype=np.int64)


def encode_grid(grid: NDArray[np.int8], to_play: int) -> NDArray[np.float32]:
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
    return planes


def encode(board: Any) -> NDArray[np.float32]:
    """Planes for a Python or C++ BoxesBoard."""
    return encode_grid(board.to_numpy(), int(board.to_play()))


def encode_batch(boards: list[Any]) -> NDArray[np.float32]:
    """Planes (B, K, H, W) for same-size boards."""
    return np.stack([encode(b) for b in boards])
