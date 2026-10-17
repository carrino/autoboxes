"""Canonical masks agree with the independent lattice transforms at byte boundaries."""
from __future__ import annotations

import random

import alpha_go_cpp
import pytest

from alpha_go.boxes.solver import Solver


@pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 4),
                                       (4, 3), (5, 5), (1, 21), (21, 1)])
def test_byte_images_against_lattice_transforms(rows: int, cols: int) -> None:
    reference = Solver(rows, cols)
    solver = alpha_go_cpp.BoxesSolver(rows, cols, 2)
    edges = reference.geo.num_edges
    full = (1 << edges) - 1
    rng = random.Random(20261017 + 31 * rows + cols)
    for shift in range(0, edges, 8):
        background = rng.getrandbits(edges) & ~(255 << shift)
        for value in range(min(256, 1 << (edges - shift))):
            mask = (value << shift) | background
            assert solver.canonical(mask) == reference.canonical(mask)
            assert solver.canonical(full ^ mask) == reference.canonical(full ^ mask)
    assert solver.canonical(0) == 0
    assert solver.canonical(full) == full
