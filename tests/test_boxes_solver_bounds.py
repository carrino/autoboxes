"""Exact transposition bounds remain sound through collisions and budget restarts."""
from __future__ import annotations

import random

import alpha_go_cpp  # type: ignore[import-not-found]
import pytest

from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.symmetry import edge_permutation


class TestBoundReuse:
    @pytest.mark.parametrize("entries", [2, 16, 1024])
    @pytest.mark.parametrize("mtdf", [False, True])
    def test_restarted_bounds_match_oracle(self, entries: int, mtdf: bool) -> None:
        oracle = Oracle(2, 3)
        solver = alpha_go_cpp.BoxesSolver(2, 3, entries)
        solver.set_mtdf(mtdf)
        perm = edge_permutation(oracle.geo, 2)
        rng = random.Random(2718)
        for _ in range(80):
            mask = rng.getrandbits(oracle.geo.num_edges)
            image = sum(1 << int(perm[e]) for e in range(oracle.geo.num_edges) if (mask >> e) & 1)
            expected = oracle.value(mask)
            for budget in (0, 1, 8, 64):
                result = solver.value_within(mask, budget)
                assert result is None or result == expected
            assert solver.value(image) == expected
            assert solver.value(mask) == expected
        assert solver.table_bytes() == entries * 16
