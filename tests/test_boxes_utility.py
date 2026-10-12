"""Exact utility is monotone and respects the search's 1-u perspective flip."""
from __future__ import annotations

import pytest

from alpha_go.boxes.utility import exact_value, shaped_value


@pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("k", [3.0, 6.0, 12.0])
def test_exact_utility_properties(lam: float, k: float) -> None:
    values = [exact_value(m, lam, k) for m in range(-25, 26)]
    assert values == sorted(values)
    for margin in range(-25, 26):
        win = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        assert exact_value(margin, lam, k) == shaped_value(win, margin, lam, k)
        assert exact_value(-margin, lam, k) == pytest.approx(1 - exact_value(margin, lam, k))
    assert exact_value(0, lam, k) == 0.5
