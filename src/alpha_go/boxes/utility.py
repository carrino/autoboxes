"""Reference utilities for network estimates and exact Boxes outcomes."""
from __future__ import annotations

import math


def shaped_value(win: float, expected_margin: float, lam: float, k: float) -> float:
    """Return P(win) + lambda * tanh(E[margin] / k), with draws worth one half."""
    return win + lam * math.tanh(expected_margin / k)


def exact_value(margin: int, lam: float, k: float) -> float:
    """Apply the same utility to an exact final margin, in the requested perspective."""
    win = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
    return shaped_value(win, margin, lam, k)
