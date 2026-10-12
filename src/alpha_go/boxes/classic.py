"""Chain-theory evaluation reference for the optional classical Boxes engine.

The search stores remaining margins in hundredths of a box; the board's score is a
constant offset at the root. Only the existing loony endgame recurrence is exact.
Control parity and controlled value in positions with junctions are heuristics, never
proofs or bounds for the exact solver.
"""
from __future__ import annotations

from typing import Any

from alpha_go.boxes.chains import components, degrees, independent
from alpha_go.boxes.forced import collapse_mask
from alpha_go.boxes.rules import BoxesGeometry, geometry
from alpha_go.boxes.solver import completed, loony_value

SCALE = 100


def controlled_value(chains: tuple[int, ...], loops: tuple[int, ...]) -> int:
    """Controller's guaranteed margin from keeping control of long chains/even loops.

    Each chain costs a two-box handout (four margin points), each loop a four-box
    handout (eight points). The final component is taken in full: terminal bonus 8
    for loops alone, 6 for loops plus only three-chains, otherwise 4. This is the
    standard controlled value, not the exact game value when it is below 2.
    """
    assert all(length >= 3 for length in chains)
    assert all(length >= 4 and length % 2 == 0 for length in loops)
    if not chains and not loops:
        return 0
    bonus = 8 if not chains else 6 if loops and max(chains) == 3 else 4
    return sum(length - 4 for length in chains) + sum(length - 8 for length in loops) + bonus


def quiet_estimate(mask: int, geo: BoxesGeometry) -> int:
    """Estimate a quiet position in hundredths of a box, from its mover's perspective.

    If there are C eventual long chains and B remaining boxes, a loony position has
    B+C undrawn edges (loops contribute one per box). U-B-C is therefore the estimated
    number of noncapturing turns before someone must open the ending. Odd means the
    present mover gets control. Junctions can change C, so divide this speculative
    advantage by 1+junctions. Short chains and unassigned boxes are counted in the
    potential haul, but this approximation is bypassed for every simple ending.
    """
    deg = degrees(mask, geo)
    comps = components(mask, geo)
    assert 1 not in deg, "quiet_estimate needs forced captures resolved first"
    chains = tuple(sorted(c.size for c in comps if not c.is_loop))
    loops = tuple(sorted(c.size for c in comps if c.is_loop))
    if max(deg, default=0) <= 2 and all(independent(c) for c in comps):
        return SCALE * loony_value(chains, loops)
    long_chains = tuple(length for length in chains if length >= 3)
    remaining = sum(d > 0 for d in deg)
    unknown = remaining - sum(long_chains) - sum(loops)
    potential = max(0, controlled_value(long_chains, loops) + unknown)
    junctions = sum(d >= 3 for d in deg)
    advantage = min(SCALE * remaining, SCALE * potential // (1 + junctions))
    safe_turns = max(0, geo.num_edges - mask.bit_count() - remaining - len(long_chains))
    return advantage if safe_turns % 2 else -advantage


def remaining_estimate(mask: int, geo: BoxesGeometry) -> int:
    """Resolve forced captures and handout choices before evaluating the quiet leaf."""
    quiet, _, decision = collapse_mask(mask, geo)
    gained = SCALE * completed(mask, quiet, geo)
    if quiet == (1 << geo.num_edges) - 1:
        return gained
    if decision is not None:
        after = quiet | sum(1 << edge for edge in decision.take)
        take = SCALE * completed(quiet, after, geo) + remaining_estimate(after, geo)
        control = -remaining_estimate(quiet | (1 << decision.control), geo)
        return gained + max(take, control)
    return gained + quiet_estimate(quiet, geo)


def evaluate(board: Any) -> float:
    """Estimated final margin for the side to move, including the score already earned."""
    mask = int(board.edges()) if callable(board.edges) else int(board.edges)
    geo = geometry(board.rows(), board.cols())
    return float(board.margin()) + remaining_estimate(mask, geo) / SCALE
