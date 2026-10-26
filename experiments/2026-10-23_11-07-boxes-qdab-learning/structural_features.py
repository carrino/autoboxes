"""Isolated structural-value feature prototype; not wired into production inference.

Counts describe the current strings-and-coins graph. They do not assert who has
control or apply closed-endgame formulas to a mixed middlegame.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from alpha_go.boxes.chains import GROUND, OPEN, components, degrees
from alpha_go.boxes.rules import BoxesBoard, BoxesGeometry

SCHEMA = "structural-value-v1"
HISTOGRAMS = ("ground_chain", "junction_path", "opened_path", "loop")
ENDPOINTS = ("GG", "GJ", "GO", "JJ", "JO", "OO")
SCALARS = (
    *(f"degree_{degree}" for degree in range(5)),
    "safe_edges", "capture_edges", "double_capture_edges", "undrawn_edges",
    "mover_score", "opponent_score", "unclaimed_boxes", "score_margin",
    *(f"endpoints_{pair}" for pair in ENDPOINTS),
    "same_junction_paths", "long_chain_parity", "has_junction",
    "quiet_closed_endgame",
)


def feature_names(geo: BoxesGeometry) -> tuple[str, ...]:
    """Stable names; histograms retain every possible box length for this board."""
    return tuple(f"{kind}_length_{length}" for kind in HISTOGRAMS
                 for length in range(1, geo.num_boxes + 1)) + SCALARS


def structural_features(
    mask: int, geo: BoxesGeometry, mover_score: int, opponent_score: int,
) -> NDArray[np.float32]:
    """Return 4*boxes+23 scalars, with scores explicitly in mover perspective.

    Histograms, degree and endpoint counts divide by the number of boxes; edge
    counts divide by the number of edges. The last three flags are unscaled.
    """
    if mask < 0 or mask >> geo.num_edges:
        raise ValueError("Edge mask exceeds geometry")
    deg = degrees(mask, geo)
    if min(mover_score, opponent_score) < 0 or mover_score + opponent_score != deg.count(0):
        raise ValueError("Scores must account for every completed box")
    values = dict.fromkeys(feature_names(geo), 0.0)
    comps = components(mask, geo)
    for comp in comps:
        if comp.is_loop:
            kind = "loop"
        elif comp.opened:
            kind = "opened_path"
        elif comp.ends == (GROUND, GROUND):
            kind = "ground_chain"
        else:
            kind = "junction_path"
        values[f"{kind}_length_{comp.size}"] += 1 / geo.num_boxes
        if not comp.is_loop:
            ends = sorted("G" if end == GROUND else "O" if end == OPEN else "J"
                          for end in comp.ends)
            values[f"endpoints_{''.join(ends)}"] += 1 / geo.num_boxes
            if comp.ends[0] >= 0 and comp.ends[0] == comp.ends[1]:
                values["same_junction_paths"] += 1 / geo.num_boxes
    for degree in range(5):
        values[f"degree_{degree}"] = deg.count(degree) / geo.num_boxes
    for edge, boxes in enumerate(geo.edge_boxes):
        if mask >> edge & 1:
            continue
        values["undrawn_edges"] += 1 / geo.num_edges
        if all(deg[box] >= 3 for box in boxes):
            values["safe_edges"] += 1 / geo.num_edges
        captured = sum(deg[box] == 1 for box in boxes)
        values["capture_edges"] += int(captured > 0) / geo.num_edges
        values["double_capture_edges"] += int(captured == 2) / geo.num_edges
    values["mover_score"] = mover_score / geo.num_boxes
    values["opponent_score"] = opponent_score / geo.num_boxes
    values["unclaimed_boxes"] = 1 - (mover_score + opponent_score) / geo.num_boxes
    values["score_margin"] = (mover_score - opponent_score) / geo.num_boxes
    values["long_chain_parity"] = sum(not comp.is_loop and comp.size >= 3
                                      for comp in comps) % 2
    values["has_junction"] = float(any(degree >= 3 for degree in deg))
    values["quiet_closed_endgame"] = float(
        bool(comps) and not any(degree in (1, 3, 4) for degree in deg)
        and all(comp.is_loop or comp.ends == (GROUND, GROUND) for comp in comps))
    return np.asarray(list(values.values()), dtype=np.float32)


def from_board(board: BoxesBoard) -> NDArray[np.float32]:
    """Reference-board convenience wrapper with no inference or solver calls."""
    mover = board.player()
    return structural_features(board.edges, board.geo, board.boxes[mover], board.boxes[1 - mover])
