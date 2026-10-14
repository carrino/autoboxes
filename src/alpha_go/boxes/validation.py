"""Stable replay splits, independent of file names, outcomes and replay-window size.

Ordinary games are grouped by their first eight moves (including a branch prefix),
canonicalised under board symmetries. Thus branches after move eight and their parent
stay together. This is a replay diagnostic, not a substitute for an untouched external
test set: other transpositions and a warm-start checkpoint can still leak information.
Solver-labelled position collections have no game lineage; their board contents form
the group unless the writer provides an explicit ``split_group`` scalar.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from alpha_go.boxes.rules import BoxesGeometry
from alpha_go.boxes.symmetry import apply, edge_permutation, transforms


def replay_group(game: dict[str, Any], geo: BoxesGeometry) -> bytes:
    """Return a label-independent, symmetry-canonical replay group identifier."""
    if "source_replay_group" in game:
        inherited = np.asarray(game["source_replay_group"])
        if inherited.ndim != 0 or inherited.dtype.kind not in "US":
            raise ValueError("source_replay_group must be a scalar hexadecimal string")
        scalar = inherited.item()
        encoded = scalar.decode("ascii") if isinstance(scalar, bytes) else str(scalar)
        if not encoded or len(encoded) % 2 or any(c not in "0123456789abcdef" for c in encoded):
            raise ValueError("source_replay_group must be nonempty lowercase hexadecimal")
        return bytes.fromhex(encoded)
    if "split_group" in game:
        return b"declared:" + str(game["split_group"]).encode()
    if str(game.get("termination", "")) == "solver":
        grids = np.asarray(game["boards"])
        return b"positions:" + min(
            apply(grids, k).astype(np.int8).tobytes() for k in transforms(geo.rows, geo.cols)
        )
    prefix = np.asarray(game.get("start_moves", []), dtype=np.int64).reshape(-1, 2)
    moves = np.asarray(game["moves"], dtype=np.int64).reshape(-1, 2)
    opening = np.concatenate([prefix, moves])[:8]
    assert len(opening), "cannot split an empty game"
    edges = geo.lattice_edge[opening[:, 0], opening[:, 1]]
    assert np.all(edges >= 0), "invalid edge in replay opening"
    return b"opening:" + bytes(min(
        edge_permutation(geo, k)[edges].astype("<i4").tobytes()
        for k in transforms(geo.rows, geo.cols)
    ))


def validation_game(game: dict[str, Any], geo: BoxesGeometry, fraction: float,
                    seed: int = 0) -> bool:
    """Assign a group permanently for a fixed fraction and seed, as replay grows."""
    assert 0 <= fraction < 1, "validation fraction must be in [0, 1)"
    key = f"{geo.rows}x{geo.cols}:{seed}:".encode() + replay_group(game, geo)
    number = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return number < int(fraction * (1 << 64))
