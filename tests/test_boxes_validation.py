"""Replay assignments survive window turnover, relabelling, branching and symmetry."""
from __future__ import annotations

import random

import numpy as np

from alpha_go.boxes.rules import geometry
from alpha_go.boxes.symmetry import edge_permutation
from alpha_go.boxes.validation import replay_group, validation_game


def game(seed: int) -> dict:
    geo = geometry(3, 3)
    edges = list(range(geo.num_edges))
    random.Random(seed).shuffle(edges)
    return {"moves": np.array([geo.edge_rc[e] for e in edges], dtype=np.int16)}


def test_membership_does_not_depend_on_window_order_or_labels() -> None:
    geo = geometry(3, 3)
    games = [game(i) for i in range(100)]
    before = [validation_game(g, geo, 0.2) for g in games]
    assert 5 < sum(before) < 40
    for i, g in reversed(list(enumerate(games))):
        g["result"] = "B+9.0"
        g["exact_margin_p1"] = np.ones(24)
        assert validation_game(g, geo, 0.2) == before[i]
        assert not validation_game(g, geo, 0)


def test_branch_and_symmetric_parent_share_group() -> None:
    geo = geometry(3, 3)
    parent = game(4)
    branch = {"start_moves": parent["moves"][:12], "moves": parent["moves"][12:]}
    perm = edge_permutation(geo, 5)
    edges = [int(geo.lattice_edge[r, c]) for r, c in parent["moves"]]
    symmetric = {"moves": np.array([geo.edge_rc[int(perm[e])] for e in edges])}
    assert replay_group(parent, geo) == replay_group(branch, geo) == replay_group(symmetric, geo)


def test_declared_lineage_survives_short_branch_and_new_continuation() -> None:
    geo = geometry(3, 3)
    a, b = game(1), game(2)
    a["split_group"] = b["split_group"] = np.array("source-game-123")
    assert replay_group(a, geo) == replay_group(b, geo)
