"""Tests for the training-time symmetry augmentation (planes and policy must move together)."""
# ruff: noqa: N806
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from alpha_go.boxes.augment import augment, inverse_edge_perms, torch_apply
from alpha_go.boxes.encode import encode
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.symmetry import apply, transforms

CPU = torch.device("cpu")


def random_board(rows: int, cols: int, rng: random.Random) -> BoxesBoard:
    board = BoxesBoard(rows, cols)
    for e in rng.sample(range(board.num_edges()), rng.randint(0, board.num_edges() - 1)):
        board.play_edge(e)
    return board


class TestAugment:
    @pytest.mark.parametrize("rows,cols", [(3, 3), (2, 3), (5, 5)])
    def test_torch_apply_matches_numpy(self, rows: int, cols: int) -> None:
        grid = np.random.default_rng(0).random((3, 2 * rows + 1, 2 * cols + 1)).astype(np.float32)
        for k in transforms(rows, cols):
            assert np.array_equal(torch_apply(torch.from_numpy(grid), k).numpy(), apply(grid, k))

    @pytest.mark.parametrize("rows,cols", [(3, 3), (2, 3), (5, 5)])
    def test_one_hot_policy_lands_on_the_transformed_edge(self, rows: int, cols: int) -> None:
        # A one-hot at edge e must end up on the edge sitting at the transformed lattice cell.
        geo = geometry(rows, cols)
        perms = inverse_edge_perms(rows, cols, CPU)
        ks = transforms(rows, cols)
        planes = torch.from_numpy(np.stack([encode(BoxesBoard(rows, cols))]))
        for k in ks:
            index_grid = apply(geo.lattice_edge, k)  # transformed cell -> original edge

            class Chooser:
                def choice(self, n: int, size: int) -> np.ndarray:
                    return np.full(size, ks.index(k))

            for e in range(geo.num_edges):
                policy = torch.zeros(1, geo.num_edges)
                policy[0, e] = 1.0
                _, out = augment(planes.clone(), policy, rows, cols, perms, Chooser())  # type: ignore[arg-type]
                r, c = geo.edge_rc[int(out[0].argmax())]
                assert index_grid[r, c] == e

    def test_batch_keeps_policy_support_on_undrawn_edges(self) -> None:
        # After a random per-sample transform, policy mass must sit exactly where plane 0
        # (drawn edges) says the edge is still undrawn, for every sample.
        rows, cols = 3, 3
        geo = geometry(rows, cols)
        rng = random.Random(3)
        boards = [random_board(rows, cols, rng) for _ in range(32)]
        planes = torch.from_numpy(np.stack([encode(b) for b in boards]))
        policy = torch.zeros(len(boards), geo.num_edges)
        for i, b in enumerate(boards):
            policy[i, b.get_legal_moves_flat()] = 1.0 / len(b.get_legal_moves_flat())
        planes, policy = augment(planes, policy, rows, cols, inverse_edge_perms(rows, cols, CPU),
                                 np.random.default_rng(7))
        cells = [geo.edge_rc[e] for e in range(geo.num_edges)]
        for i in range(len(boards)):
            drawn = torch.tensor([planes[i, 0, r, c] for r, c in cells])
            assert torch.all((policy[i] > 0) == (drawn == 0)), i
