"""Tests for the Boxes encoder and network."""
# ruff: noqa: N806
from __future__ import annotations

import random

import alpha_go_cpp
import numpy as np
import pytest
import torch

from alpha_go.boxes.encode import NUM_PLANES, edge_flat_index, encode, encode_batch, encode_grid
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.symmetry import apply, edge_permutation, transforms


def random_board(rows: int, cols: int, rng: random.Random) -> tuple[BoxesBoard, list[int]]:
    board = BoxesBoard(rows, cols)
    moves = rng.sample(range(board.num_edges()), rng.randint(0, board.num_edges() - 1))
    for e in moves:
        board.play_edge(e)
    return board, moves


class TestEncode:
    def test_planes_describe_the_position(self) -> None:
        board = BoxesBoard(1, 2)
        for e in (0, 1, 2, 3, 4, 6):
            board.play_edge(e)
        planes = encode(board)
        assert planes.shape == (NUM_PLANES, 3, 5) and planes.dtype == np.float32
        assert planes[0].sum() == 6
        assert planes[3:8].sum() == 2 and planes[6, 1, 1] == 1 and planes[6, 1, 3] == 1  # 3 sides
        assert planes[8].sum() == 7 and planes[9].sum() == 2
        board.play_edge(5)  # PLAYER_1 captures both, still to move
        planes = encode(board)
        assert planes[1].sum() == 2 and planes[2].sum() == 0 and planes[7].sum() == 2
        assert planes[10, 0, 0] == pytest.approx(1.0)

    def test_perspective_flips_with_side_to_move(self) -> None:
        grid = BoxesBoard(2, 2).to_numpy()
        grid[1, 1] = 1  # box owned by player 1
        a, b = encode_grid(grid, 1), encode_grid(grid, 2)
        assert a[1].sum() == 1 and a[2].sum() == 0 and a[10, 0, 0] == pytest.approx(0.25)
        assert b[1].sum() == 0 and b[2].sum() == 1 and b[10, 0, 0] == pytest.approx(-0.25)

    def test_cpp_and_python_boards_encode_alike(self) -> None:
        rng = random.Random(4)
        py, moves = random_board(3, 3, rng)
        cpp = alpha_go_cpp.BoxesBoard(3, 3)
        for e in moves:
            cpp.play_edge(e)
        assert np.array_equal(encode(py), encode(cpp))
        assert encode_batch([py, cpp]).shape == (2, NUM_PLANES, 7, 7)

    @pytest.mark.parametrize("rows,cols", [(2, 3), (3, 3)])
    def test_equivariance_under_symmetries(self, rows: int, cols: int) -> None:
        rng = random.Random(rows + cols)
        geo = geometry(rows, cols)
        for _ in range(5):
            board, moves = random_board(rows, cols, rng)
            for k in transforms(rows, cols):
                perm = edge_permutation(geo, k)
                mirror = BoxesBoard(rows, cols)
                for e in moves:
                    mirror.play_edge(int(perm[e]))
                assert np.array_equal(apply(encode(board), k), encode(mirror))

    def test_edge_gather_matches_permutation(self) -> None:
        """Gathering a transformed logit map equals permuting the gathered logits."""
        rows, cols = 3, 3
        geo = geometry(rows, cols)
        flat = edge_flat_index(rows, cols)
        logits = np.random.default_rng(0).random((2 * rows + 1, 2 * cols + 1))
        for k in transforms(rows, cols):
            perm = edge_permutation(geo, k)
            gathered_then_permuted = np.empty(geo.num_edges)
            gathered_then_permuted[perm] = logits.reshape(-1)[flat]
            transformed_then_gathered = apply(logits, k).reshape(-1)[flat]
            assert np.allclose(gathered_then_permuted, transformed_then_gathered)


class TestBoxesNet:
    @pytest.mark.parametrize("rows,cols", [(3, 3), (2, 3)])
    def test_shapes_and_values(self, rows: int, cols: int) -> None:
        net = BoxesNet(rows, cols, channels=16, n_blocks=2, value_hidden=8)
        boards = [alpha_go_cpp.BoxesBoard(rows, cols) for _ in range(3)]
        boards[1].play_edge(0)
        planes = torch.from_numpy(encode_batch(boards))
        policy_BE, margin_BM = net(planes)
        assert policy_BE.shape == (3, geometry(rows, cols).num_edges)
        assert margin_BM.shape == (3, 2 * rows * cols + 1)
        win = net.win_prob(margin_BM)
        assert win.shape == (3,) and torch.all((win >= 0) & (win <= 1))
        assert torch.allclose(win, torch.full((3,), 0.5))  # zero-init value head
        assert torch.allclose(net.expected_margin(margin_BM), torch.zeros(3), atol=1e-6)

    def test_loss_and_gradients(self) -> None:
        net = BoxesNet(3, channels=16, n_blocks=2, value_hidden=8)
        planes = torch.from_numpy(encode_batch([alpha_go_cpp.BoxesBoard(3) for _ in range(4)]))
        target_policy = torch.full((4, net.num_edges), 1.0 / net.num_edges)
        target_margin = torch.tensor([3, -1, 0, 9])
        total, policy_loss, value_loss = net.compute_loss(planes, target_policy, target_margin)
        assert torch.isfinite(total) and total.item() > 0
        assert policy_loss.item() == pytest.approx(np.log(net.num_edges), abs=1e-5)
        total.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name

    def test_win_prob_from_a_peaked_distribution(self) -> None:
        net = BoxesNet(2, 2)
        margin_BM = torch.full((1, net.num_margins), -30.0)
        margin_BM[0, net.num_boxes + 2] = 30.0  # margin +2
        assert net.win_prob(margin_BM).item() == pytest.approx(1.0)
        margin_BM[0, net.num_boxes + 2] = -30.0
        margin_BM[0, net.num_boxes] = 30.0  # margin 0
        assert net.win_prob(margin_BM).item() == pytest.approx(0.5)
