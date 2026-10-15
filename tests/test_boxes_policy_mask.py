"""Value-only exact children cannot create fake policy targets or dilute policy loss."""
# ruff: noqa: N806, N812
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from alpha_go.boxes.augment import augment, inverse_edge_perms
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import save_boxes_net
from alpha_go.boxes.relabel import EXACT_UNKNOWN
from alpha_go.boxes.rules import geometry
from alpha_go.boxes.validation import replay_group
from tests.test_boxes_relabel import random_game


def masked_game() -> dict[str, Any]:
    """One searched root and two exact children with no meaningful played move."""
    source = random_game(2, 3, 8)
    game: dict[str, Any] = {key: np.asarray(source[key])[:3].copy()
                            for key in ("boards", "to_play", "moves")}
    game.update(result="B+6.0", policy_mask=np.array([True, False, False]),
                source_replay_group=np.asarray(replay_group(source, geometry(2, 3)).hex()),
                exact_margin_p1=np.array([-2, 2, -2], dtype=np.int16),
                mcts_visits=np.zeros((3, 17), dtype=np.float32),
                mcts_root_values=np.array([0.3, 0.7, 0.9], dtype=np.float32))
    game["mcts_visits"][0, 1] = 10
    game["moves"][1:] = -1
    return game


class TestPolicyMaskDataset:
    def test_absent_mask_is_identical_to_all_true(self) -> None:
        game = random_game(2, 3, 7)
        legacy = BoxesDataset([], games=[game])
        explicit = BoxesDataset([], games=[{
            **game, "policy_mask": np.ones(len(game["moves"]), dtype=bool)}])
        assert len(legacy) == len(explicit)
        for i in range(len(legacy)):
            for key, value in legacy[i].items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, explicit[i][key])
                else:
                    assert value == explicit[i][key]
            assert legacy[i]["policy_weight"].item() == 1

    def test_mask_keeps_exact_children_and_ignores_sentinel_moves(self) -> None:
        game = masked_game()
        dataset = BoxesDataset([], games=[game])
        assert len(dataset) == dataset.num_exact() == 3
        assert dataset[0]["has_mcts"] and dataset[0]["policy"].argmax().item() == 1
        for i in (1, 2):
            sample = dataset[i]
            expected = int(game["exact_margin_p1"][i]) * (1 if game["to_play"][i] == 0 else -1)
            assert sample["margin"].item() == expected
            assert sample["exact"] and not sample["has_mcts"]
            assert sample["policy_weight"].dtype == torch.float32
            assert sample["policy_weight"].item() == 0
            assert torch.count_nonzero(sample["policy"]) == 0
            assert sample["root_value"].item() == float(expected > 0)

    @pytest.mark.parametrize("mask", [np.ones(3), np.ones(2, dtype=bool),
                                      np.ones((3, 1), dtype=bool)])
    def test_invalid_masks_fail_on_load(self, mask: np.ndarray) -> None:
        with pytest.raises(ValueError, match="policy_mask must have one boolean"):
            BoxesDataset([], games=[{**masked_game(), "policy_mask": mask}])

    @pytest.mark.parametrize("problem", ["missing", "unknown", "wrong_shape", "no_exact"])
    def test_value_only_rows_require_exact_values(self, problem: str) -> None:
        game = masked_game()
        if problem == "missing":
            del game["exact_margin_p1"]
        elif problem == "unknown":
            game["exact_margin_p1"][1] = EXACT_UNKNOWN
        elif problem == "wrong_shape":
            game["exact_margin_p1"] = game["exact_margin_p1"][:1]
        with pytest.raises(ValueError, match="require known exact_margin_p1"):
            BoxesDataset([], games=[game], use_exact=problem != "no_exact")

    def test_augmentation_preserves_zero_child_policies(self) -> None:
        dataset = BoxesDataset([], games=[masked_game()])
        planes = torch.stack([dataset[i]["planes"] for i in range(3)])
        policy = torch.stack([dataset[i]["policy"] for i in range(3)])
        perms = inverse_edge_perms(2, 3, torch.device("cpu"))
        for seed in range(8):
            _, transformed = augment(planes.clone(), policy.clone(), 2, 3, perms,
                                     np.random.default_rng(seed))
            assert transformed[0].sum().item() == 1
            assert torch.count_nonzero(transformed[1:]) == 0


class TestMaskedPolicyLoss:
    @staticmethod
    def batch() -> tuple[BoxesNet, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        net = BoxesNet(2, 3, channels=8, n_blocks=1, value_hidden=8).eval()
        planes = torch.randn(3, 11, 5, 7)
        policy = torch.softmax(torch.randn(3, net.num_edges), dim=-1)
        return net, planes, policy, torch.tensor([2, -2, 4])

    def test_legacy_loss_formula_and_positional_arguments_are_unchanged(self) -> None:
        net, planes, policy, margin = self.batch()
        win, weight = torch.tensor([0.9, 0.2, 0.7]), torch.tensor([1., 2., 3.])
        total, policy_loss, value_loss = net.compute_loss(planes, policy, margin, win, weight)
        logits, values = net(planes)
        expected_policy = -(policy * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        predicted = net.win_prob(values).clamp(1e-6, 1 - 1e-6)
        expected_value = ((F.cross_entropy(values, margin + net.num_boxes, reduction="none")
                           - win * predicted.log() - (1 - win) * (1 - predicted).log())
                          * weight).mean()
        assert torch.equal(policy_loss, expected_policy)
        assert torch.allclose(value_loss, expected_value)
        assert torch.equal(total, policy_loss + value_loss)
        weighted = net.compute_loss(planes, policy, margin, win, weight, torch.ones(3))
        assert all(torch.equal(a, b) for a, b in zip((total, policy_loss, value_loss), weighted))

    @pytest.mark.parametrize("autocast", [False, True])
    def test_all_children_have_zero_policy_gradient_and_nonzero_value_gradient(
        self, autocast: bool,
    ) -> None:
        net, planes, policy, margin = self.batch()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            total, policy_loss, value_loss = net.compute_loss(
                planes, policy, margin, policy_weight_B=torch.zeros(3))
        assert policy_loss.item() == 0 and torch.isfinite(total)
        assert torch.equal(total, value_loss)
        total.backward()
        assert net.policy_conv.weight.grad is not None
        assert torch.count_nonzero(net.policy_conv.weight.grad) == 0
        assert net.value_fc2.weight.grad is not None
        assert torch.count_nonzero(net.value_fc2.weight.grad) > 0

    def test_adding_child_rows_does_not_dilute_policy_gradient(self) -> None:
        net, planes, policy, margin = self.batch()
        _, original, _ = net.compute_loss(planes[:1], policy[:1], margin[:1])
        gradient, = torch.autograd.grad(original, net.policy_conv.weight)
        _, mixed, _ = net.compute_loss(planes, policy, margin,
                                      policy_weight_B=torch.tensor([1., 0., 0.]))
        mixed_gradient, = torch.autograd.grad(mixed, net.policy_conv.weight)
        assert torch.equal(original, mixed)
        assert torch.allclose(gradient, mixed_gradient, atol=1e-7)

    @pytest.mark.parametrize("weight", [torch.ones(3, 1), torch.ones(2), torch.ones(3).bool()])
    def test_bad_weight_shapes_and_types_are_rejected(self, weight: torch.Tensor) -> None:
        net, planes, policy, margin = self.batch()
        with pytest.raises(ValueError, match="policy_weight_B"):
            net.compute_loss(planes, policy, margin, policy_weight_B=weight)


@pytest.mark.parametrize("value_only", [False, True])
def test_training_cli_counts_only_policy_rows(tmp_path: Path, value_only: bool) -> None:
    """Exercise the real train/evaluate path with exact children in its replay manifest."""
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments/2026-10-08_14-42-boxes-3x3-loop/train.py"
    games = tmp_path / "games"
    games.mkdir()
    game = masked_game()
    if value_only:
        game["policy_mask"][:] = False
    np.savez_compressed(games / "positions.npz", **game)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(str(games) + "\n")
    checkpoint = tmp_path / "start.pt"
    save_boxes_net(BoxesNet(2, 3, channels=8, n_blocks=1, value_hidden=8), checkpoint)
    command = [sys.executable, str(script), "--dataset-txt", str(manifest), "--iteration", "1",
               "--rows", "2", "--cols", "3", "--tag", str(tmp_path / "trained"),
               "--resume-from", str(checkpoint), "--cpu", "--max-epochs", "1",
               "--time-budget", "0", "--val-fraction", "0", "--batch-size", "8"]
    result = subprocess.run(command, capture_output=True, text=True, check=True,
                            env={**os.environ, "OMP_NUM_THREADS": "1"})
    report = json.loads(result.stdout.split("===RESULT===\n")[1])
    assert report["positions"] == 3
    assert report["train_policy_positions"] == int(not value_only)
    assert report["train_policy_acc"] in (0., 1.)
    if value_only:
        assert report["train_policy_acc"] == 0
    assert report["val_positions"] == report["val_policy_positions"] == 0
    assert report["steps_completed"] == 1
