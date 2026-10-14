"""Remaining-margin heads retain the final-value API and legacy checkpoint semantics."""
# ruff: noqa: N806, N812
from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from alpha_go.boxes.agents import BoxesRandomAgent
from alpha_go.boxes.encode import encode, encode_batch
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import load_boxes_net, save_boxes_net
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


def trajectory(rows: int, cols: int) -> torch.Tensor:
    """Include empty, capturing, player-swapped and terminal states from a legal game."""
    board = BoxesBoard(rows, cols)
    rng = random.Random(12)
    boards = [board.copy()]
    for edge in rng.sample(range(board.num_edges()), board.num_edges()):
        board.play_edge(edge)
        boards.append(board.copy())
    return torch.from_numpy(encode_batch(boards))


class TestRemainingMargin:
    @pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2), (2, 3), (3, 3)])
    def test_exact_shift_support_and_mass(self, rows: int, cols: int) -> None:
        torch.manual_seed(7)
        net = BoxesNet(rows, cols, channels=8, n_blocks=1, value_target="remaining")
        planes = trajectory(rows, cols)
        raw_BM = torch.randn(len(planes), net.num_margins)
        shifted_BM = net.remaining_to_final_logits(raw_BM, planes)
        probs_BM = shifted_BM.softmax(-1)
        assert torch.allclose(probs_BM.sum(-1), torch.ones(len(planes)))
        for i, sample in enumerate(planes):
            own, other = (int(sample[k].sum()) for k in (1, 2))
            score, remaining = own - other, net.num_boxes - own - other
            possible = torch.arange(-remaining, remaining + 1, 2)
            raw_probs = raw_BM[i, possible + net.num_boxes].softmax(-1)
            expected = torch.zeros(net.num_margins)
            expected[possible + score + net.num_boxes] = raw_probs
            assert torch.allclose(probs_BM[i], expected)
            assert net.expected_margin(shifted_BM[i:i + 1]).item() == pytest.approx(
                score + float(raw_probs @ possible.float()), abs=2e-6)
            weights = ((possible + score > 0).float()
                       + 0.5 * (possible + score == 0).float())
            assert net.win_prob(shifted_BM[i:i + 1]).item() == pytest.approx(
                float(raw_probs @ weights), abs=1e-6)
        # The final margin always has the parity of the total number of boxes.
        impossible = (net.margins_M.long() - net.num_boxes).remainder(2) != 0
        assert torch.count_nonzero(probs_BM[:, impossible]) == 0
        assert torch.count_nonzero(probs_BM[-1]) == 1  # no unclaimed boxes at terminal

    def test_final_loss_equals_remaining_loss_and_gradients(self) -> None:
        torch.manual_seed(4)
        net = BoxesNet(2, 3, channels=8, n_blocks=1, value_target="remaining")
        planes = trajectory(2, 3)
        raw_BM = torch.randn(len(planes), net.num_margins, requires_grad=True)
        own_B = planes[:, 1].sum((1, 2)).long()
        other_B = planes[:, 2].sum((1, 2)).long()
        unclaimed_B = net.num_boxes - own_B - other_B
        score_B = own_B - other_B
        margin_BM = net.margins_M.long()[None, :]
        possible_BM = ((margin_BM.abs() <= unclaimed_B[:, None])
                       & ((margin_BM - unclaimed_B[:, None]).remainder(2) == 0))
        loss_final = F.cross_entropy(net.remaining_to_final_logits(raw_BM, planes),
                                     score_B + unclaimed_B + net.num_boxes)
        loss_remaining = F.cross_entropy(raw_BM.masked_fill(~possible_BM, -torch.inf),
                                         unclaimed_B + net.num_boxes)
        assert torch.allclose(loss_final, loss_remaining)
        grad_final, = torch.autograd.grad(loss_final, raw_BM, retain_graph=True)
        grad_remaining, = torch.autograd.grad(loss_remaining, raw_BM)
        assert torch.allclose(grad_final, grad_remaining, atol=1e-7)
        assert torch.count_nonzero(grad_final[~possible_BM]) == 0

    @pytest.mark.parametrize("autocast", [False, True])
    def test_loss_backward_and_compiled_forward(self, autocast: bool) -> None:
        net = BoxesNet(2, 3, channels=8, n_blocks=1, value_hidden=8,
                       value_target="remaining")
        planes = trajectory(2, 3)
        own_B = planes[:, 1].sum((1, 2)).long()
        other_B = planes[:, 2].sum((1, 2)).long()
        target_B = own_B - other_B + net.num_boxes - own_B - other_B
        policy_BE = torch.full((len(planes), net.num_edges), 1 / net.num_edges)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            total, _, _ = net.compute_loss(planes, policy_BE, target_B,
                                           (target_B > 0).float())
        assert torch.isfinite(total)
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in net.parameters())
        net.eval()
        compiled = torch.compile(net, backend="eager", fullgraph=True)
        assert torch.equal(compiled(planes)[1], net(planes)[1])

    def test_conversion_preserves_policy_exactly(self) -> None:
        torch.manual_seed(11)
        net = BoxesNet(2, 3, channels=8, n_blocks=1, value_hidden=8).eval()
        with torch.no_grad():
            net.policy_conv.weight.normal_()
            net.value_fc2.weight.normal_()
        planes = trajectory(2, 3)
        policy_before, _ = net(planes)
        weights_before = {name: value.clone() for name, value in net.state_dict().items()}
        net.reset_value_head("remaining")
        assert net.value_target == "remaining"
        assert torch.equal(policy_before, net(planes)[0])
        for name, value in net.state_dict().items():
            if name.startswith("value_fc2."):
                assert torch.count_nonzero(value) == 0
            else:
                assert torch.equal(value, weights_before[name]), name

    def test_capturing_twice_and_terminal_win(self) -> None:
        board = BoxesBoard(1, 2)
        for edge in (0, 1, 2, 3, 4, 6):
            board.play_edge(edge)
        before = board.to_play()
        board.play_edge(5)
        assert board.to_play() == before  # taking both boxes retains the mover
        net = BoxesNet(1, 2, channels=8, n_blocks=1, value_target="remaining").eval()
        _, final_BM = net(torch.from_numpy(encode(board))[None])
        assert net.win_prob(final_BM).item() == 1.0
        assert net.expected_margin(final_BM).item() == 2.0


class TestRemainingCheckpoint:
    def test_remaining_roundtrip_and_legacy_default(self, tmp_path: Path) -> None:
        net = BoxesNet(2, 3, channels=8, n_blocks=1, value_target="remaining").eval()
        planes = trajectory(2, 3)
        path = tmp_path / "net.pt"
        save_boxes_net(net, path)
        loaded = load_boxes_net(path, torch.device("cpu"))
        assert loaded.value_target == "remaining"
        assert torch.equal(loaded(planes)[1], net(planes)[1])
        # Old checkpoints omit the setting. Their unmasked final logits stay identical.
        net.reset_value_head("final")
        with torch.no_grad():
            net.value_fc2.weight.normal_()
        save_boxes_net(net, path)
        checkpoint = torch.load(path, weights_only=False)
        del checkpoint["config"]["value_target"]
        torch.save(checkpoint, path)
        legacy = load_boxes_net(path, torch.device("cpu"))
        assert legacy.value_target == "final"
        assert torch.equal(legacy(planes)[0], net(planes)[0])
        assert torch.equal(legacy(planes)[1], net(planes)[1])
        assert torch.isfinite(legacy(planes)[1]).all()

    def test_cli_requires_explicit_conversion_and_inherits_on_resume(self, tmp_path: Path) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "experiments/2026-10-08_14-42-boxes-3x3-loop/train.py"
        game_dir = tmp_path / "games"
        game_dir.mkdir()
        record = play_game(BoxesRandomAgent(), BoxesRandomAgent(),
                           board_size=2, seed=1, max_moves=12, game=get_game("boxes"))
        save_game_data(record, game_dir, 0, "test")
        manifest = tmp_path / "dataset.txt"
        manifest.write_text(str(game_dir) + "\n")
        start = tmp_path / "final.pt"
        save_boxes_net(BoxesNet(2, channels=8, n_blocks=1, value_hidden=8), start)
        output = tmp_path / "checkpoints"
        command = [sys.executable, str(script), "--dataset-txt", str(manifest),
                   "--rows", "2", "--tag", str(output), "--cpu", "--max-epochs", "1",
                   "--time-budget", "0", "--val-fraction", "0", "--batch-size", "8"]
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        rejected = subprocess.run(
            command + ["--iteration", "1", "--resume-from", str(start),
                       "--value-target", "remaining"], env=env, capture_output=True, text=True)
        assert rejected.returncode != 0
        assert "requires --reset-value-head" in rejected.stderr
        assert not output.exists()
        converted = subprocess.run(
            command + ["--iteration", "1", "--resume-from", str(start),
                       "--value-target", "remaining", "--reset-value-head"],
            env=env, capture_output=True, text=True, check=True)
        assert "reset value readout: final -> remaining" in converted.stdout
        assert load_boxes_net(output / "iter1.pt", torch.device("cpu")).value_target == "remaining"
        resumed = subprocess.run(
            command + ["--iteration", "2", "--resume-from", str(output / "iter1.pt")],
            env=env, capture_output=True, text=True, check=True)
        assert '"value_target": "remaining"' in resumed.stdout
        assert "reset value readout" not in resumed.stdout
        assert load_boxes_net(output / "iter2.pt", torch.device("cpu")).value_target == "remaining"
        # Shaped utilities may lie inside [0, 1]; checking that numeric range alone is
        # insufficient to establish that they are probabilities for the binary target.
        game_path = next(game_dir.glob("*.npz"))
        game = dict(np.load(game_path))
        game["root_value_semantics"] = np.asarray("shaped_utility")
        np.savez_compressed(game_path, **game)
        shaped = subprocess.run(command + ["--iteration", "3", "--q-mix", "0.25"],
                                env=env, capture_output=True, text=True)
        assert shaped.returncode != 0
        assert "replay contains margin-shaped search Q" in shaped.stderr
        assert not (output / "iter3.pt").exists()
