"""Teacher policy/value separation, matched draws and actual tiny CPU fitting."""

# ruff: noqa: N812
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fit_teacher_screen import (
    ARMS,
    fit_arm,
    legal_policies,
    mixed_loss,
    paired_teacher_difference,
    sample_plan,
    teacher_reads,
    teacher_tensors,
)

from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import save_boxes_net
from alpha_go.boxes.policy_target_ab import sha256
from alpha_go.boxes.rules import BoxesBoard


def test_teacher_policy_gradient_and_replay_only_value_mean() -> None:
    policy = torch.zeros(3, 4, requires_grad=True)
    values = torch.tensor(
        [[0.2, 0.5, -0.1], [-0.3, 0.4, 0.7], [-float("inf"), 0.0, -float("inf")]],
        requires_grad=True,
    )
    targets = F.one_hot(torch.tensor([0, 1, 2]), 4).float()
    # Third margin is intentionally invalid. Teacher CE must not even evaluate it.
    margins = torch.tensor([-1, 1, 500])
    total, policy_loss, value_loss = mixed_loss(
        policy, values, targets, margins, torch.tensor([1.0, 0.0, 1.0]), 2, 1
    )
    baseline = values[:2].detach().clone().requires_grad_()
    ordinary_loss = F.cross_entropy(baseline, torch.tensor([0, 2]))
    assert float(value_loss.detach()) == pytest.approx(float(ordinary_loss.detach()))
    assert float(policy_loss.detach()) == pytest.approx(float(torch.log(torch.tensor(4.0))))
    total.backward()
    ordinary_loss.backward()
    assert torch.isfinite(total)
    assert policy.grad is not None and policy.grad[2].abs().sum() > 0
    assert policy.grad[1].abs().sum() == 0
    assert values.grad is not None and values.grad[2].abs().sum() == 0
    torch.testing.assert_close(values.grad[:2], baseline.grad)


def test_teacher_prefix_keeps_capture_mover_and_rejects_drawn_target() -> None:
    # On2x2, edge7 completes the top-left box for player2, who retains the turn.
    record = dict(prefix=[0, 2, 6, 7], teacher_edge=1, source_family="capture")
    planes, targets = teacher_tensors([record], 2, 2, "chains")
    assert planes[0, 1, 1, 1] == 1 and planes[0, 2, 1, 1] == 0
    assert planes[0, 10].mean() == 0.25
    assert targets[0, 1] == 1
    with pytest.raises(ValueError, match="legal normalized"):
        teacher_tensors([dict(record, teacher_edge=7)], 2, 2, "chains")


def test_sample_plan_has_exact_quota_and_matching_cycles() -> None:
    settings = dict(seed=17, batch_size=8, teacher_per_batch=2, steps=4)
    replay, teacher = sample_plan(12, 4, settings)
    a, b = sample_plan(12, 4, settings)
    assert np.array_equal(replay, a) and np.array_equal(teacher, b)
    assert replay.shape == (4, 6) and teacher.shape == (4, 2)
    assert sorted(replay[:2].flat) == list(range(12))
    assert sorted(teacher[:2].flat) == list(range(4))


def tiny_replay() -> BoxesDataset:
    board = BoxesBoard(2)
    grids, movers, moves, visits = [], [], [], []
    for edge in (0, 2, 6, 7, 1, 3):
        grids.append(board.to_numpy())
        movers.append(board.player())
        moves.append(board.row_col(edge))
        policy = np.zeros(board.geo.num_edges, np.float32)
        policy[edge] = 1
        visits.append(policy)
        assert board.play_edge(edge)
    game = dict(
        boards=np.stack(grids),
        to_play=np.array(movers),
        moves=np.array(moves),
        result=np.array("B+2.0"),
        mcts_visits=np.stack(visits),
        mcts_root_values=np.zeros(len(grids)),
        source_family="replay",
    )
    return BoxesDataset([], games=[game], features="chains")


def test_real_cpu_arms_match_indices_and_augmentation_without_changing_reference(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    checkpoint = tmp_path / "reference.pt"
    save_boxes_net(
        BoxesNet(2, channels=4, n_blocks=1, value_hidden=4, features="chains"), checkpoint
    )
    original = sha256(checkpoint)
    model = BoxesNet(2, channels=4, n_blocks=1, value_hidden=4, features="chains")
    records = [
        dict(prefix=[0, 2, 6, 7], teacher_edge=1, source_family="capture"),
        dict(prefix=[1, 3], teacher_edge=0, source_family="other"),
    ]
    planes, targets = teacher_tensors(records, 2, 2, "chains")
    teachers = {
        name: dict(records=records, planes=planes, targets=targets)
        for name in ("training", "validation", "diagnostic")
    }
    soft = legal_policies(model, planes, 4, torch.device("cpu"))
    assert soft[0, [0, 2, 6, 7]].sum() == 0
    settings = dict(seed=17, batch_size=4, teacher_per_batch=1, steps=2, lr=1e-3)
    dataset = tiny_replay()
    replay, teacher = sample_plan(len(dataset), len(records), settings)
    results = []
    for arm in ARMS:
        output = tmp_path / arm
        output.mkdir()
        result = fit_arm(
            arm,
            checkpoint,
            dataset,
            dataset,
            {},
            teachers,
            soft,
            replay,
            teacher,
            output,
            settings,
            torch.device("cpu"),
        )
        assert result["teacher_exposures"] == 2
        assert result["teacher_value_exposures"] == 0
        assert result["replay_value_exposures"] == 6
        assert result["unique_teacher_rows"] == 2
        assert result["teacher_fraction"] == 0.25
        assert (output / "model.pt").is_file()
        assert json.loads((output / "summary.json").read_text())["status"] == "complete"
        results.append(result)
    assert results[0]["index_trace_sha256"] == results[1]["index_trace_sha256"]
    assert results[0]["augmentation_schedule_sha256"] == results[1]["augmentation_schedule_sha256"]
    assert sha256(checkpoint) == original
    assert results[0]["checkpoint_sha256"] != results[1]["checkpoint_sha256"]


def test_teacher_read_uses_legal_support_and_family_bootstrap() -> None:
    model = BoxesNet(2, channels=4, n_blocks=1, value_hidden=4, features="chains")
    records = [dict(prefix=[0, 2, 6, 7], teacher_edge=1, source_family="one")]
    planes, targets = teacher_tensors(records, 2, 2, "chains")
    read = teacher_reads(model, records, planes, targets, 4, torch.device("cpu"))
    assert read["mean_teacher_probability"] == pytest.approx(1 / 8)
    a = dict(
        records=[
            dict(source_family="a", teacher_ce=2.0, teacher_probability=0.1, agrees=False),
            dict(source_family="b", teacher_ce=2.0, teacher_probability=0.1, agrees=False),
        ]
    )
    b = dict(
        records=[
            dict(source_family="a", teacher_ce=1.0, teacher_probability=0.2, agrees=True),
            dict(source_family="b", teacher_ce=1.0, teacher_probability=0.2, agrees=True),
        ]
    )
    difference = paired_teacher_difference(a, b, 9)
    assert difference["families"] == 2
    assert difference["metrics"]["teacher_ce"] == dict(delta=-1.0, ci95=[-1.0, -1.0])
    assert difference["metrics"]["top_move_agreement"] == dict(delta=1.0, ci95=[1.0, 1.0])


def test_resume_reuses_completed_arms_and_rejects_changed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fit_teacher_screen as fitter
    import prepare_teacher_screen as preparer

    torch.set_num_threads(1)
    reference = tmp_path / "reference.pt"
    save_boxes_net(
        BoxesNet(2, channels=4, n_blocks=1, value_hidden=4, features="chains"), reference
    )
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "receipt.json").write_text("{}")
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "receipt.json").write_text("{}")
    records = [
        dict(prefix=[0, 2, 6, 7], teacher_edge=1, source_family="capture"),
        dict(prefix=[1, 3], teacher_edge=0, source_family="other"),
    ]
    content = {
        key: records for key in ("train_records", "validation_records", "diagnostic_records")
    }
    data = tiny_replay()
    monkeypatch.setattr(preparer, "load_payload", lambda _: content)
    monkeypatch.setattr(preparer, "load_replay", lambda *args: (data, data, {}, {}))
    monkeypatch.setattr(
        fitter, "bundle_files", lambda p: {"receipt.json": sha256(p / "receipt.json")}
    )
    real_fit = fitter.fit_arm
    calls = []

    def record_fit(*args, **kwargs):
        calls.append(args[0])
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(fitter, "fit_arm", record_fit)
    settings = dict(steps=1, batch_size=4, teacher_per_batch=1, lr=0.001, seed=17, cpu=True)
    output = tmp_path / "fit"
    first = fitter.run(reference, replay, payload, output, **settings)
    assert first["status"] == "complete" and calls == list(ARMS)
    second = fitter.run(reference, replay, payload, output, **settings)
    assert calls == list(ARMS) and second == first
    with pytest.raises(ValueError, match="changed on resume"):
        fitter.run(reference, replay, payload, output, **{**settings, "steps": 2})
    receipt = json.loads((output / ARMS[0] / "completed.json").read_text())
    (output / ARMS[0] / receipt["attempt"] / "model.pt").write_bytes(b"altered")
    with pytest.raises(ValueError, match="Completed arm changed"):
        fitter.run(reference, replay, payload, output, **settings)
    assert calls == list(ARMS)
