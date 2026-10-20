"""Evaluation recovery preserves training artifacts and the frozen experiment identity."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def recovery() -> Any:
    path = (Path(__file__).resolve().parents[1] /
            "experiments/2026-10-20_08-13-boxes-balanced-policy/resume_evaluation.py")
    spec = importlib.util.spec_from_file_location("balanced_recovery", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_run(recovery: Any, output: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Provide small valid stage receipts without GPU training or search."""
    runner = recovery.runner
    config = dict(scope="test", cpu=True, seed=20261020, prepared="unchanged-data",
                  training=dict(steps=2048, batch_size=256, lr=.0001, root_fraction=.25),
                  caps_seconds=dict(runner.CAPS))
    runner.write_json(output / "config.json", config)
    monkeypatch.setattr(runner, "verify", lambda *_: None)
    (output / "inputs").mkdir()
    (output / "inputs/iter24.pt").write_bytes(b"original")

    def training_job(_command: Any, log: Path, _seconds: int) -> None:
        folder = log.parent / "train"
        folder.mkdir()
        (folder / "control.pt").write_bytes(b"trained-control")
        (folder / "balanced.pt").write_bytes(b"trained-balanced")
        (folder / "validation-positions.jsonl").write_text('{"fresh": true}\n')
        runner.write_json(folder / "validation-labels.json",
                          dict(schema=1, rows=5, cols=5, remaining_values={"123": 1}))
        runner.write_json(folder / "summary.json", {"status": "complete"})
        log.write_text("training completed\n")

    monkeypatch.setattr(runner.helpers, "job", training_job)
    runner.pilot.stage(output, "train", 5, lambda _: ["training-fixture"])
    return config


def test_extension_only_runs_evaluation_and_preserves_partial_attempt(
    recovery: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alpha_go.boxes import probe_report

    runner = recovery.runner
    original = setup_run(recovery, tmp_path, monkeypatch)
    original_config = (tmp_path / "config.json").read_bytes()
    training = {str(p): p.read_bytes() for p in (tmp_path / "train").rglob("*") if p.is_file()}
    old = tmp_path / "control-vs-balanced/attempt-old/probe"
    old.mkdir(parents=True)
    (old / "positions.csv").write_text("partial diagnostics\n")
    runner.write_json(old / "labels.json",
                      dict(schema=1, rows=5, cols=5, remaining_values={"123": 1, "456": -1}))
    calls = []

    def evaluation_job(command: list[str], log: Path, seconds: int) -> None:
        assert command[3] == "alpha_go.boxes.search_probe", "attempted to retrain"
        assert seconds == 10800
        assert command[command.index("--seed") + 1] == str(original["seed"])
        assert command[command.index("--simulations") + 1] == "400"
        assert command[command.index("--solver-budget") + 1] == "50000"
        calls.append(command)
        folder = Path(command[command.index("--output") + 1])
        folder.mkdir()
        cache = json.loads(Path(command[command.index("--label-cache") + 1]).read_text())
        assert cache["remaining_values"] == {"123": 1, "456": -1}
        runner.write_json(folder / "labels.json", cache)
        log.write_text("evaluation complete\n")

    def report(data: Any, folder: Path) -> None:
        folder.mkdir()
        runner.write_json(folder / "report.json", data)

    monkeypatch.setattr(runner.helpers, "job", evaluation_job)
    monkeypatch.setattr(probe_report, "build_report", lambda *_a, **_k: {"paired": True})
    monkeypatch.setattr(probe_report, "write_report", report)
    recovery.resume(tmp_path, 180)
    assert len(calls) == 2
    assert original == json.loads((tmp_path / "config.json").read_text())
    assert original_config == (tmp_path / "config.json").read_bytes()
    assert training == {str(p): p.read_bytes() for p in (tmp_path / "train").rglob("*")
                        if p.is_file()}
    assert (old / "positions.csv").read_text() == "partial diagnostics\n"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["evaluation_extension"]["effective_caps_seconds"]["train"] == 3600
    assert "--probe-minutes 180" in (tmp_path / "resume-evaluation.sh").read_text()
    recovery.resume(tmp_path, 180)
    assert len(calls) == 2, "completed evaluations were rerun"


def test_missing_training_cannot_start_training(
    recovery: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovery.runner.write_json(tmp_path / "config.json", {})
    monkeypatch.setattr(recovery.runner, "verify", lambda *_: None)
    monkeypatch.setattr(recovery.runner, "run", lambda *_: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="Training must be complete"):
        recovery.resume(tmp_path, 180)


def test_changed_checkpoint_rejected_before_evaluation(
    recovery: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_run(recovery, tmp_path, monkeypatch)
    checkpoint = next((tmp_path / "train").rglob("balanced.pt"))
    checkpoint.write_bytes(b"changed")
    monkeypatch.setattr(recovery.runner, "run", lambda *_: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="Missing or changed"):
        recovery.resume(tmp_path, 180)
