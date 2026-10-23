"""Exercise paired runner wiring, checkpoint identity and completed-stage recovery."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def runner() -> Any:
    path = (Path(__file__).resolve().parents[1] /
            "experiments/2026-10-20_08-13-boxes-balanced-policy/run_balanced.py")
    spec = importlib.util.spec_from_file_location("balanced_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(command: list[str]) -> dict[str, str]:
    """Read CLI option values without assuming an option order."""
    return {value: command[index + 1] for index, value in enumerate(command[:-1])
            if value.startswith("--") and value != "--cpu"}


def test_runner_resumes_fixed_endpoints_and_fresh_validation(
    runner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alpha_go.boxes import probe_report

    config = dict(
        scope="test", prepared=str(tmp_path / "prepared"), seed=20261020, cpu=True,
        caps_seconds=runner.CAPS,
        training=dict(steps=2048, batch_size=256, lr=.0001, root_fraction=.25))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "iter24.pt").write_bytes(b"frozen reference")
    calls = []

    def job(command: list[str], log: Path, _seconds: int) -> None:
        calls.append(command)
        args = options(command)
        folder = Path(args["--output"])
        folder.mkdir()
        log.write_text("complete\n")
        if command[3] == "alpha_go.boxes.balanced_policy":
            (folder / "control.pt").write_bytes(b"ordinary")
            (folder / "balanced.pt").write_bytes(b"balanced")
            runner.write_json(folder / "summary.json", {"status": "complete"})
            (folder / "validation-positions.jsonl").write_text('{"fresh": true}\n')
            runner.write_json(folder / "validation-labels.json",
                              dict(schema=1, rows=5, cols=5, remaining_values={"123": 1}))
        else:
            assert Path(args["--positions-file"]).name == "validation-positions.jsonl"
            assert args["--min-undrawn"] == "35" and args["--max-undrawn"] == "36"
            assert args["--solver"] == "34" and args["--solver-budget"] == "50000"
            assert args["--simulations"] == "400" and "--cpu" in command
            assert Path(args["--candidate"]).read_bytes() == b"balanced"
            cache = json.loads(Path(args["--label-cache"]).read_text())
            assert cache["remaining_values"]["123"] == 1
            runner.write_json(folder / "labels.json", cache)

    monkeypatch.setattr(runner.helpers, "job", job)
    monkeypatch.setattr(probe_report, "build_report",
                        lambda *_args, **_kwargs: {"paired": True})

    def report(value: Any, destination: Path) -> None:
        destination.mkdir()
        runner.write_json(destination / "report.json", value)

    monkeypatch.setattr(probe_report, "write_report", report)
    runner.run(tmp_path, config)
    assert len(calls) == 3
    training = options(calls[0])
    assert training["--steps"] == "2048" and training["--root-fraction"] == "0.25"
    assert Path(options(calls[1])["--reference"]).read_bytes() == b"ordinary"
    assert Path(options(calls[2])["--reference"]).read_bytes() == b"frozen reference"
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "complete"

    # A completed invocation must not retrain or rerun the probes.
    runner.run(tmp_path, config)
    assert len(calls) == 3
    checkpoint = Path(options(calls[1])["--candidate"])
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Missing or changed"):
        runner.run(tmp_path, config)


def test_resume_rejects_changed_preparation(
    runner: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alpha_go.boxes import policy_target_ab

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    runner.write_json(inputs / "collection-config.json", {})
    runner.write_json(inputs / "prepared-hashes.json", {"a.npz": "old"})
    config = dict(cwd=str(runner.REPO), code_sha256={}, input_sha256={},
                  cpu=True, runtime={"binary": "same"}, prepared=str(tmp_path / "prepared"))
    runner.write_json(tmp_path / "config.json", config)
    (tmp_path / "config.sha256").write_text(runner.digest(tmp_path / "config.json"))
    monkeypatch.setattr(runner, "runtime_identity", lambda *_: {"binary": "same"})
    monkeypatch.setattr(policy_target_ab, "bundle_files", lambda _: {"a.npz": "changed"})
    with pytest.raises(ValueError, match="Prepared training data changed"):
        runner.verify(tmp_path, config)
