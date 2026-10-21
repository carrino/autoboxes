"""Guard incumbent identity, excluded replay and completed diagnostic recovery."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def launcher() -> Any:
    repo = Path(__file__).resolve().parents[1]
    path = repo / "experiments/2026-10-21_13-32-boxes-frontier37/pilot.py"
    spec = importlib.util.spec_from_file_location("test_frontier37_launcher", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_freezes_incumbent_monitors_and_extends_only_teaching_band(
    launcher: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = launcher.REPO
    reference = tmp_path / launcher.BALANCED / "iter24.pt"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"incumbent")
    source = reference.with_name("collection-config.json")
    source.write_text('{}\n')
    for name in launcher.helpers.INPUT_HASHES:
        dest = tmp_path / launcher.helpers.INPUTS / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / launcher.helpers.INPUTS / name, dest)
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    monkeypatch.setattr(launcher, "REFERENCE_SHA", launcher.digest(reference))
    monkeypatch.setattr(launcher, "SOURCE_SHA", launcher.digest(source))
    monkeypatch.setattr(launcher.balanced, "runtime_identity", lambda *_: {"binary": "same"})
    # The published helper lacks the local legacy-recovery normalizer. These
    # pinned monitor inputs already use the native rows/cols/family schema.
    monkeypatch.delattr(launcher.stages, "normalize_exclusions", raising=False)
    output = tmp_path / "run"
    output.mkdir()

    def create(args: Any, **options: Any) -> tuple[Path, dict[str, Any]]:
        assert args.reference == reference
        assert options["final_seed"] == 20261103
        assert options["reference_label"] == "proof34 iter24"
        assert options["resume_script"] == launcher.SCRIPT
        return output, dict(environment=launcher.runner.recipe(), code_sha256={},
                            reference=str(reference), reference_sha256=launcher.REFERENCE_SHA)

    monkeypatch.setattr(launcher.runner, "create", create)
    _, config = launcher.create(argparse.Namespace(reference=None, cpu=True))
    env = config["environment"]
    assert env["POLICY_MIN_UNDRAWN"] == "35" and env["POLICY_MAX_UNDRAWN"] == "37"
    assert env["POLICY_MAX_PER_GAME"] == "3"
    assert env["SOLVER_N"] == env["ARENA_SOLVER_N"] == "34"
    assert env["GATE"] == "0" and env["EXTRA_DATA"] == ""
    assert config["finish_started_iteration"] and config["final_workers"] == 2
    roots = [json.loads(x) for x in (output / "inputs/positions-37.jsonl").read_text().splitlines()]
    assert roots and all(len(r["prefix"]) == 23 for r in roots)
    excluded = [json.loads(x) for x in
                Path(env["POLICY_EXCLUDE_POSITIONS"]).read_text().splitlines()]
    assert len(excluded) > len(roots)
    assert all(r["rows"] == r["cols"] == 5 for r in excluded)
    assert (output / "config.sha256").read_text().strip() == launcher.digest(output / "config.json")


def test_wrong_checkpoint_rejected_before_runtime_or_output(
    launcher: Any, tmp_path: Path,
) -> None:
    path = tmp_path / "balanced.pt"
    path.write_bytes(b"unpromoted balanced candidate")
    with pytest.raises(ValueError, match="Missing or changed verified input"):
        launcher.create(argparse.Namespace(reference=path, cpu=True))


def test_missing_policy_manifest_cannot_reintroduce_excluded_games(
    launcher: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    output = tmp_path / launcher.BASE / "logs/test-run"
    output.mkdir(parents=True)
    league = tmp_path / launcher.BASE / "league_state-test-run.json"
    league.write_text(json.dumps(dict(history=[dict(iteration=1)])))
    data = tmp_path / "data"
    folder = data / launcher.BASE / "test-run/policy-it0"
    folder.mkdir(parents=True)
    manifest = folder / "dataset.txt"
    manifest.write_text("games/game-0000000/exact\n")
    (folder / "preparation.json").write_text(json.dumps(dict(
        status="complete", dataset_sha256=launcher.digest(manifest),
        identity={"sources": []}, completed_receipts={})))
    config = dict(environment={"GAME_DATA_DIR": str(data), "WINDOW": "8"})
    launcher.verify_replay(output, config)
    manifest.unlink()
    with pytest.raises(ValueError, match="Missing or changed"):
        launcher.verify_replay(output, config)


def test_completed_diagnostics_are_reused_and_failure_does_not_lose_final(
    launcher: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alpha_go.boxes import probe_report

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    launcher.write_json(inputs / "labels.json", dict(schema=1, rows=5, cols=5, remaining_values={}))
    (tmp_path / "state.json").write_text(json.dumps(dict(phase="complete", completed_iterations=3)))
    config = dict(checkpoint_dir=str(tmp_path), reference="iter24.pt", cpu=True,
                  diagnostic_caps_seconds={"35-36": 7200, "37": 7200})
    calls = []

    def job(command: list[str], log: Path, seconds: int) -> None:
        calls.append(command)
        assert seconds == 7200 and command[command.index("--solver")+1] == "34"
        assert command[command.index("--candidate")+1] == str(tmp_path / "iter3.pt")
        assert command[command.index("--reference")+1] == "iter24.pt"
        band = command[command.index("--min-undrawn")+1]
        assert command[command.index("--max-undrawn")+1] == ("36" if band == "35" else "37")
        dest = Path(command[command.index("--output")+1])
        dest.mkdir()
        launcher.write_json(dest / "labels.json",
                            dict(schema=1, rows=5, cols=5, remaining_values={}))
        log.write_text("done")

    monkeypatch.setattr(launcher.helpers, "job", job)
    monkeypatch.setattr(probe_report, "build_report", lambda *a, **kw: {})
    monkeypatch.setattr(probe_report, "write_report", lambda *a, **kw: None)
    launcher.diagnostics(tmp_path, config)
    launcher.diagnostics(tmp_path, config)
    assert len(calls) == 2
    assert json.loads((tmp_path / "diagnostics.json").read_text())["complete"]
    (tmp_path / "diagnostic-37/completed.json").unlink()

    def timeout(*_args: Any, **_kwargs: Any) -> None:
        raise TimeoutError("bounded diagnostic")
    monkeypatch.setattr(launcher.helpers, "job", timeout)
    launcher.diagnostics(tmp_path, config)
    status = json.loads((tmp_path / "diagnostics.json").read_text())
    assert status["bands"]["35-36"]["complete"] and not status["bands"]["37"]["complete"]
    assert json.loads((tmp_path / "state.json").read_text())["phase"] == "complete"


def test_runtime_change_rejected_on_resume(
    launcher: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    reference = tmp_path / "iter24.pt"
    reference.write_bytes(b"reference")
    monkeypatch.setattr(launcher, "REFERENCE_SHA", launcher.digest(reference))
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/collection-config.json").write_text('{}')
    config = dict(cwd=str(tmp_path), reference=str(reference),
                  reference_sha256=launcher.REFERENCE_SHA, input_sha256={},
                  code_sha256={}, cpu=True, runtime={"old": True})
    launcher.write_json(tmp_path / "config.json", config)
    (tmp_path / "config.sha256").write_text(launcher.digest(tmp_path / "config.json"))
    monkeypatch.setattr(launcher.balanced, "runtime_identity", lambda *_: {"old": False})
    with pytest.raises(ValueError, match="Runtime changed"):
        launcher.verify(tmp_path, config)


def test_label_coverage_distinguishes_incomplete_and_mover_perspective(
    launcher: Any, tmp_path: Path,
) -> None:
    import csv

    output = tmp_path / "run"
    output.mkdir()
    folder = tmp_path / launcher.BASE / "run/policy-it0"
    bundle = folder / "games/game-0000000"
    bundle.mkdir(parents=True)
    (folder / "preparation.json").write_text(json.dumps(dict(status="complete", counts={})))
    with (bundle / "roots.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["edges", "mover", "exact_margin_p1", "regret", "complete"])
        writer.writerow([(1 << 23)-1, 1, -3, 4, True])
        writer.writerow([(1 << 23)-1, 0, 1, 2, False])
        writer.writerow([(1 << 24)-1, 0, -1, 2, True])
    launcher.label_summary(output, dict(environment={"GAME_DATA_DIR": str(tmp_path)}))
    bands = json.loads((output / "label-coverage.json").read_text())["bands"]
    assert bands["37"] == dict(attempted=2, complete=1, win_to_loss=2, complete_win_to_loss=1)
    assert bands["36"] == dict(attempted=1, complete=1, win_to_loss=0, complete_win_to_loss=0)
