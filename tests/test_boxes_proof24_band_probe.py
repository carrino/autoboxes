"""The bounded endpoint diagnostic cannot silently change models or lose partial work."""
from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def launcher():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "experiments/2026-10-18_10-40-boxes-proof24-result/band_probe.py"
    spec = importlib.util.spec_from_file_location("proof24_band_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrong_candidate_stops_before_runtime_or_output(launcher, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    monkeypatch.setattr(launcher, "SOURCE_CONFIG_SHA", launcher.digest(source / "config.json"))
    reference, candidate = tmp_path / "reference.pt", tmp_path / "candidate.pt"
    reference.write_bytes(b"real reference")
    candidate.write_bytes(b"old iter22, not iter24")
    monkeypatch.setattr(launcher, "CHECKPOINTS", {
        "reference": launcher.digest(reference), "candidate": "expected iter24 digest"})
    monkeypatch.setattr(launcher, "runtime_identity", lambda *_: pytest.fail("Runtime reached"))
    args = SimpleNamespace(source_run=source, reference=reference, candidate=candidate)
    with pytest.raises(ValueError, match="candidate.pt"):
        launcher.create(args)


def test_exact_cache_reuses_successes_not_timeouts_and_rejects_conflicts(launcher, tmp_path):
    (tmp_path / "inputs").mkdir()
    prior = tmp_path / "35-36/attempt-old/probe"
    prior.mkdir(parents=True)
    base = dict(schema=1, rows=5, cols=5, remaining_values={"10": 1}, unknown_masks=[20])
    launcher.write_json(tmp_path / "inputs/labels.json", base)
    extra = dict(base, remaining_values={"20": -3}, unknown_masks=[30])
    launcher.write_json(prior / "labels.json", extra)
    launcher.merged_cache(tmp_path, tmp_path / "merged.json")
    merged = json.loads((tmp_path / "merged.json").read_text())
    assert merged["remaining_values"] == {"10": 1, "20": -3}
    assert "30" not in merged["remaining_values"]
    launcher.write_json(prior / "labels.json", dict(base, remaining_values={"10": -1}))
    with pytest.raises(ValueError, match="Conflicting"):
        launcher.merged_cache(tmp_path, tmp_path / "merged.json")


def test_timeout_packages_partial_results_and_continues_other_band(launcher, tmp_path, monkeypatch):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/candidate.pt").write_bytes(b"large weights excluded from export")
    seen = []

    def stage(output, config, minimum, maximum):
        seen.append((minimum, maximum))
        attempt = output / f"{minimum}-{maximum}" / "attempt-test"
        attempt.mkdir(parents=True)
        (attempt / "partial.json").write_text('{"saved": true}')
        if minimum == 35:
            raise TimeoutError("finite band budget exhausted")
        return {"attempt": str(attempt), "band_report": {"common_known": 2}}

    monkeypatch.setattr(launcher, "stage", stage)
    assert launcher.run(tmp_path, {"scope": "test diagnostic"}) == 1
    assert seen == [(35, 36), (37, 38)]
    state = json.loads((tmp_path / "state.json").read_text())
    assert not state["complete"] and not state["bands"]["35..36"]["complete"]
    assert state["bands"]["37..38"]["complete"]
    with tarfile.open(tmp_path / "results.tgz") as archive:
        assert "35-36/attempt-test/partial.json" in archive.getnames()
        assert "37-38/attempt-test/partial.json" in archive.getnames()
        assert "inputs/candidate.pt" not in archive.getnames()


def test_completed_stage_survives_resume_and_detects_modified_artifact(
        launcher, tmp_path, monkeypatch):
    folder = tmp_path / "35-36"
    folder.mkdir()
    result = folder / "original-report.json"
    result.write_text('{"metric": 1}')
    receipt = {"artifacts_sha256": {"35-36/original-report.json": launcher.digest(result)}}
    launcher.write_json(folder / "completed.json", receipt)
    monkeypatch.setattr(launcher, "job", lambda *_: pytest.fail("Completed band reran"))
    assert launcher.stage(tmp_path, {}, 35, 36) == receipt
    result.write_text('{"metric": 2}')
    with pytest.raises(ValueError, match="changed"):
        launcher.stage(tmp_path, {}, 35, 36)


def test_finite_job_stops_real_child_and_preserves_log(launcher, tmp_path):
    log = tmp_path / "run.log"
    command = [sys.executable, "-u", "-c",
               "import time; print('saved partial progress', flush=True); time.sleep(30)"]
    with pytest.raises(TimeoutError, match="Band exceeded"):
        launcher.job(command, log, seconds=.25, heartbeat=.1)
    assert "saved partial progress" in log.read_text()


def test_fixed_probe_command_matches_arena_and_uses_frozen_roots(launcher, tmp_path):
    command = launcher.probe_command(tmp_path, tmp_path / "attempt", 35, 36, cpu=True)
    assert command[command.index("--solver") + 1] == "34"
    assert command[command.index("--solver-budget") + 1] == "50000"
    assert command[command.index("--simulations") + 1] == "400"
    assert command[command.index("--mixes") + 1:command.index("--simulations")] == [
        "reference", "candidate"]
    assert command[command.index("--label-budget") + 1] == "20000000"
    assert command[command.index("--positions-file") + 1].endswith("positions-35-36.jsonl")
    assert "--games" not in command and "--cpu" in command


def test_stale_extension_rejected_before_any_search(launcher, tmp_path, monkeypatch):
    import torch

    binary = tmp_path / "stale.so"
    binary.write_bytes(b"wrong build")
    real_import = launcher.importlib.import_module

    def load(name):
        if name == "alpha_go_cpp":
            return SimpleNamespace(__file__=str(binary))
        return real_import(name)

    monkeypatch.setattr(launcher.importlib, "import_module", load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    source = {"preflight": {"runtime": {"extension_sha256": "expected compiled hash"}}}
    with pytest.raises(ValueError, match="stale.so"):
        launcher.runtime_identity(source, cpu=False)
