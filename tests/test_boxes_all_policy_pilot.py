"""The new pilot preserves its current incumbent and expensive completed stages."""
from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def pilot():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "experiments/2026-10-18_20-07-boxes-all-policy-pilot/pilot.py"
    spec = importlib.util.spec_from_file_location("all_policy_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reject_wrong_incumbent_before_runtime_or_collection(pilot, tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "REPO", tmp_path)
    source = tmp_path / pilot.helpers.SOURCE_RUN / "config.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    monkeypatch.setattr(pilot.helpers, "SOURCE_CONFIG_SHA", pilot.digest(source))
    checkpoint = tmp_path / pilot.BASE / "checkpoints/5x5-proof34-a6gdsd3i/iter24.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"wrong iter24")
    monkeypatch.setattr(pilot.helpers, "runtime_identity", lambda *_: pytest.fail("runtime"))
    with pytest.raises(ValueError, match="iter24.pt"):
        pilot.create(SimpleNamespace())
    assert not list((tmp_path / pilot.BASE / "logs").glob("all-policy-ab-*"))


def test_completed_report_resumes_without_rerunning_or_overwriting(pilot, tmp_path):
    calls = []

    def command(attempt):
        calls.append("run")
        return [sys.executable, "-c", "print('finished search')"]

    def report(attempt):
        calls.append("report")
        folder = attempt / "report"
        folder.mkdir()
        (folder / "report.json").write_text('{"done": true}')

    first = pilot.stage(tmp_path, "probe", 5, command, report)
    second = pilot.stage(tmp_path, "probe", 5, command, report)
    assert first == second and calls == ["run", "report"]
    (first / "report/report.json").write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        pilot.stage(tmp_path, "probe", 5, command, report)


def test_completed_stage_rejects_added_data(pilot, tmp_path):
    attempt = pilot.stage(tmp_path, "collect", 5, lambda _: [sys.executable, "-c", "pass"])
    (attempt / "unexpected.npz").write_bytes(b"new training data")
    with pytest.raises(ValueError, match="file set"):
        pilot.stage(tmp_path, "collect", 5, lambda _: pytest.fail("reran collection"))


def test_timeout_keeps_partial_outputs_and_no_completed_receipt(pilot, tmp_path):
    with pytest.raises(TimeoutError):
        pilot.stage(tmp_path, "train", .2, lambda _: [sys.executable, "-u", "-c",
                    "import time; print('partial training'); time.sleep(30)"])
    assert not (tmp_path / "train/completed.json").exists()
    log = next(tmp_path.glob("train/attempt-*/run.log"))
    assert "partial training" in log.read_text()
    pilot.package(tmp_path)
    with tarfile.open(tmp_path / "results.tgz") as archive:
        assert str(log.relative_to(tmp_path)) in archive.getnames()


def test_collection_and_search_match_current_recipe(pilot, tmp_path):
    commands = [pilot.collection_command(tmp_path, tmp_path / "attempt", 200, p)
                for p in range(4)]
    assert [c[c.index("--game_index_offset") + 1] for c in commands] == ["0", "50", "100", "150"]
    for command in commands:
        for key, value in {"--num_games": "50", "--num_workers": "2",
                           "--num_simulations": "400", "--solver_max_undrawn": "34",
                           "--solver_node_budget": "50000", "--c_puct": "1.5",
                           "--leaf_batch_size": "16"}.items():
            assert command[command.index(key) + 1] == value
        assert command[command.index("--checkpoint") + 1].endswith("inputs/iter24.pt")
        assert "--stop-when-solved" in command
        assert "--start-from" not in command
    command = pilot.probe_command(tmp_path, tmp_path / "attempt", Path("control.pt"),
                                  Path("exact.pt"), Path("labels.json"))
    for key, value in {"--reference": "control.pt", "--candidate": "exact.pt",
                       "--solver": "34", "--solver-budget": "50000",
                       "--simulations": "400", "--min-undrawn": "35",
                       "--max-undrawn": "38"}.items():
        assert command[command.index(key) + 1] == value
    assert sum(pilot.CAPS.values()) == 4 * 3600 + 45 * 60


def test_merge_cache_does_not_promote_unknowns_and_rejects_conflicts(pilot, tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    pilot.write_json(first, dict(schema=1, rows=5, cols=5,
                                remaining_values={"10": 1}, unknown_masks=[20]))
    pilot.write_json(second, dict(schema=1, rows=5, cols=5, remaining_values={"20": -3}))
    merged = tmp_path / "merged.json"
    pilot.merge_caches([first, second], merged)
    assert json.loads(merged.read_text())["remaining_values"] == {"10": 1, "20": -3}
    pilot.write_json(second, dict(schema=1, rows=5, cols=5, remaining_values={"10": -1}))
    with pytest.raises(ValueError, match="Conflicting"):
        pilot.merge_caches([first, second], merged)


def test_archive_excludes_large_weights_games_and_old_archive(pilot, tmp_path):
    (tmp_path / "weights.pt").write_bytes(b"weights")
    (tmp_path / "game.npz").write_bytes(b"game")
    (tmp_path / "summary.json").write_text("{}")
    pilot.package(tmp_path)
    pilot.package(tmp_path)
    with tarfile.open(tmp_path / "results.tgz") as archive:
        assert archive.getnames() == ["summary.json"]
