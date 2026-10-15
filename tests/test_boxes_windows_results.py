"""Native-result imports reject mismatched states and retain incomplete runs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_go.boxes.external_probe import TABLES, digest, read_roots

EXPERIMENT = (Path(__file__).resolve().parents[1] / "experiments" /
              "2026-10-14_19-52-boxes-external-probe")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / "windows" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scorer = load_script("score_results")


@pytest.fixture
def native_results(tmp_path):
    """Synthetic receipts for the three observed optimal smoke choices, no engine claim."""
    tables = EXPERIMENT / "data"
    positions = [json.loads(line) for line in (tables / TABLES[0]).read_text().splitlines()]
    selected = [positions[index] for index in (1, 3, 6)]
    shutil.copyfile(tables / TABLES[0], tmp_path / "all-positions.jsonl")
    (tmp_path / "positions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in selected))
    runtime = dict(input_sha256={"positions.jsonl": digest(tables / TABLES[0])},
                   selected_positions_sha256=digest(tmp_path / "positions.jsonl"),
                   selected_position_ids=[1, 3, 6], requested_seconds=5)
    (tmp_path / "runtime-provenance.json").write_text(json.dumps(runtime), encoding="utf-8-sig")
    roots = {r.index: r for r in read_roots(tables)}
    for name, engine_hash in scorer.ENGINE_HASHES.items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "exit-status.txt").write_text("0\n")
        (directory / "engine-provenance.json").write_text(
            json.dumps(dict(engine_sha256=engine_hash)))
        for position, edge in zip(selected, (45, 37, 55)):
            board = roots[position["position"]].board()
            response = dict(position=position["position"], edge=edge, requested_seconds=5,
                            native_root_roundtrip_verified=True, mode="normal-gui-computer-move",
                            executable_sha256=engine_hash, input_to_play=board.to_play(),
                            input_scores=list(board.boxes))
            root = directory / "roots" / f"root-{position['position']:03}"
            root.mkdir(parents=True)
            (root / "result.json").write_text(json.dumps(dict(
                request=dict(position, seconds=5), response=response, wall_seconds=2.0)))
    return tmp_path, tables


def test_native_scores_match_known_smoke_moves_and_paired_reference(native_results):
    results, tables = native_results
    report = scorer.score_results(results, tables)
    assert report["complete"]
    for engine in report["engines"].values():
        assert engine["aggregate"]["optimal_rate"] == 1
        assert engine["aggregate"]["mean_margin_regret"] == 0
        assert engine["aggregate"]["wins_preserved"] == 3
        comparison = engine["comparisons"]["reference"]["overall"]
        assert comparison["common_positions"] == 3
        assert comparison["metrics"]["margin_regret"]["delta"] == pytest.approx(-8 / 3)


def test_failed_root_keeps_prior_results_and_does_not_count_as_loss(native_results):
    results, tables = native_results
    root = results / "dabble/roots/root-006"
    (root / "result.json").rename(root / "result.json.tmp")
    (results / "dabble/exit-status.txt").write_text("1")
    report = scorer.score_results(results, tables)
    assert not report["complete"] and report["engines"]["prsboxes"]["complete"]
    dabble = report["engines"]["dabble"]
    assert dabble["missing_positions"] == [6]
    assert dabble["aggregate"]["positions"] == 2
    assert dabble["aggregate"]["wins_preserved"] == 2
    assert dabble["comparisons"]["reference"]["overall"]["common_positions"] == 2


@pytest.mark.parametrize("change", ["prefix", "edge", "score", "budget", "hash", "duplicate"])
def test_rejects_invalid_native_receipts(native_results, change):
    results, tables = native_results
    path = results / "dabble/roots/root-003/result.json"
    saved = json.loads(path.read_text())
    if change == "prefix":
        saved["request"]["prefix"].reverse()
    elif change == "edge":
        saved["response"]["edge"] = saved["request"]["prefix"][0]
    elif change == "score":
        saved["response"]["input_scores"] = [0, 0]
    elif change == "budget":
        saved["response"]["requested_seconds"] = 10
    elif change == "hash":
        saved["response"]["executable_sha256"] = "bad"
    else:
        path = results / "dabble/roots/root-099/result.json"
        path.parent.mkdir()
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        scorer.score_results(results, tables)


def test_prs_worker_saves_completed_root_before_later_failure(tmp_path, monkeypatch):
    worker = load_script("prsboxes_windows_smoke")
    engine = tmp_path / "engine.exe"
    engine.write_bytes(b"test executable")
    monkeypatch.setattr(worker, "ENGINE_SHA256", hashlib.sha256(engine.read_bytes()).hexdigest())
    monkeypatch.setattr(worker, "os", SimpleNamespace(name="nt"))
    positions = tmp_path / "positions.jsonl"
    positions.write_text('{"position": 1}\n{"position": 2}\n')
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["worker", "--exe", str(engine), "--positions", str(positions),
                                     "--output", str(output), "--limit", "100"])

    def decide(engine, position, root, seconds, seed, watchdog):
        if position["position"] == 2:
            raise RuntimeError("simulated native failure")
        return dict(position=1, edge=45)

    monkeypatch.setattr(worker, "run_position", decide)
    with pytest.raises(RuntimeError, match="simulated native failure"):
        worker.main()
    saved = json.loads((output / "root-001/result.json").read_text())
    assert saved["response"]["edge"] == 45
    assert saved["request"]["seconds"] == 5
    assert not (output / "root-002/result.json").exists()
    assert "simulated native failure" in (output / "root-002/failure.txt").read_text()


def test_dabble_full_mode_does_not_silently_keep_three_root_limit(tmp_path, monkeypatch):
    worker = load_script("dabble_windows_smoke")
    positions = tmp_path / "positions.jsonl"
    positions.write_text("".join(json.dumps(dict(position=i)) + "\n" for i in range(100)))
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["worker", "--exe", "unused.exe",
                                     "--positions", str(positions), "--output", str(output),
                                     "--limit", "100"])

    def decide(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, '{"edge": 0}', "")

    monkeypatch.setattr(worker.subprocess, "run", decide)
    worker.main()
    receipts = list(output.glob("root-*/result.json"))
    assert len(receipts) == 100
    assert {json.loads(p.read_text())["response"]["position"] for p in receipts} == set(range(100))
