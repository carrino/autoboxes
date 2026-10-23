"""Exercise the actual timed launcher, final replay, and completed-run recovery."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import save_boxes_net


def load_launcher():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "experiments/2026-10-16_21-41-boxes-solver34-loop/overnight.py"
    spec = importlib.util.spec_from_file_location("solver34_overnight", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("proof_audit", [False, True])
def test_real_cpu_iteration_final_match_and_recovery(
        tmp_path: Path, monkeypatch, proof_audit: bool) -> None:
    repo = Path(__file__).resolve().parents[1]
    launcher = load_launcher()
    experiment = tmp_path / launcher.EXP
    experiment.mkdir(parents=True)
    for path in [*(repo / launcher.EXP).glob("*.py"),
                 repo / launcher.EXP / "run_iteration_local.sh"]:
        shutil.copyfile(path, experiment / path.name)
    reference = tmp_path / "reference.pt"
    torch.manual_seed(20261028)
    save_boxes_net(BoxesNet(5, 5, channels=8, n_blocks=1, value_hidden=8,
                           features="chains"), reference)
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(f"#!{sys.executable}\n" + '''import os, sys
args = sys.argv[1:]
assert args.pop(0) == 'run'
if args[0] == 'python': args.pop(0)
os.execv(sys.executable, [sys.executable, *args])
''')
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}:{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", f"{repo / 'src'}:{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.setenv("GAME_DATA_DIR", str(tmp_path / "data"))
    git_dir = subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"],
                                     cwd=repo, text=True).strip()
    monkeypatch.setenv("GIT_DIR", git_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "source_code", lambda: {})
    args = argparse.Namespace(reference=reference, hours=0.00001, final_games=2, cpu=True)
    resume_script = repo / "experiments/2026-10-17_14-14-boxes-proof-overnight/overnight.py"
    options = dict(tag_prefix="5x5-proof34-", final_seed=20261029,
                   resume_script=resume_script,
                   reference_label="solver34 iter15") if proof_audit else {}
    output, config = launcher.create(args, **options)
    if proof_audit:
        assert str(resume_script) in (output / "resume.sh").read_text()
        assert config["final_seed"] == 20261029
    settings = config["environment"]
    settings.update(SP_GAMES="4", SP_PROCS="2", SP_WORKERS="2", SP_SIMS="2",
                    SOLVER_N="0", ARENA_SOLVER_N="0", TRAIN_EPOCHS="1",
                    TRAIN_ARGS="--lr 0.0001 --batch-size 16", ARENA_GAMES="2",
                    ARENA_SIMS="2", POLICY_NODE_BUDGET="64", POLICY_CHILD_BUDGET="64")
    if proof_audit:
        settings["PROOF_TARGET_AUDIT"] = "1"
    config["final_workers"] = 2
    launcher.write_json(output / "config.json", config)
    (output / "config.sha256").write_text(launcher.digest(output / "config.json") + "\n")
    launcher.run(output, config)
    state = json.loads((output / "state.json").read_text())
    assert state["phase"] == "complete" and state["completed_iterations"] == 1
    assert state["active_seconds"] > args.hours * 3600
    summary = json.loads((output / "final.json").read_text())
    assert summary["games"] == 2 and summary["independent_replay_verified"]
    assert summary["reference_sha256"] == config["reference_sha256"]
    assert summary["reference_label"] == ("solver34 iter15" if proof_audit else "iter0")
    assert (output / "results.tgz").is_file()
    if proof_audit:
        audit = json.loads((output / "targets-it0.json").read_text())
        assert audit["games"] == 4
    train = (output / "train-it1.log").read_text()
    result = json.loads(train.split("===RESULT===")[-1].strip())
    assert result["resumed_from"] == config["reference"]
    assert result["steps_completed"] > 0
    before = {p: p.stat().st_mtime_ns for p in (
        output / "final-arena.log", output / "train-it1.log", output / "final-games.jsonl")}
    launcher.run(output, config)
    assert before == {p: p.stat().st_mtime_ns for p in before}
    # Recover the committed iteration if the controller's state write was interrupted.
    state.update(phase="training", completed_iterations=0, checkpoints={})
    launcher.write_json(output / "state.json", state)
    # Final match will rerun; training must not. That is safe recovery, not a second update.
    launcher.run(output, config)
    assert (output / "train-it1.log").stat().st_mtime_ns == before[output / "train-it1.log"]
    assert len(json.loads((output / "league.json").read_text())["history"]) == 1
    (output / "final-games.jsonl").write_text("corrupted")
    with pytest.raises(ValueError, match="changed"):
        launcher.run(output, config)


def test_night_recipe_and_final_match_are_matched() -> None:
    launcher = load_launcher()
    settings = launcher.recipe()
    assert settings["GATE"] == "0"
    assert settings["SOLVER_N"] == settings["ARENA_SOLVER_N"] == "34"
    assert settings["SOLVER_BUDGET"] == "50000"
    assert settings["POLICY_MIN_UNDRAWN"] == "35" and settings["POLICY_MAX_UNDRAWN"] == "36"
    config = dict(environment=settings, reference="reference.pt", final_games=200,
                  final_workers=4, final_seed=20261028, cpu=False)
    command = launcher.final_command(config, Path("last.pt"), Path("games.jsonl"))
    assert command[command.index("--a") + 1] == "ckpt:last.pt"
    assert command[command.index("--b") + 1] == "ckpt:reference.pt"
    assert command[command.index("--solver") + 1] == "34"
    assert command[command.index("--solver_budget") + 1] == "50000"
    assert command[command.index("--num_games") + 1] == "200"


@pytest.mark.parametrize("finish_started,started,last_iteration", [
    (True, True, 2), (True, False, 1), (False, True, 1),
])
def test_resume_finishes_started_iteration_only_when_opted_in(
        tmp_path: Path, monkeypatch, finish_started: bool, started: bool,
        last_iteration: int) -> None:
    launcher = load_launcher()
    experiment = tmp_path / "experiment"
    output = experiment / "logs" / "resume-test"
    output.mkdir(parents=True)
    checkpoints = experiment / "checkpoints"
    checkpoints.mkdir()
    reference, first = checkpoints / "iter0.pt", checkpoints / "iter1.pt"
    reference.write_bytes(b"frozen reference")
    first.write_bytes(b"completed first iteration")
    league_path = experiment / f"league_state-{output.name}.json"
    league = dict(champion=0, history=[dict(
        iteration=1, candidate_checkpoint_sha256=launcher.digest(first))])
    launcher.write_json(league_path, league)
    state_path = output / "state.json"
    launcher.write_json(state_path, dict(phase="training", completed_iterations=1,
        active_seconds=4 * 3600 + 1, checkpoints={str(first): launcher.digest(first)}))
    if started:
        (output / "iteration-2.log").write_text("interrupted policy preparation\n")
    config = dict(cwd=str(tmp_path), hours=4, cpu=True, environment=launcher.recipe(),
                  checkpoint_dir=str(checkpoints), reference=str(reference),
                  reference_sha256=launcher.digest(reference), code_sha256={},
                  final_games=2, final_workers=1, final_seed=20261103)
    if finish_started:
        config["finish_started_iteration"] = True
    launcher.write_json(output / "config.json", config)
    (output / "config.sha256").write_text(launcher.digest(output / "config.json") + "\n")
    monkeypatch.setattr(launcher, "EXP", experiment)
    monkeypatch.setattr(launcher, "package_results", lambda _output: None)
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_job(command, log, env, state, saved_state, count_work):
        calls.append(count_work)
        if count_work:
            assert command == ["bash", str(experiment / "run_iteration_local.sh"),
                               "1", "1", "--cpu"]
            assert log == output / "iteration-2.log"
            assert state["active_seconds"] > 4 * 3600
            candidate = checkpoints / "iter2.pt"
            candidate.write_bytes(b"completed resumed iteration")
            league["history"].append(dict(
                iteration=2, candidate_checkpoint_sha256=launcher.digest(candidate)))
            launcher.write_json(league_path, league)
            state["active_seconds"] += 10
        else:
            assert command[command.index("--a") + 1] == (
                f"ckpt:{checkpoints / f'iter{last_iteration}.pt'}")
            (output / "final-games.jsonl").write_text("mock verified games\n")
            log.write_text('===RESULT===\n{"games": 2}\n')
        launcher.write_json(saved_state, state)

    def verify_match(games, summary, expected_games):
        assert games == output / "final-games.jsonl"
        assert summary == {"games": 2} and expected_games == 2

    monkeypatch.setattr(launcher, "job", fake_job)
    monkeypatch.setattr(launcher, "verify_match", verify_match)
    launcher.run(output, config)
    saved = json.loads(state_path.read_text())
    assert saved["phase"] == "complete" and saved["completed_iterations"] == last_iteration
    assert calls == ([True, False] if last_iteration == 2 else [False])
    assert len(json.loads((output / "league.json").read_text())["history"]) == last_iteration
    assert saved["checkpoints"][str(checkpoints / f"iter{last_iteration}.pt")] == (
        launcher.digest(checkpoints / f"iter{last_iteration}.pt"))
    before = calls.copy()
    launcher.run(output, config)
    assert calls == before
