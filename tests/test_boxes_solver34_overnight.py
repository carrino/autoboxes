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


def test_real_cpu_iteration_final_match_and_recovery(tmp_path: Path, monkeypatch) -> None:
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
    output, config = launcher.create(args)
    settings = config["environment"]
    settings.update(SP_GAMES="4", SP_PROCS="2", SP_WORKERS="2", SP_SIMS="2",
                    SOLVER_N="0", ARENA_SOLVER_N="0", TRAIN_EPOCHS="1",
                    TRAIN_ARGS="--lr 0.0001 --batch-size 16", ARENA_GAMES="2",
                    ARENA_SIMS="2", POLICY_NODE_BUDGET="64", POLICY_CHILD_BUDGET="64")
    config["final_workers"] = 2
    launcher.write_json(output / "config.json", config)
    (output / "config.sha256").write_text(launcher.digest(output / "config.json") + "\n")
    launcher.run(output, config)
    state = json.loads((output / "state.json").read_text())
    assert state["phase"] == "complete" and state["completed_iterations"] == 1
    assert state["active_seconds"] > args.hours * 3600
    summary = json.loads((output / "final.json").read_text())
    assert summary["games"] == 2 and summary["independent_replay_verified"]
    assert (output / "results.tgz").is_file()
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
