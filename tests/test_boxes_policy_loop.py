"""Exercise the actual collect -> exact policies -> train -> arena shell loop on CPU."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import save_boxes_net


def test_loop_restart_reuses_collection_and_trains_labelled_window(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "experiments/2026-10-08_14-42-boxes-3x3-loop"
    experiment = tmp_path / "experiments/loop"
    experiment.mkdir(parents=True)
    for path in [*source.glob("*.py"), source / "run_iteration_local.sh"]:
        shutil.copyfile(path, experiment / path.name)
    checkpoint = experiment / "checkpoints/tiny/iter0.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.manual_seed(42)
    save_boxes_net(BoxesNet(2, 2, channels=8, n_blocks=1, value_hidden=8), checkpoint)
    (experiment / "league_state-tiny.json").write_text('{"champion": 0, "history": []}')
    # Reuse the test interpreter and installed dependencies without invoking uv sync.
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(f"#!{sys.executable}\n" + '''import os, sys
from pathlib import Path
args = sys.argv[1:]
assert args.pop(0) == "run"
if args[0] == "python": args.pop(0)
marker = Path(os.environ["FAIL_POLICY_ONCE"])
if "alpha_go.boxes.policy_loop_data" in args and marker.exists():
    marker.unlink()
    sys.exit(17)
os.execv(sys.executable, [sys.executable, *args])
''')
    uv.chmod(0o755)
    stop = tmp_path / "stop-once"
    stop.touch()
    data = tmp_path / "data"
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}",
               PYTHONPATH=f"{repo / 'src'}:{os.environ.get('PYTHONPATH', '')}",
               FAIL_POLICY_ONCE=str(stop), GAME_DATA_DIR=str(data),
               ROWS="2", COLS="2", TAG="tiny", POLICY_LABELS="1", GATE="0", RELABEL_N="0",
               POLICY_MIN_UNDRAWN="1", POLICY_MAX_UNDRAWN="12", POLICY_WORKERS="2",
               POLICY_NODE_BUDGET="1000000", POLICY_TIMEOUT="1m", SP_GAMES="8", SP_SIMS="8",
               SP_PROCS="2", SP_WORKERS="2", TRAIN_BUDGET="0", TRAIN_EPOCHS="1",
               TRAIN_ARGS="--batch-size 16", TRAIN_MIN_UNDRAWN="0", FEATURES="basic",
               ARENA_GAMES="2", BASE_GAMES="2", ARENA_SIMS="4", BASELINES="boxes-greedy",
               SOLVER_N="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = ["bash", str(experiment / "run_iteration_local.sh"), "0", "0", "--cpu"]
    first = subprocess.run(command, cwd=tmp_path, env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert first.returncode == 17, first.stdout
    collection = data / "experiments/loop/tiny/selfplay-it0"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in collection.glob("*.npz")}
    assert len(hashes) == 8
    resumed = [*command[:-2], "1", "--cpu"]  # resume iteration 0, then play/train iteration 1
    second = subprocess.run(resumed, cwd=tmp_path, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert second.returncode == 0, second.stdout
    assert "Reusing completed self-play" in second.stdout
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in collection.glob("*.npz")}
    prepared = data / "experiments/loop/tiny/policy-it0"
    metadata = json.loads((prepared / "preparation.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["counts"]["replaced_roots"] > 0
    manifest = (experiment / "dataset-tiny-it1.txt").read_text()
    directories = [line for line in manifest.splitlines() if line and not line.startswith("#")]
    assert directories and all("/policy-it0/" in path for path in directories)
    assert any(path.endswith("/children") for path in directories)
    assert all((data / path).is_dir() for path in directories)
    assert (experiment / "checkpoints/tiny/iter1.pt").is_file()
    assert (experiment / "timing/tiny/it1.json").is_file()
    train_log = (experiment / "logs/tiny/train-it1.log").read_text()
    result = json.loads(train_log.split("===RESULT===")[-1].strip())
    assert result["train_exact_policy"]["positions"] > 0
    assert 0 <= result["train_exact_policy"]["optimal_mass"] <= 1
    assert result["train_exact_policy"]["families"] > 0
    window = (experiment / "dataset-tiny-it2.txt").read_text()
    assert "/policy-it0/" in window and "/policy-it1/" in window
    assert "/selfplay-it" not in window
    next_collection = data / "experiments/loop/tiny/selfplay-it1/collection-cache.json"
    expected = hashlib.sha256((experiment / "checkpoints/tiny/iter1.pt").read_bytes()).hexdigest()
    assert json.loads(next_collection.read_text())["request"]["checkpoint_sha256"] == expected
    assert (experiment / "checkpoints/tiny/iter2.pt").is_file()
    # A league commit can survive while the following timing write is interrupted.
    timing = experiment / "timing/tiny/it2.json"
    timing.unlink()
    before = hashlib.sha256((experiment / "checkpoints/tiny/iter2.pt").read_bytes()).hexdigest()
    third = subprocess.run(resumed, cwd=tmp_path, env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    assert third.returncode == 0, third.stdout
    assert "already committed to the league" in third.stdout
    assert json.loads(timing.read_text())["recovered_from_league"]
    state = json.loads((experiment / "league_state-tiny.json").read_text())
    assert [entry["iteration"] for entry in state["history"]] == [1, 2]
    assert before == hashlib.sha256(
        (experiment / "checkpoints/tiny/iter2.pt").read_bytes()).hexdigest()


def test_terminated_loop_reaps_collectors(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "experiments/2026-10-08_14-42-boxes-3x3-loop"
    experiment = tmp_path / "loop"
    experiment.mkdir()
    for name in ["run_iteration_local.sh", "collection_cache.py", "arena_completed.py"]:
        shutil.copyfile(source / name, experiment / name)
    checkpoint = experiment / "checkpoints/tiny/iter0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"only hashing this checkpoint in the collection test")
    binary, pids = tmp_path / "bin", tmp_path / "pids"
    binary.mkdir()
    pids.mkdir()
    uv = binary / "uv"
    uv.write_text(f"#!{sys.executable}\n" + '''import os, sys, time
from pathlib import Path
args = sys.argv[1:]
assert args.pop(0) == "run"
if args[0].endswith("run_games.py"):
    (Path(os.environ["WORKER_PIDS"]) / str(os.getpid())).touch()
    time.sleep(60)
else:
    os.execv(sys.executable, [sys.executable, *args])
''')
    uv.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", WORKER_PIDS=str(pids),
               GAME_DATA_DIR=str(tmp_path / "data"), TAG="tiny", ROWS="2", COLS="2",
               POLICY_LABELS="1", GATE="0", SP_GAMES="2", SP_PROCS="2", SP_WORKERS="2")
    with subprocess.Popen(["bash", str(experiment / "run_iteration_local.sh"), "0", "0", "--cpu"],
                          cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT) as process:
        deadline = time.monotonic() + 15
        while (len(list(pids.iterdir())) < 2 and time.monotonic() < deadline
               and process.poll() is None):
            time.sleep(0.02)
        workers = [int(path.name) for path in pids.iterdir()]
        process.terminate()
        output, _ = process.communicate(timeout=10)
        assert len(workers) == 2, output
        assert process.returncode == 143, output
        assert all(not Path(f"/proc/{pid}").exists() for pid in workers), output


@pytest.mark.parametrize("final_games", [0, 200])
def test_fresh_wrapper_saves_settings_and_resumes_completed_run(
    tmp_path: Path, final_games: int,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    name = "experiments/2026-10-08_14-42-boxes-3x3-loop"
    experiment = tmp_path / name
    source = experiment / "logs/astra-probe-PCcE6W"
    (source / "mix").mkdir(parents=True)
    (source / "reference.pt").write_bytes(b"frozen-reference")
    (source / "mix/positions.jsonl").write_text("{}\n")
    shutil.copyfile(repo / name / "arena_completed.py", experiment / "arena_completed.py")
    # The real loop is exercised above. Here verify the generated shell wrapper itself.
    fake = experiment / "fake_loop.py"
    fake.write_text('''import hashlib, json, os, sys
from pathlib import Path
exp, tag = Path(os.environ["EXP"]), os.environ["TAG"]
state = {"champion": 0, "history": []}
for iteration in range(int(sys.argv[1]) + 1, int(sys.argv[2]) + 2):
    checkpoint = exp / "checkpoints" / tag / f"iter{iteration}.pt"
    checkpoint.write_bytes(f"candidate{iteration}".encode())
    state["history"].append({"iteration": iteration,
        "candidate_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    timing = exp / "timing" / tag / f"it{iteration}.json"
    timing.parent.mkdir(parents=True, exist_ok=True)
    timing.write_text(json.dumps({"iteration": iteration}))
(exp / f"league_state-{tag}.json").write_text(json.dumps(state))
(Path(os.environ["LOGS"]) / "invoked").write_text("once")
''')
    (experiment / "run_iteration_local.sh").write_text(
        f'#!/usr/bin/env bash\nexec "{sys.executable}" "{fake}" "$@"\n')
    git_dir = subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"],
                                     cwd=repo, text=True).strip()
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:5] == ["run", "python", "-u", "-m", "alpha_go.boxes.arena"]
def value(flag): return args[args.index(flag) + 1]
assert value("--a").endswith("/iter2.pt")
assert value("--b").endswith("/iter0.pt")
assert value("--num_games") == "200" and value("--opening_moves") == "4"
assert value("--seed") == "20261016" and value("--sims") == "400"
assert value("--solver") == "30" and value("--solver_budget") == "50000"
out = Path(value("--save_games"))
counter = out.parent / "final-invocations"
counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else "1")
fail = Path(os.environ["FAIL_FINAL_ONCE"])
if fail.exists():
    fail.unlink()
    raise SystemExit(9)
out.write_text("".join(json.dumps(dict(index=i)) + "\\n" for i in range(200)))
print("===RESULT===")
print(json.dumps(dict(num_games=200)))
''')
    uv.chmod(0o755)
    fail = tmp_path / "fail-final-once"
    fail.touch()
    env = dict(os.environ, GIT_DIR=git_dir, ITERATIONS="2", SP_GAMES="12",
               FINAL_GAMES=str(final_games), FAIL_FINAL_ONCE=str(fail),
               PATH=f"{binary}:{os.environ['PATH']}")
    wrapper = repo / "experiments/2026-10-14_17-34-boxes-policy-target-ab/run_loop.sh"
    first = subprocess.run(["bash", str(wrapper)], cwd=tmp_path, env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    assert first.returncode == (9 if final_games else 0), first.stdout
    logs = next((experiment / "logs").glob("5x5-policy-loop-*"))
    script = (logs / "resume.sh").read_text()
    assert "export SP_GAMES=12" in script and "export POLICY_LABELS=1" in script
    assert f"export FINAL_GAMES={final_games}" in script
    assert not (logs / "final-vs-iter0-complete.sha256").exists()
    invoked = (logs / "invoked").stat().st_mtime_ns
    second = subprocess.run(["bash", str(logs / "resume.sh")], cwd=tmp_path, env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    assert second.returncode == 0, second.stdout
    assert "All 2 iterations already completed" in second.stdout
    assert (logs / "invoked").stat().st_mtime_ns == invoked
    assert (logs / "results.tgz").is_file()
    assert (source / "reference.pt").read_bytes() == b"frozen-reference"
    if final_games:
        assert (logs / "final-invocations").read_text() == "2"
        assert (logs / "final-vs-iter0-complete.sha256").is_file()
        third = subprocess.run(["bash", str(logs / "resume.sh")], cwd=tmp_path, env=env,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=20)
        assert third.returncode == 0, third.stdout
        assert "Reusing completed final comparison" in third.stdout
        assert (logs / "final-invocations").read_text() == "2"
        assert (logs / "invoked").stat().st_mtime_ns == invoked
        (logs / "final-vs-iter0.jsonl").write_text("changed results")
        corrupted = subprocess.run(["bash", str(logs / "resume.sh")], cwd=tmp_path, env=env,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=20)
        assert corrupted.returncode != 0 and "FAILED" in corrupted.stdout
        assert (logs / "final-invocations").read_text() == "2"
