"""Eight hours of continuing solver-assisted self-play, then a matched final arena."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from alpha_go.boxes.curriculum import (
    checked_files,
    digest,
    package_results,
    source_code,
    verify_match,
    write_json,
)

EXP = Path("experiments/2026-10-08_14-42-boxes-3x3-loop")
SCRIPT = Path(__file__).resolve()
ITER0_SHA = "46c7264c4a0e9c7efa1cfccbb1b511c1c5faa3824fb69a62705d8cbfb1f953f0"
SEARCH = ("--c_puct 1.5 --leaf_batch_size 16 --policy_temperature 1.0 "
          "--prove_terminals 1 --shape_terminals 1 "
          "--margin_utility_lambda 0.5 --margin_utility_k 6")


def recipe() -> dict[str, str]:
    """Use fresh games, the latest learner, and natural-weight exact mistake targets."""
    return dict(
        ROWS="5", COLS="5", FEATURES="chains", GATE="0", RELABEL_N="0",
        SP_GAMES="100", SP_SIMS="400", SP_PROCS="4", SP_WORKERS="8",
        SOLVER_N="34", SOLVER_BUDGET="50000", ARENA_SOLVER_N="34", MERGE_EQ="0",
        TRAIN_BUDGET="0", TRAIN_EPOCHS="4", TRAIN_MIN_UNDRAWN="24",
        TRAIN_ARGS="--lr 0.0001", WINDOW="8", EXTRA_DATA="", START_FROM="",
        START_FRACTION="0", START_UNDRAWN="36 40", STOP_WHEN_SOLVED="1",
        POLICY_LABELS="1", POLICY_MIN_UNDRAWN="35", POLICY_MAX_UNDRAWN="36",
        POLICY_MAX_PER_GAME="2", POLICY_WORKERS="4", POLICY_NODE_BUDGET="20000000",
        POLICY_CHILD_BUDGET="20000000", POLICY_TIMEOUT="2h", POLICY_EXCLUDE_POSITIONS="",
        ARENA_GAMES="20", BASE_GAMES="0", BASELINES="", ARENA_SIMS="400",
        SEARCH_ARGS=SEARCH, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )


def create(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Freeze inputs and settings in a fresh run; never reuse an old league."""
    if not 0 < args.hours <= 168 or args.final_games <= 0 or args.final_games % 2:
        raise ValueError("Require 0 < hours <= 168 and positive even final-games")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    choices = [Path("work/solver34-vs30-Qeqlwh/reference.pt"),
               Path("experiments/2026-10-15_20-57-boxes-qdab-curriculum/logs/"
                    "qdab-20261016-044033/reference.pt"),
               EXP / "logs/astra-probe-PCcE6W/reference.pt"]
    reference = args.reference or next((p for p in choices if p.is_file()), None)
    if reference is None or not reference.is_file():
        raise FileNotFoundError("Supply --reference /path/to/frozen/iter0.pt")
    if not args.reference and digest(reference) != ITER0_SHA:
        raise ValueError("Automatic reference is not the verified iter0")
    model_config = torch.load(reference, map_location="cpu", weights_only=False)["config"]
    board = model_config["rows"], model_config["cols"], model_config.get("features")
    if board != (5, 5, "chains"):
        raise ValueError("This run requires a 5x5 checkpoint with chain features")
    data = Path(os.environ.get("GAME_DATA_DIR", str(Path.home() / "autoboxes-data/game_data_root")))
    (EXP / "logs").mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="5x5-solver34-", dir=EXP / "logs")).resolve()
    checkpoint_dir = (EXP / "checkpoints" / output.name).resolve()
    checkpoint_dir.mkdir(parents=True)
    frozen = checkpoint_dir / "iter0.pt"
    shutil.copyfile(reference, frozen)
    write_json(EXP / f"league_state-{output.name}.json", dict(champion=0, history=[]))
    settings = recipe()
    settings.update(TAG=output.name, GAME_DATA_DIR=str(data.resolve()))
    code = source_code()
    for path in [SCRIPT, *EXP.glob("*.py"), EXP / "run_iteration_local.sh"]:
        code[str(path.resolve())] = digest(path)
    import alpha_go_cpp
    code[str(Path(alpha_go_cpp.__file__).resolve())] = digest(Path(alpha_go_cpp.__file__))
    config = dict(hours=args.hours, final_games=args.final_games, final_workers=4,
                  final_seed=20261028, cpu=args.cpu, checkpoint_dir=str(checkpoint_dir),
                  environment=settings, reference=str(frozen), reference_sha256=digest(frozen),
                  code_sha256=code, cwd=str(Path.cwd()), scope="Exploratory stronger-teacher loop")
    write_json(output / "config.json", config)
    (output / "config.sha256").write_text(digest(output / "config.json") + "\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (output / "code-commit.txt").write_text(commit + "\n")
    (output / "resume.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(Path.cwd())) + "\n"
        "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\nuv run python -u " +
        shlex.quote(str(SCRIPT)) + " --resume " + shlex.quote(str(output)) + "\n")
    return output, config


def job(command: list[str], log: Path, env: dict[str, str],
        state: dict[str, Any], state_path: Path, count_work: bool) -> None:
    """Show heartbeats, persist elapsed work, and terminate child groups on interruption."""
    started = time.monotonic()
    before = state["active_seconds"]
    print("Running: " + shlex.join(command) + "\nLog: " + str(log), flush=True)
    with log.open("a") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            heartbeat = started
            while process.poll() is None:
                time.sleep(1)
                now = time.monotonic()
                if count_work:
                    state["active_seconds"] = before + now - started
                if now - heartbeat >= 30:
                    write_json(state_path, state)
                    lines = log.read_text(errors="replace").splitlines()
                    print(f"{log.stem}: {now - started:.0f}s; "
                          f"{lines[-1][-240:] if lines else 'starting'}", flush=True)
                    heartbeat = now
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if count_work:
                state["active_seconds"] = before + time.monotonic() - started
            write_json(state_path, state)


def final_command(config: dict[str, Any], candidate: Path, games: Path) -> list[str]:
    """Both networks get the same solver, simulations, utility and paired openings."""
    settings = config["environment"]
    command = [sys.executable, "-u", "-m", "alpha_go.boxes.arena", "--rows", "5", "--cols", "5",
               "--a", f"ckpt:{candidate}", "--b", "ckpt:" + config["reference"],
               "--num_games", str(config["final_games"]),
               "--num_workers", str(config["final_workers"]),
               "--seed", str(config["final_seed"]), "--opening_moves", "4",
               "--sims", settings["ARENA_SIMS"], "--solver", settings["ARENA_SOLVER_N"],
               "--solver_budget", settings["SOLVER_BUDGET"], "--save_games", str(games),
               *shlex.split(settings["SEARCH_ARGS"])]
    return command + (["--cpu"] if config["cpu"] else [])


def run(output: Path, config: dict[str, Any]) -> None:
    """Finish each iteration, retain every learner, and evaluate the last one after eight hours."""
    os.chdir(config["cwd"])
    checked_files({str(output / "config.json"): (output / "config.sha256").read_text().strip()})
    if config != json.loads((output / "config.json").read_text()):
        raise ValueError("Run configuration differs from the saved recipe")
    checked_files(config["code_sha256"])
    checked_files({config["reference"]: config["reference_sha256"]})
    if not config["cpu"] and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on resume")
    state_path = output / "state.json"
    state: dict[str, Any] = json.loads(state_path.read_text()) if state_path.exists() else dict(
        phase="training", completed_iterations=0, active_seconds=0.0, checkpoints={})
    checked_files(state["checkpoints"])
    env = dict(os.environ, **config["environment"])
    print(f"Run: {output}\nResume: bash {output / 'resume.sh'}\n"
          f"Work budget: {config['hours']:g} hours; finish the current iteration, then "
          f"{config['final_games']} games of LAST learner vs iter0 with identical search.\n"
          "Every iteration continues the latest weights; small arena results never reset them.",
          flush=True)
    try:
        while state["phase"] == "training":
            iteration = state["completed_iterations"]
            # Recover a committed iteration even if its arena completed just before interruption.
            candidate = Path(config["checkpoint_dir"]) / f"iter{iteration + 1}.pt"
            league = json.loads((EXP / f"league_state-{output.name}.json").read_text())
            entries = [r for r in league["history"] if r["iteration"] == iteration + 1]
            if entries:
                if len(entries) != 1:
                    raise ValueError("Duplicate iteration in league")
                checked_files({str(candidate): entries[0]["candidate_checkpoint_sha256"]})
                state["completed_iterations"] = iteration + 1
                state["checkpoints"][str(candidate)] = digest(candidate)
                write_json(state_path, state)
                continue
            if iteration and state["active_seconds"] >= config["hours"] * 3600:
                state["phase"] = "final"
                write_json(state_path, state)
                break
            command = ["bash", str(EXP / "run_iteration_local.sh"), str(iteration), str(iteration)]
            if config["cpu"]:
                command.append("--cpu")
            job(command, output / f"iteration-{iteration + 1}.log", env, state, state_path, True)
            # The next pass validates the league's checkpoint hash before advancing.
            league = json.loads((EXP / f"league_state-{output.name}.json").read_text())
            if not any(r["iteration"] == iteration + 1 for r in league["history"]):
                raise RuntimeError("Iteration exited without a committed arena")
        if state["phase"] == "final":
            candidate = Path(config["checkpoint_dir"]) / f"iter{state['completed_iterations']}.pt"
            games, log = output / "final-games.jsonl", output / "final-arena.log"
            command = final_command(config, candidate, games)
            write_json(output / "final-config.json", dict(command=command, checkpoints={
                str(candidate): digest(candidate),
                config["reference"]: config["reference_sha256"]}))
            job(command, log, env, state, state_path, False)
            text = log.read_text().rsplit("===RESULT===", 1)[1].lstrip()
            summary = json.JSONDecoder().raw_decode(text)[0]
            verify_match(games, summary, config["final_games"])
            write_json(output / "final.json", summary)
            state["phase"] = "complete"
            state["final_sha256"] = {str(p): digest(p) for p in (games, log, output / "final.json")}
            write_json(state_path, state)
        checked_files(state.get("final_sha256", {}))
        print("Final match: " + (output / "final.json").read_text(), flush=True)
    finally:
        # Include league, timings and training manifests alongside per-stage logs.
        shutil.copyfile(EXP / f"league_state-{output.name}.json", output / "league.json")
        for path in EXP.glob(f"dataset-{output.name}-it*.txt"):
            shutil.copyfile(path, output / path.name)
        timing = EXP / "timing" / output.name
        if timing.exists():
            shutil.copytree(timing, output / "timing", dirs_exist_ok=True)
        package_results(output)


def main() -> None:
    """Start an isolated run or resume its immutable recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--final-games", type=int, default=200)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    def interrupted(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / "config.json").read_text())
    else:
        output, config = create(args)
    with (output / "overnight.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(output, config)


if __name__ == "__main__":
    main()
