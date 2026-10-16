"""Resumable overnight exact-policy curriculum on fresh Qdab and searched histories.

The work budget includes collection, solving and training. A fresh, side-swapped
full-game match follows the budget. All outputs are isolated from the existing league.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import torch

EXP = Path("experiments/2026-10-15_20-57-boxes-qdab-curriculum")
OLD_EXP = Path("experiments/2026-10-08_14-42-boxes-3x3-loop")


def digest(path: Path) -> str:
    """Hash a recorded input or completed artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Publish state atomically after its dependent artifacts exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def source_code() -> dict[str, str]:
    """Freeze the Boxes implementation plus the shared game/data path for resume."""
    paths = sorted(Path("src/alpha_go/boxes").glob("*.py"))
    paths += [Path("src/alpha_go") / name for name in ("gameplay.py", "self_play.py")]
    return {str(p): digest(p) for p in paths}


def checked_files(files: dict[str, str]) -> None:
    """Refuse to resume with changed or missing recorded files."""
    for name, expected in files.items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Recorded input/artifact changed: {name}")


def check_receipt(path: Path) -> dict[str, Any]:
    """Verify completed artifacts, including unexpected additions to prepared datasets."""
    result: dict[str, Any] = json.loads(path.read_text())
    checked_files(result["artifacts"])
    for directory in result.get("directories", []):
        actual = {str(p) for p in Path(directory).rglob("*") if p.is_file()}
        expected = {name for name in result["artifacts"] if Path(name).is_relative_to(directory)}
        if actual != expected:
            raise ValueError(f"Completed directory has added/missing artifacts: {directory}")
    return result


def check_rehearsal(config: dict[str, Any]) -> None:
    """Keep the opening rehearsal population fixed across training cycles."""
    recorded = config.get("extra_inputs_sha256", {})
    checked_files(recorded)
    if "extra_inputs_sha256" in config:
        actual = {
            str(p.resolve())
            for d in config["rehearsal_dirs"]
            for p in Path(d).rglob("*.npz")
            if not p.name.endswith(".tmp.npz")
        }
        expected = {p for p in recorded if p.endswith(".npz")}
        if actual != expected:
            raise ValueError(
                "Opening rehearsal files changed; resume requires the frozen population"
            )


def run_job(directory: Path, command: list[str], artifacts: list[Path], timeout: float) -> None:
    """Run one bounded child, preserving its partial outputs and verifying completed jobs."""
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "job.json"
    if receipt.exists():
        done = check_receipt(receipt)
        if done["command"] != command:
            raise ValueError(f"Completed job recipe changed: {directory}")
        print(f"Reuse completed {directory.name}", flush=True)
        return
    # Collection and solving resume internally. A partially written training stage
    # restarts from its frozen parent; retain the interrupted attempt for inspection.
    if "alpha_go.boxes.curriculum_train" in command:
        training = Path(command[command.index("--output") + 1])
        if training.exists():
            attempt = 1
            previous = training.with_name(f"interrupted-train-{attempt}")
            while previous.exists():
                attempt += 1
                previous = training.with_name(f"interrupted-train-{attempt}")
            training.rename(previous)
    print(f"Running {directory.name}; log: {directory / 'run.log'}", flush=True)
    started = time.monotonic()
    with (directory / "run.log").open("a") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > timeout:
                    raise TimeoutError(
                        f"{directory.name} exceeded {timeout:.0f}s; partial work saved"
                    )
                time.sleep(min(10.0, max(0.01, timeout - elapsed)))
                if int(time.monotonic() - started) // 30 != int(elapsed) // 30:
                    lines = (directory / "run.log").read_text(errors="replace").splitlines()
                    print(
                        f"  {directory.name}: {time.monotonic() - started:.0f}s; "
                        f"{lines[-1][-240:] if lines else 'starting'}",
                        flush=True,
                    )
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    files = [
        p
        for item in artifacts
        for p in (sorted(item.rglob("*")) if item.is_dir() else [item])
        if p.is_file()
    ]
    if any(not p.exists() for p in artifacts) or not files:
        raise ValueError(f"Job finished without expected artifacts: {artifacts}")
    write_json(
        receipt,
        dict(
            command=command,
            seconds=time.monotonic() - started,
            artifacts={str(p): digest(p) for p in files},
            directories=[str(p) for p in artifacts if p.is_dir()],
        ),
    )


def retained_sources(root: Path) -> tuple[list[Path], list[Path]]:
    """Find existing searched games; opening rehearsal comes from the iter0 collector."""
    for tag in ("5x5-policy-loop-cS6u3n", "5x5-exact", "5x5-s1600"):
        base = root / OLD_EXP / tag
        directories = sorted(base.glob("selfplay-it*"))
        files = sorted(
            p for d in directories for p in d.rglob("*.npz") if not p.name.endswith(".tmp.npz")
        )
        rehearsal = base / "selfplay-it0"
        if files and rehearsal.is_dir() and list(rehearsal.rglob("*.npz")):
            return files, [rehearsal]
    raise FileNotFoundError(
        "No searched replay with selfplay-it0 found. Pass --replay-dir "
        "and --rehearsal-dir to select existing searched games."
    )


def make_config(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Validate all local prerequisites before spending time collecting teacher games."""
    if (
        not 0 < args.hours <= 168
        or args.cycles <= 0
        or args.final_games < 2
        or args.final_games % 2
    ):
        raise ValueError("Require positive hours/cycles and positive even final-games")
    if (
        min(
            args.teacher_games,
            args.monitor_games,
            args.retained_games,
            args.workers,
            args.epochs,
            args.final_workers,
            args.teacher_seconds,
        )
        <= 0
    ):
        raise ValueError("Collection, training and worker counts must be positive")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; --cpu is only for tiny integration checks")
    root = Path(os.environ.get("GAME_DATA_DIR", str(Path.home() / "autoboxes-data/game_data_root")))
    choices = [
        OLD_EXP / "checkpoints/5x5-policy-loop-cS6u3n/iter0.pt",
        OLD_EXP / "logs/astra-probe-PCcE6W/reference.pt",
    ]
    reference = args.reference or next((p for p in choices if p.is_file()), None)
    if reference is None or not reference.is_file():
        raise FileNotFoundError("Frozen iter0 missing; supply --reference /path/to/iter0.pt")
    checkpoint = torch.load(reference, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if (cfg["rows"], cfg["cols"], cfg.get("features")) != (5, 5, "chains"):
        raise ValueError("This curriculum requires a 5x5 checkpoint with chain features")
    if args.replay_dir:
        files = sorted(
            p for d in args.replay_dir for p in d.rglob("*.npz") if not p.name.endswith(".tmp.npz")
        )
        rehearsal = args.rehearsal_dir or args.replay_dir
    else:
        files, rehearsal = retained_sources(root)
    if not files or any(not p.is_dir() for p in rehearsal):
        raise ValueError("Require existing searched replay and opening rehearsal directories")
    random.Random(args.seed).shuffle(files)
    output = (
        args.output or EXP / "logs" / time.strftime("qdab-%Y%m%d-%H%M%S", time.gmtime())
    ).resolve()
    if output.exists():
        raise ValueError(f"Output already exists; use --resume {output}")
    output.mkdir(parents=True)
    frozen = output / "reference.pt"
    shutil.copyfile(reference, frozen)
    inputs = {str(p.resolve()): digest(p) for p in files}
    write_json(output / "source-games.json", inputs)
    config = dict(
        schema=1,
        hours=args.hours,
        cycles=args.cycles,
        monitor_games=args.monitor_games,
        teacher_games=args.teacher_games,
        retained_games=args.retained_games,
        workers=args.workers,
        teacher_seconds=args.teacher_seconds,
        epochs=args.epochs,
        seed=args.seed,
        final_games=args.final_games,
        final_workers=args.final_workers,
        reference=str(frozen),
        reference_sha256=digest(frozen),
        release=str(args.release.resolve()),
        cpu=args.cpu,
        rehearsal_dirs=[str(p.resolve()) for p in rehearsal],
        sources_file_sha256=digest(output / "source-games.json"),
        code_sha256=source_code(),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        starter=str((EXP / "data/qdab-starter.jsonl").resolve()),
    )
    starter = Path(config["starter"])
    if starter.is_file():
        config["starter_sha256"] = digest(starter)
    probe = OLD_EXP / "logs/astra-probe-PCcE6W/mix/positions.jsonl"
    config["exclude_positions"] = [str(probe.resolve())] if probe.is_file() else []
    config["extra_inputs_sha256"] = {
        str(p.resolve()): digest(p)
        for folder in rehearsal
        for p in folder.rglob("*.npz")
        if not p.name.endswith(".tmp.npz")
    }
    config["extra_inputs_sha256"].update({p: digest(Path(p)) for p in config["exclude_positions"]})
    write_json(output / "config.json", config)
    resume = "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(Path.cwd())) + "\n"
    resume += "export GAME_DATA_DIR=" + shlex.quote(str(root)) + "\n"
    resume += (
        "uv run python -u -m alpha_go.boxes.curriculum --resume " + shlex.quote(str(output)) + "\n"
    )
    (output / "resume.sh").write_text(resume)
    return output, config


def collect(output: Path, config: dict[str, Any], count: int, seed: int) -> Path:
    """Generate a bounded batch of actual Qdab searched histories."""
    prefix = output / "prefixes.jsonl"
    command = [
        sys.executable,
        "-u",
        "-m",
        "alpha_go.boxes.qdab_collect",
        "--release",
        config["release"],
        "--output",
        str(prefix),
        "--num-games",
        str(count),
        "--max-moves",
        "30",
        "--seconds",
        str(config["teacher_seconds"]),
        "--seed",
        str(seed),
    ]
    run_job(
        output,
        command,
        [prefix, prefix.with_suffix(".meta.json")],
        timeout=count * 30 * (config["teacher_seconds"] + 15) + 120,
    )
    return prefix


def label(
    output: Path, sources: list[Path], config: dict[str, Any], monitor: Path | None = None
) -> Path:
    """Solve all suitable roots, with independent exact child checks and finite budgets."""
    prepared = output / "prepared"
    command = [
        sys.executable,
        "-u",
        "-m",
        "alpha_go.boxes.curriculum_labels",
        "--sources",
        *map(str, sources),
        "--output",
        str(prepared),
        "--min-undrawn",
        "31",
        "--max-undrawn",
        "36",
        "--node-budget",
        "20000000",
        "--child-node-budget",
        "5000000",
        "--workers",
        str(config["workers"]),
        "--split",
        "train" if monitor else "validation",
    ]
    if monitor:
        command += ["--exclude-prepared", str(monitor)]
    if config.get("exclude_positions"):
        command += ["--exclude-positions", *config["exclude_positions"]]
    run_job(output, command, [prepared], timeout=2 * 3600)
    return prepared


def final_match(output: Path, config: dict[str, Any], candidate: Path) -> None:
    """Run the predeclared last-candidate comparison using fresh side-swapped openings."""
    directory = output / "final-vs-iter0"
    games = directory / "games.jsonl"
    command = [
        sys.executable,
        "-u",
        "-m",
        "alpha_go.boxes.arena",
        "--rows",
        "5",
        "--cols",
        "5",
        "--a",
        f"ckpt:{candidate}",
        "--b",
        "ckpt:" + config["reference"],
        "--num_games",
        str(config["final_games"]),
        "--num_workers",
        str(config["final_workers"]),
        "--seed",
        str(config["seed"] + 900000),
        "--opening_moves",
        "4",
        "--sims",
        "400",
        "--solver",
        "30",
        "--solver_budget",
        "50000",
        "--policy_temperature",
        "1.0",
        "--prove_terminals",
        "1",
        "--shape_terminals",
        "1",
        "--margin_utility_lambda",
        "0.5",
        "--margin_utility_k",
        "6",
        "--save_games",
        str(games),
    ]
    if config["cpu"]:
        command += ["--cpu"]
    run_job(directory, command, [games, directory / "run.log"], timeout=90 * 60)
    text = (directory / "run.log").read_text()
    summary = json.JSONDecoder().raw_decode(text.rsplit("===RESULT===", 1)[1].lstrip())[0]
    verify_match(games, summary, config["final_games"])
    write_json(output / "final.json", summary)
    print("Final match: " + json.dumps(summary), flush=True)


def verify_match(games: Path, summary: dict[str, Any], expected: int) -> None:
    """Replay the final match independently and attach an opening-pair bootstrap interval."""
    import numpy as np

    from alpha_go.boxes.rules import BoxesBoard

    records = [json.loads(line) for line in games.read_text().splitlines()]
    assert len(records) == summary["games"] == expected
    margins = []
    for index, record in enumerate(records):
        assert record["index"] == index and record["a_first"] == (index % 2 == 0)
        assert record["opening"] == 4 and len(record["moves"]) == 60
        board = BoxesBoard(5, 5)
        for row, col in record["moves"]:
            assert board.play_edge(board.edge_index(row, col)), "illegal final-match edge"
        assert board.is_game_over()
        margin = round(board.score()) * (1 if record["a_first"] else -1)
        assert margin == record["a_margin"]
        margins.append(margin)
        if index % 2:
            assert record["moves"][:4] == records[index - 1]["moves"][:4]
    values = np.asarray(margins)
    assert int((values > 0).sum()) == summary["a_wins"]
    assert int((values < 0).sum()) == summary["b_wins"]
    assert int((values == 0).sum()) == summary["draws"]
    pairs = (values.reshape(-1, 2) > 0).mean(axis=1)
    indices = np.random.default_rng(20261018).integers(0, len(pairs), (10000, len(pairs)))
    summary["paired_game_score_ci95"] = np.quantile(
        pairs[indices].mean(axis=1), [0.025, 0.975]
    ).tolist()
    summary["independent_replay_verified"] = True


def package_results(output: Path) -> None:
    """Package small evidence and resume/config files; leave weights and NPZs on WSL."""
    archive = output / "results.tgz"
    temporary = output / "results.tmp.tgz"
    with tarfile.open(temporary, "w:gz") as packed:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix not in (".pt", ".npz", ".tgz", ".lock"):
                packed.add(path, arcname=str(path.relative_to(output)))
    temporary.replace(archive)
    print(f"Send this file: {archive}", flush=True)


def run(output: Path, config: dict[str, Any]) -> None:
    """Collect/solve/train throughout the work budget, then evaluate the final candidate."""
    checked_files(config["code_sha256"])
    checked_files(
        {
            config["reference"]: config["reference_sha256"],
            str(output / "source-games.json"): config["sources_file_sha256"],
        }
    )
    if "starter_sha256" in config:
        checked_files({config["starter"]: config["starter_sha256"]})
    check_rehearsal(config)
    sources = json.loads((output / "source-games.json").read_text())
    checked_files(sources)
    state_path = output / "state.json"
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else dict(
            completed_cycles=0,
            active_seconds=0.0,
            accepted_checkpoint=config["reference"],
            last_candidate=None,
            maximum_undrawn=34,
            prepared=[],
            receipts={},
            config_sha256=config_hash,
        )
    )
    if state["config_sha256"] != config_hash:
        raise ValueError("Saved run settings changed; start a new isolated run")
    checked_files(state["receipts"])
    for path in state["receipts"]:
        check_receipt(Path(path))
    started = time.monotonic()
    previous_seconds = state["active_seconds"]

    def save() -> None:
        state["active_seconds"] = previous_seconds + time.monotonic() - started
        write_json(state_path, state)

    print(
        f"Run: {output}\nResume: bash {output / 'resume.sh'}\n"
        f"Work budget: {config['hours']:g} hours, then {config['final_games']} games "
        "of the LAST candidate vs frozen iter0. Completed stages survive interruption.",
        flush=True,
    )
    try:
        if state.get("status") == "complete":
            print("Run already complete.", flush=True)
            return
        monitor_prefix = collect(
            output / "monitor-collect", config, config["monitor_games"], config["seed"]
        )
        monitor = label(output / "monitor-label", [monitor_prefix], config)
        while state["completed_cycles"] < config["cycles"]:
            save()
            if state["active_seconds"] >= config["hours"] * 3600 and state["last_candidate"]:
                break
            check_rehearsal(config)
            checked_files(state["receipts"])
            for path in state["receipts"]:
                check_receipt(Path(path))
            cycle = state["completed_cycles"] + 1
            batch = output / f"cycle-{cycle:03d}"
            print(f"Cycle {cycle}: train at 31..{state['maximum_undrawn']} undrawn", flush=True)
            teacher = collect(
                batch / "collect",
                config,
                config["teacher_games"],
                config["seed"] + 1000 + cycle * config["teacher_games"],
            )
            names = list(sources)
            begin = (cycle - 1) * config["retained_games"]
            selected = [Path(p) for p in names[begin : begin + config["retained_games"]]]
            checked_files({str(p): sources[str(p)] for p in selected})
            selected.append(teacher)
            if cycle == 1 and "starter_sha256" in config:
                selected.append(Path(config["starter"]))
            prepared = label(batch / "label", selected, config, monitor)
            training = [*state["prepared"], str(prepared)]
            command = [
                sys.executable,
                "-u",
                "-m",
                "alpha_go.boxes.curriculum_train",
                "--reference",
                state["accepted_checkpoint"],
                "--train-dirs",
                *[str(Path(p) / "train") for p in training],
                "--validation-dirs",
                str(monitor / "validation"),
                "--rehearsal-dirs",
                *config["rehearsal_dirs"],
                "--output",
                str(batch / "train"),
                "--max-undrawn",
                str(state["maximum_undrawn"]),
                "--epochs",
                str(config["epochs"]),
                "--lr",
                "0.0001",
                "--batch-size",
                "128",
                "--seed",
                str(config["seed"] + cycle),
            ]
            if config["cpu"]:
                command += ["--cpu"]
            run_job(batch / "training-job", command, [batch / "train"], timeout=3600)
            result = json.loads((batch / "train/summary.json").read_text())
            candidate = str((batch / "train/candidate.pt").resolve())
            state.update(completed_cycles=cycle, last_candidate=candidate, prepared=training)
            for receipt in (batch / "label/job.json", batch / "training-job/job.json"):
                state["receipts"][str(receipt)] = digest(receipt)
            if result["accepted"]:
                state["accepted_checkpoint"] = candidate
            # This gate selects curriculum difficulty on a development set, not promotion.
            if result.get("advance_curriculum", False):
                state["maximum_undrawn"] = 36
            print(
                f"Cycle {cycle}: accepted={result['accepted']}; next boundary "
                f"{state['maximum_undrawn']}; {state['active_seconds'] / 3600:.2f}h used",
                flush=True,
            )
            save()
        if not state["last_candidate"]:
            raise RuntimeError("No candidate trained; retained partial work is resumable")
        save()
        state["status"] = "final-match"
        write_json(state_path, state)
        final_match(output, config, Path(state["last_candidate"]))
        final_receipt = output / "final-vs-iter0/job.json"
        state["receipts"][str(final_receipt)] = digest(final_receipt)
        state["status"] = "complete"
    finally:
        save()
        package_results(output)


def main() -> None:
    """Start an isolated eight-hour run or resume its frozen recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--replay-dir", type=Path, nargs="+")
    parser.add_argument("--rehearsal-dir", type=Path, nargs="+")
    parser.add_argument("--release", type=Path, default=Path("work/qdab-linux"))
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--cycles", type=int, default=128)
    parser.add_argument("--monitor-games", type=int, default=16)
    parser.add_argument("--teacher-games", type=int, default=4)
    parser.add_argument("--retained-games", type=int, default=64)
    parser.add_argument("--teacher-seconds", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--final-games", type=int, default=200)
    parser.add_argument("--final-workers", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # Turn TERM into a normal unwind so child process groups and partial receipts
    # are handled exactly like an interactive interruption.
    def interrupted(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("Interrupted; completed stages are saved")

    signal.signal(signal.SIGTERM, interrupted)
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / "config.json").read_text())
        if not config["cpu"] and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable on resume")
    else:
        output, config = make_config(args)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(output, config)


if __name__ == "__main__":
    main()
