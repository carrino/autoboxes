"""Compare verified proof24 weights on two frozen development bands, without training."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
BASE = Path("experiments/2026-10-08_14-42-boxes-3x3-loop")
SOURCE_RUN = BASE / "logs/5x5-proof34-a6gdsd3i"
SOURCE_CONFIG_SHA = "b4c5b049727a6e9795acc6db3a676e31ef012af17c5ccd21dbeb37bb0ec9e0b1"
CHECKPOINTS = {
    "reference": "3f17b19f3caa66905be1f33a960c038fec01fa499cda483ed00367b99e164dac",
    "candidate": "73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d",
}
INPUTS = Path("experiments/2026-10-17_15-27-boxes-band-progression/data/pilot-20m")
INPUT_HASHES = {
    "positions-35-36.jsonl": "64d362643fe399c9a99b7ef6bdc82b50aac5b50af5fd2895b72ceea5c9597749",
    "positions-37-38.jsonl": "f097e28c1ec8a834f112ef67fa556a12a4ab769bdb4f694900798698cb2e3cc1",
    "labels.json": "81ea3b4a84392cf9c5cfb1143b68584531564f8be822cb6615d6aff51fefe9f6",
}
BANDS = ((35, 36), (37, 38))


def digest(path: Path) -> str:
    """Hash an input or result without changing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Replace progress atomically so interruptions preserve the last complete record."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checked(path: Path, expected: str) -> None:
    """Reject missing or changed files before evaluating a different experiment."""
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Missing or changed verified input: {path}")


def pin_runtime() -> None:
    """Use this checkout and the invoking interpreter for all subprocesses."""
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    os.environ.update(PYTHONPATH=str(REPO / "src"), PYTHONNOUSERSITE="1",
                      UV_PROJECT_ENVIRONMENT=sys.prefix,
                      PATH=str(Path(sys.prefix) / "bin") + os.pathsep + os.environ["PATH"],
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def runtime_identity(source: dict[str, Any], cpu: bool) -> dict[str, Any]:
    """Require the overnight binary and unchanged search code; do not rebuild."""
    import torch

    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; --cpu is for explicit CPU diagnostics only")
    modules = {}
    for name in ("alpha_go", "alpha_go.boxes.search_probe", "alpha_go.boxes.probe_report",
                 "alpha_go.boxes.nn_agent", "alpha_go.boxes.mixed_evaluator",
                 "alpha_go.boxes.model", "alpha_go.boxes.rules"):
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError(f"Imported module has no file: {name}")
        actual = Path(module.__file__).resolve()
        relative = "alpha_go/__init__.py" if name == "alpha_go" else name.replace(".", "/") + ".py"
        expected = REPO / "src" / relative
        if actual != expected:
            raise ValueError(f"Wrong checkout imported: {name}: {actual}")
        modules[name] = dict(path=str(actual), sha256=digest(actual))
    native = importlib.import_module("alpha_go_cpp")
    if native.__file__ is None:
        raise ValueError("Imported native extension has no file")
    extension = Path(native.__file__).resolve()
    checked(extension, source["preflight"]["runtime"]["extension_sha256"])
    for name, expected_sha in source["code_sha256"].items():
        if name.startswith("src/"):
            checked(REPO / name, expected_sha)
    return dict(interpreter=sys.executable, prefix=sys.prefix, python=sys.version,
                modules=modules, extension=str(extension), extension_sha256=digest(extension),
                device="cpu" if cpu else "cuda")


def create(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Freeze the predeclared models, positions and exact cache in a fresh directory."""
    source_path = args.source_run.resolve() / "config.json"
    checked(source_path, SOURCE_CONFIG_SHA)
    source = json.loads(source_path.read_text())
    checkpoint_dir = REPO / BASE / "checkpoints/5x5-proof34-a6gdsd3i"
    models = {"reference": args.reference or checkpoint_dir / "iter0.pt",
              "candidate": args.candidate or checkpoint_dir / "iter24.pt"}
    for name, path in models.items():
        checked(path, CHECKPOINTS[name])
    for name, expected in INPUT_HASHES.items():
        checked(REPO / INPUTS / name, expected)
    runtime = runtime_identity(source, args.cpu)
    logs = REPO / BASE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="proof24-bands-", dir=logs))
    (output / "inputs").mkdir()
    shutil.copyfile(source_path, output / "inputs/source-config.json")
    for name, path in models.items():
        shutil.copyfile(path, output / "inputs" / f"{name}.pt")
        checked(output / "inputs" / f"{name}.pt", CHECKPOINTS[name])
    for name in INPUT_HASHES:
        shutil.copyfile(REPO / INPUTS / name, output / "inputs" / name)
    code = {str(path.relative_to(REPO)): digest(path) for path in
            [SCRIPT, SCRIPT.with_name("run_band_probe.sh"), *(REPO / "src/alpha_go").rglob("*.py")]}
    config = dict(schema=1, cwd=str(REPO), runtime=runtime, cpu=args.cpu,
                  band_timeout_seconds=args.band_minutes * 60, code_sha256=code,
                  checkpoint_sha256=CHECKPOINTS, input_sha256=INPUT_HASHES,
                  source_config_sha256=SOURCE_CONFIG_SHA, bands=BANDS,
                  scope="Fixed development diagnostic, not an arena promotion. "
                        "Unknown labels are excluded only on the same roots for both models. "
                        "Coverage can favor easier solves; no 39..40 read or training.")
    write_json(output / "config.json", config)
    (output / "config.sha256").write_text(digest(output / "config.json") + "\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (output / "code-commit.txt").write_text(commit + "\n")
    (output / "resume.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(REPO)) + "\n" +
        shlex.join([sys.executable, "-u", str(SCRIPT), "--resume", str(output)]) + "\n")
    return output, config


def verify(output: Path, config: dict[str, Any]) -> None:
    """Resume only with the original sources, runtime, checkpoints and positions."""
    checked(output / "config.json", (output / "config.sha256").read_text().strip())
    if config["cwd"] != str(REPO):
        raise ValueError("Resume in the original checkout")
    for name, expected in config["code_sha256"].items():
        checked(REPO / name, expected)
    for name, expected in CHECKPOINTS.items():
        checked(output / "inputs" / f"{name}.pt", expected)
    for name, expected in INPUT_HASHES.items():
        checked(output / "inputs" / name, expected)
    source_path = output / "inputs/source-config.json"
    checked(source_path, SOURCE_CONFIG_SHA)
    if runtime_identity(json.loads(source_path.read_text()), config["cpu"]) != config["runtime"]:
        raise ValueError("Runtime changed; restore the original interpreter and native extension")


def merged_cache(output: Path, destination: Path) -> None:
    """Reuse exact successes from earlier attempts; never cache a timeout as a value."""
    values: dict[str, int] = {}
    paths = [output / "inputs/labels.json", *sorted(output.glob("*/attempt-*/probe/labels.json"))]
    for path in paths:
        document = json.loads(path.read_text())
        if (document["schema"], document["rows"], document["cols"]) != (1, 5, 5):
            raise ValueError(f"Invalid exact cache geometry/schema: {path}")
        for key, value in document["remaining_values"].items():
            if (type(value) is not int or abs(value) > 25 or not key.isdecimal()
                    or not 0 <= int(key) < 1 << 60):
                raise ValueError(f"Invalid exact cache entry: {path}: {key}")
            if key in values and values[key] != value:
                raise ValueError(f"Conflicting exact cache entry: {key}")
            values[key] = value
    write_json(destination, dict(schema=1, rows=5, cols=5, remaining_values=values))


def job(command: list[str], log: Path, seconds: float, heartbeat: float = 30) -> None:
    """Bound wall time, show progress, and stop the process group on every exit path."""
    started = time.monotonic()
    print("Running: " + shlex.join(command) + "\nLog: " + str(log), flush=True)
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            last = started
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= seconds:
                    raise TimeoutError(f"Band exceeded {seconds / 60:g} minutes: {log}")
                if time.monotonic() - last >= heartbeat:
                    lines = log.read_text(errors="replace").splitlines()
                    print(f"{log.parent.parent.name}: {elapsed:.0f}s; " +
                          (lines[-1][-240:] if lines else "starting"), flush=True)
                    last = time.monotonic()
                time.sleep(min(1, max(.01, seconds - elapsed)))
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()


def probe_command(output: Path, attempt: Path, minimum: int, maximum: int,
                  cpu: bool) -> list[str]:
    """Use identical fixed search settings for the old winner and proof24 endpoint."""
    command = [sys.executable, "-u", "-m", "alpha_go.boxes.search_probe",
               "--reference", str(output / "inputs/reference.pt"),
               "--candidate", str(output / "inputs/candidate.pt"),
               "--positions-file", str(output / "inputs" / f"positions-{minimum}-{maximum}.jsonl"),
               "--label-cache", str(attempt / "input-labels.json"),
               "--output", str(attempt / "probe"), "--rows", "5", "--cols", "5",
               "--min-undrawn", str(minimum), "--max-undrawn", str(maximum),
               "--mixes", "reference", "candidate", "--simulations", "400", "--lambdas", "0.5",
               "--solver", "34", "--solver-budget", "50000", "--label-budget", "20000000",
               "--c-puct", "1.5", "--leaf-batch-size", "16", "--policy-temperature", "1.0",
               "--margin-utility-k", "6", "--prove-terminals", "1", "--shape-terminals", "1",
               "--seed", "20261030"]
    return command + (["--cpu"] if cpu else [])


def stage(output: Path, config: dict[str, Any], minimum: int, maximum: int) -> dict[str, Any]:
    """Keep completed bands; retry interrupted bands in new directories with exact caches."""
    from alpha_go.boxes.probe_report import build_report, write_report

    folder = output / f"{minimum}-{maximum}"
    folder.mkdir(exist_ok=True)
    receipt_path = folder / "completed.json"
    if receipt_path.is_file():
        receipt: dict[str, Any] = json.loads(receipt_path.read_text())
        for name, expected in receipt["artifacts_sha256"].items():
            checked(output / name, expected)
        print(f"{minimum}..{maximum}: completed band preserved", flush=True)
        return receipt
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=folder))
    merged_cache(output, attempt / "input-labels.json")
    command = probe_command(output, attempt, minimum, maximum, config["cpu"])
    write_json(attempt / "command.json", command)
    job(command, attempt / "run.log", config["band_timeout_seconds"])
    report = build_report(attempt / "probe", seed=20261030, resamples=10_000)
    write_report(report, attempt / "report")
    artifacts = {str(path.relative_to(output)): digest(path)
                 for path in attempt.rglob("*") if path.is_file()}
    receipt = dict(attempt=str(attempt.relative_to(output)), artifacts_sha256=artifacts,
                   band_report=report["bands"][f"{minimum}..{maximum}"])
    write_json(receipt_path, receipt)
    return receipt


def package(output: Path) -> None:
    """Always export partial logs/labels/results; large checkpoint copies remain local."""
    temporary = output / "results.tgz.tmp"
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(output.rglob("*")):
            if (path.is_file() and not path.is_symlink() and path.suffix != ".pt"
                    and path.name not in {"results.tgz", "results.tgz.tmp", "run.lock"}):
                archive.add(path, arcname=str(path.relative_to(output)), recursive=False)
    temporary.replace(output / "results.tgz")
    print(f"Send this file: {output / 'results.tgz'}", flush=True)


def run(output: Path, config: dict[str, Any]) -> int:
    """Attempt both bounded bands and package successes and failures independently."""
    state: dict[str, Any] = dict(complete=False, bands={}, scope=config["scope"])
    try:
        for minimum, maximum in BANDS:
            key = f"{minimum}..{maximum}"
            try:
                state["bands"][key] = dict(complete=True, **stage(output, config, minimum, maximum))
            except (TimeoutError, subprocess.CalledProcessError) as error:
                state["bands"][key] = dict(complete=False, error=str(error))
                print(f"{key}: {error}; keeping partial artifacts and attempting the next band",
                      flush=True)
            write_json(output / "state.json", state)
        state["complete"] = all(band["complete"] for band in state["bands"].values())
        return 0 if state["complete"] else 1
    except BaseException as error:
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "state.json", state)
        package(output)


def main(argv: list[str] | None = None) -> int:
    """Start a fixed diagnostic, or resume its saved input and execution identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=REPO / SOURCE_RUN)
    parser.add_argument("--reference", type=Path,
                        help="alternate path to the same verified weights")
    parser.add_argument("--candidate", type=Path,
                        help="alternate path to the same verified weights")
    parser.add_argument("--band-minutes", type=float, default=60,
                        help="wall limit per band, maximum 60 (default: 60)")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true",
                        help="explicit CPU diagnostic; never automatic")
    args = parser.parse_args(argv)
    if not math.isfinite(args.band_minutes) or not 0 < args.band_minutes <= 60:
        parser.error("--band-minutes must be positive and at most 60")
    pin_runtime()

    def interrupted(_signal: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / "config.json").read_text())
    else:
        output, config = create(args)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(output, config)
        print(f"Run: {output}\nResume: bash {output / 'resume.sh'}\n"
              f"80 frozen positions per band; at most {config['band_timeout_seconds'] / 60:g} "
              "minutes per band, plus shutdown/report time. No training.", flush=True)
        return run(output, config)


if __name__ == "__main__":
    raise SystemExit(main())
