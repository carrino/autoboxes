"""A bounded policy-supervision comparison from the verified proof34 iter24 winner."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
BASE = Path("experiments/2026-10-08_14-42-boxes-3x3-loop")
HELPER = REPO / "experiments/2026-10-18_10-40-boxes-proof24-result/band_probe.py"
spec = importlib.util.spec_from_file_location("proof24_band_helpers", HELPER)
assert spec and spec.loader
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
digest = helpers.digest
checked = helpers.checked
write_json = helpers.write_json
SEED = 20261031
CAPS = {"collect": 3600, "prepare": 5400, "train": 900,
        "control-vs-exact": 3600, "iter24-vs-exact": 3600}


def merge_caches(paths: list[Path], destination: Path) -> None:
    """Keep only exact successes, rejecting conflicting or malformed cache entries."""
    values: dict[str, int] = {}
    for path in paths:
        source = json.loads(path.read_text())
        if (source["schema"], source["rows"], source["cols"]) != (1, 5, 5):
            raise ValueError(f"Wrong exact-cache schema: {path}")
        for key, value in source["remaining_values"].items():
            if (not key.isdecimal() or not 0 <= int(key) < 1 << 60
                    or type(value) is not int or abs(value) > 25):
                raise ValueError(f"Invalid exact-cache entry: {path}: {key}")
            if key in values and values[key] != value:
                raise ValueError(f"Conflicting exact value: {key}")
            values[key] = value
    write_json(destination, dict(schema=1, rows=5, cols=5, remaining_values=values))


def create(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Check model/runtime first, then freeze all inputs without changing the old run."""
    source_path = REPO / helpers.SOURCE_RUN / "config.json"
    checked(source_path, helpers.SOURCE_CONFIG_SHA)
    checkpoint = REPO / BASE / "checkpoints/5x5-proof34-a6gdsd3i/iter24.pt"
    checked(checkpoint, helpers.CHECKPOINTS["candidate"])
    runtime = helpers.runtime_identity(json.loads(source_path.read_text()), False)
    for name, expected in helpers.INPUT_HASHES.items():
        checked(REPO / helpers.INPUTS / name, expected)
    logs = REPO / BASE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="all-policy-ab-", dir=logs))
    inputs = output / "inputs"
    inputs.mkdir()
    shutil.copyfile(checkpoint, inputs / "iter24.pt")
    checked(inputs / "iter24.pt", helpers.CHECKPOINTS["candidate"])
    shutil.copyfile(source_path, inputs / "source-config.json")
    positions = "".join((REPO / helpers.INPUTS / f"positions-{lo}-{hi}.jsonl").read_text()
                        for lo, hi in helpers.BANDS)
    (inputs / "positions.jsonl").write_text(positions)
    # Also keep the unused 39..40 monitor and earlier diagnostic families out of training.
    exclusions = [REPO / helpers.INPUTS / "positions-39-40.jsonl",
                  REPO / "experiments/2026-10-17_10-17-boxes-search-target-audit"
                  / "data/positions.jsonl"]
    (inputs / "exclude.jsonl").write_text(positions + "".join(p.read_text() for p in exclusions))
    caches = [REPO / helpers.INPUTS / "labels.json"]
    previous = args.previous_probe.resolve()
    if previous.is_dir():
        for lo, hi in helpers.BANDS:
            receipt_path = previous / f"{lo}-{hi}/completed.json"
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                for name, expected in receipt["artifacts_sha256"].items():
                    checked(previous / name, expected)
                cache = previous / receipt["attempt"] / "probe/labels.json"
                caches.append(cache)
    merge_caches(caches, inputs / "labels.json")
    code = [SCRIPT, SCRIPT.with_name("run_pilot.sh"), HELPER,
            REPO / BASE / "run_games.py", *(REPO / "src/alpha_go").rglob("*.py")]
    config = dict(schema=1, cwd=str(REPO), runtime=runtime, games=args.games, seed=SEED,
                  caps_seconds=CAPS, input_sha256={p.name: digest(p) for p in inputs.iterdir()},
                  code_sha256={str(p.relative_to(REPO)): digest(p) for p in code},
                  cached_from={str(p): digest(p) for p in caches},
                  scope="Development A/B, no automatic promotion. Control: mistake-only exact "
                        "policies; treatment: every complete searched 35..36 policy. Both share "
                        "all-root exact values and children, fresh full-game rows and updates.")
    write_json(output / "config.json", config)
    (output / "config.sha256").write_text(digest(output / "config.json") + "\n")
    (output / "code-commit.txt").write_text(subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True))
    (output / "resume.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(REPO)) + "\n" +
        shlex.join([sys.executable, "-u", str(SCRIPT), "--resume", str(output)]) + "\n")
    return output, config


def verify(output: Path, config: dict[str, Any]) -> None:
    """Resume only with identical frozen inputs, source code and native extension."""
    checked(output / "config.json", (output / "config.sha256").read_text().strip())
    if config["cwd"] != str(REPO):
        raise ValueError("Resume in the original checkout")
    for name, expected in config["input_sha256"].items():
        checked(output / "inputs" / name, expected)
    for name, expected in config["code_sha256"].items():
        checked(REPO / name, expected)
    source = json.loads((output / "inputs/source-config.json").read_text())
    if helpers.runtime_identity(source, False) != config["runtime"]:
        raise ValueError("Runtime changed; restore the original interpreter and native extension")


def collection_command(output: Path, attempt: Path, games: int, worker: int) -> list[str]:
    """Match the winning run's 400-simulation, full-start, solver34 collection."""
    return [sys.executable, "-u", str(REPO / BASE / "run_games.py"),
            "--checkpoint", str(output / "inputs/iter24.pt"), "--rows", "5", "--cols", "5",
            "--num_games", str(games // 4), "--game_index_offset", str(worker * (games // 4)),
            "--num_simulations", "400", "--num_workers", "2", "--solver_max_undrawn", "34",
            "--solver_node_budget", "50000", "--merge_equivalent", "0",
            "--c_puct", "1.5", "--leaf_batch_size", "16",
            "--policy_temperature", "1.0", "--prove_terminals", "1", "--shape_terminals", "1",
            "--margin_utility_lambda", "0.5", "--margin_utility_k", "6", "--stop-when-solved",
            "--seed", str(SEED + worker * 1000), "--save-name", str(attempt / "games")]


def collect(output: Path, attempt: Path) -> None:
    """Four child collectors inherit this bounded stage's process group."""
    config = json.loads((output / "config.json").read_text())
    workers = []
    streams = []
    try:
        for index in range(4):
            command = collection_command(output, attempt, config["games"], index)
            write_json(attempt / f"worker-{index}-command.json", command)
            stream = (attempt / f"worker-{index}.log").open("w")
            streams.append(stream)
            workers.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT))
        while True:
            statuses = [p.poll() for p in workers]
            if any(status not in (None, 0) for status in statuses):
                raise RuntimeError("Self-play collector failed; inspect worker logs")
            if all(status is not None for status in statuses):
                break
            count = len(list((attempt / "games").glob("*.npz")))
            print(f"Fresh self-play: {count}/{config['games']} games saved", flush=True)
            time.sleep(5)
        from alpha_go.boxes.proof_target_audit import audit_collection
        report = audit_collection(attempt / "games", expected_games=config["games"])
        write_json(attempt / "targets.json", report)
    finally:
        for process in workers:
            if process.poll() is None:
                process.terminate()
        for process in workers:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in streams:
            stream.close()


def stage(output: Path, name: str, seconds: int,
          command: Callable[[Path], list[str]],
          finish: Callable[[Path], None] | None = None) -> Path:
    """Preserve complete stages; retry incomplete stages in fresh, auditable directories."""
    folder = output / name
    folder.mkdir(exist_ok=True)
    receipt_path = folder / "completed.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        attempt = output / str(receipt["attempt"])
        actual = {str(p.relative_to(output)) for p in attempt.rglob("*") if p.is_file()}
        if actual != set(receipt["artifacts_sha256"]):
            raise ValueError(f"Changed completed stage file set: {name}")
        for path, expected in receipt["artifacts_sha256"].items():
            checked(output / path, expected)
        print(f"Reusing completed {name}", flush=True)
        return output / str(receipt["attempt"])
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=folder))
    cmd = command(attempt)
    write_json(attempt / "command.json", cmd)
    helpers.job(cmd, attempt / "run.log", seconds)
    if finish is not None:
        finish(attempt)
    receipt = dict(attempt=str(attempt.relative_to(output)), artifacts_sha256={
        str(p.relative_to(output)): digest(p) for p in attempt.rglob("*") if p.is_file()})
    write_json(receipt_path, receipt)
    return attempt


def probe_command(output: Path, attempt: Path, reference: Path, candidate: Path,
                  cache: Path) -> list[str]:
    """Identical fixed search for both policies; labels remain a development diagnostic."""
    return [sys.executable, "-u", "-m", "alpha_go.boxes.search_probe",
            "--reference", str(reference), "--candidate", str(candidate),
            "--positions-file", str(output / "inputs/positions.jsonl"),
            "--label-cache", str(cache), "--output", str(attempt / "probe"),
            "--rows", "5", "--cols", "5", "--min-undrawn", "35", "--max-undrawn", "38",
            "--mixes", "reference", "candidate", "--simulations", "400", "--lambdas", "0.5",
            "--solver", "34", "--solver-budget", "50000", "--label-budget", "20000000",
            "--policy-temperature", "1.0", "--margin-utility-k", "6", "--prove-terminals", "1",
            "--shape-terminals", "1", "--c-puct", "1.5", "--leaf-batch-size", "16",
            "--seed", str(SEED)]


def package(output: Path) -> None:
    """Export diagnostics; large reusable training data and checkpoints stay local."""
    archive = output / "results.tgz"
    temporary = output / "results.tgz.tmp"
    with tarfile.open(temporary, "w:gz") as stream:
        for path in sorted(output.rglob("*")):
            if (path.is_file() and not path.is_symlink() and path not in (archive, temporary)
                    and path.suffix not in (".pt", ".npz")
                    and path.name != ".lock"):
                stream.add(path, arcname=str(path.relative_to(output)), recursive=False)
    temporary.replace(archive)


def run(output: Path, config: dict[str, Any]) -> None:
    """Collect once, train matched arms, then read primary and incumbent comparisons."""
    from alpha_go.boxes.probe_report import build_report, write_report

    caps = config["caps_seconds"]
    collected = stage(output, "collect", caps["collect"], lambda attempt: [
        sys.executable, "-u", str(SCRIPT), "--collect-worker", str(output), str(attempt)])
    # The preparer owns resumable per-game receipts, so it has one stable output directory.
    prepared = output / "prepared"
    helpers.job([sys.executable, "-u", "-m", "alpha_go.boxes.all_policy_data",
                 "--games", str(collected / "games"), "--output", str(prepared),
                 "--exclude-positions", str(output / "inputs/exclude.jsonl"),
                 "--workers", "4", "--min-undrawn", "35", "--max-undrawn", "36",
                 "--node-budget", "20000000", "--child-node-budget", "20000000"],
                output / "prepare.log", caps["prepare"])
    trained = stage(output, "train", caps["train"], lambda attempt: [
        sys.executable, "-u", "-m", "alpha_go.boxes.policy_target_ab",
        "--reference", str(output / "inputs/iter24.pt"), "--prepared", str(prepared),
        "--output", str(attempt / "train"), "--epochs", "4", "--batch-size", "256",
        "--lr", "0.0001", "--min-undrawn", "24", "--seed", str(SEED), "--split-seed", "0"])
    checkpoints = trained / "train"
    summary: dict[str, Any] = dict(
        status="running", scope=config["scope"],
        preparation=json.loads((prepared / "preparation.json").read_text()),
        training=json.loads((checkpoints / "summary.json").read_text()), comparisons={})
    write_json(output / "summary.json", summary)
    cache = output / "inputs/labels.json"
    for name, reference in (("control-vs-exact", checkpoints / "control.pt"),
                            ("iter24-vs-exact", output / "inputs/iter24.pt")):
        def command(attempt: Path) -> list[str]:
            paths = [cache, *sorted((output / name).glob("attempt-*/probe/labels.json"))]
            merged = attempt / "labels.json"
            merge_caches(paths, merged)
            return probe_command(output, attempt, reference, checkpoints / "exact.pt", merged)
        def finish(attempt: Path) -> None:
            report = build_report(attempt / "probe", seed=SEED, resamples=10_000)
            write_report(report, attempt / "report")
        attempt = stage(output, name, caps[name], command, finish)
        report = json.loads((attempt / "report/report.json").read_text())
        summary["comparisons"][name] = report
        write_json(output / "summary.json", summary)
        cache = attempt / "probe/labels.json"
    summary["status"] = "complete"
    write_json(output / "summary.json", summary)
    print("Pilot complete. Read control-vs-exact first; iter24-vs-exact checks retention. "
          "No model was promoted.", flush=True)


def main() -> None:
    """Launch a new isolated pilot, or resume its immutable saved configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--previous-probe", type=Path,
                        default=REPO / BASE / "logs/proof24-bands-me7lvz_n")
    parser.add_argument("--collect-worker", type=Path, nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    helpers.pin_runtime()

    def interrupted(_signal: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    if args.collect_worker:
        collect(*args.collect_worker)
        return
    if args.games < 40 or args.games % 4:
        parser.error("--games must be at least 40 and divisible by 4")
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / "config.json").read_text())
    else:
        output, config = create(args)
    print(f"Run: {output}\nResume: bash {output / 'resume.sh'}\n"
          "Two matched training arms from proof34 iter24; no full-game arena.\n"
          "Stage limits total 4h45m plus shutdown; completed stages survive interruption.",
          flush=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(output, config)
        try:
            run(output, config)
        finally:
            package(output)
            print(f"Send this file: {output / 'results.tgz'}", flush=True)


if __name__ == "__main__":
    main()
