"""Wait for fresh Qdab collection, then run a bounded offline value experiment.

This controller does not play games, promote weights, or alter production models.
All subprocesses use the frozen experiment and checkpoint. Pending external-state
exclusions make the fit a development diagnostic, never a final strength result.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(path: Path) -> str:
    """Hash immutable experiment inputs without importing heavy libraries."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Publish durable progress atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checked(files: dict[str, str]) -> None:
    """Reject changed files rather than continuing another experiment."""
    for name, expected in files.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")


@contextmanager
def collection_lock(collection: Path) -> Iterator[None]:
    """Allow labeling readers while excluding a resumed collection writer."""
    with (collection / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def collection_ready(collection: Path, supervisor: Path, config_hash: str) -> bool:
    """Require successful collection and every frozen game slot before solving."""
    if digest(collection / "config.json") != config_hash:
        raise ValueError("Collection configuration changed while waiting")
    status = json.loads(supervisor.read_text())
    if status["status"] == "failed":
        raise RuntimeError("Collection failed; inspect its log before labeling")
    if status["status"] != "complete":
        if not Path(f"/proc/{int(status['supervisor_pid'])}").exists():
            raise RuntimeError("Collection supervisor disappeared without a final receipt")
        return False
    if status.get("returncode") != 0:
        raise RuntimeError("Collection did not finish successfully")
    config = json.loads((collection / "config.json").read_text())
    expected = {f"{slot['index']:04d}.json" for slot in config["schedule"]}
    actual = {p.name for p in (collection / "games").glob("*.json")}
    if actual != expected:
        raise RuntimeError("Collection stopped before all frozen games completed; resume it first")
    for slot in config["schedule"]:
        row = json.loads((collection / "games" / f"{slot['index']:04d}.json").read_text())
        if row["slot"] != slot or row["config_sha256"] != config_hash:
            raise ValueError("Saved game does not belong to the frozen collection")
    return True


def stage(command: list[str], output: Path, seconds: int, artifacts: list[Path]) -> None:
    """Bound a process group and retain logs; failures never mark a stage complete."""
    receipt = output / "stage.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["command"] != command:
            raise ValueError("Stage command changed on resume")
        if old["status"] == "complete":
            checked(old["output_sha256"])
            print(f"Reusing completed stage: {output.name}", flush=True)
            return
    started = time.time()
    write_json(output / "stage.json", dict(status="running", command=command,
                                           started_unix=started, limit_seconds=seconds))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (output / "run.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = process.wait(timeout=seconds)
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            write_json(output / "stage.json", dict(status="interrupted", command=command,
                       elapsed_seconds=time.time() - started, returncode=process.returncode))
            raise
    if result:
        write_json(receipt, dict(status="failed", command=command,
                   elapsed_seconds=time.time() - started, returncode=result))
        raise subprocess.CalledProcessError(result, command)
    files = [file for path in artifacts for file in
             (sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path])]
    if not files:
        raise ValueError("Completed stage produced no artifacts")
    hashes = {str(path): digest(path) for path in files}
    write_json(receipt, dict(status="complete", command=command,
               elapsed_seconds=time.time() - started, returncode=result, output_sha256=hashes))


def initialize(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Freeze recipe and code now; source-game snapshots are made after collection."""
    collection = args.collection.resolve()
    config = json.loads((collection / "config.json").read_text())
    checkpoint = Path(config["checkpoint"])
    checked({str(checkpoint): config["checkpoint_sha256"]})
    scripts = [HERE / name for name in ("queue_structural.py", "prepare_structural.py",
                                        "fit_structural.py", "structural_features.py",
                                        "structural_value.py")]
    scripts += list((REPO / "src/alpha_go").rglob("*.py"))
    extension = importlib.util.find_spec("alpha_go_cpp")
    if extension is None or extension.origin is None:
        raise ValueError("Native Boxes extension is missing")
    native = Path(extension.origin)
    checked({str(native): config["runtime"]["extension_sha256"]})
    scripts.append(native)
    code = {str(path): digest(path) for path in scripts}
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="structural-queue-", dir=args.output_root)).resolve()
    recipe = dict(schema=1, collection=str(collection), supervisor=str(args.supervisor.resolve()),
                  collection_config_sha256=digest(collection / "config.json"),
                  checkpoint=str(checkpoint), checkpoint_sha256=config["checkpoint_sha256"],
                  code_sha256=code, python=sys.executable, wait_seconds=4 * 3600,
                  label_seconds=2700, fit_seconds=1800, steps=1024, seed=20261124,
                  scope="Offline development fit; actual GPU benchmark-state exclusions pending. "
                        "No production model change, arena evaluation or promotion.")
    write_json(output / "recipe.json", recipe)
    (output / "recipe.sha256").write_text(digest(output / "recipe.json") + "\n")
    (output / "resume.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(REPO)) + "\n"
        + shlex.join([sys.executable, "-u", str(HERE / "queue_structural.py"),
                      "--resume", str(output)]) + "\n")
    return output, recipe


def run(output: Path, recipe: dict[str, Any]) -> None:
    """Wait without CPU work, then serialize solver work and small-head fitting."""
    checked({str(output / "recipe.json"): (output / "recipe.sha256").read_text().strip(),
             recipe["checkpoint"]: recipe["checkpoint_sha256"], **recipe["code_sha256"]})
    collection = Path(recipe["collection"])
    deadline = time.monotonic() + recipe["wait_seconds"]
    write_json(output / "state.json", dict(status="waiting_for_collection", pid=os.getpid(),
                                           scope=recipe["scope"]))
    while not collection_ready(collection, Path(recipe["supervisor"]),
                               recipe["collection_config_sha256"]):
        if time.monotonic() >= deadline:
            raise TimeoutError("Collection wait budget exhausted; no labels or fit started")
        print("Waiting for all frozen Qdab games; no solver or fitting work running.", flush=True)
        time.sleep(30)
    # Hold both locks throughout heavy work so a new timed Qdab query cannot overlap.
    with collection_lock(collection), \
            Path("/tmp/autoboxes-qdab-12345.lock").open("a") as qdab_lock:
        # The preparer also takes a shared source lock; exclusive would deadlock
        # our own child. Both shared locks exclude a resumed collection writer.
        fcntl.flock(qdab_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked(recipe["code_sha256"])
        write_json(output / "state.json", dict(status="labeling", scope=recipe["scope"]))
        stage([recipe["python"], "-u", str(HERE / "prepare_structural.py"),
               "--collection", str(collection), "--output", str(output / "prepared"),
               "--seconds", str(recipe["label_seconds"]), "--workers", "4",
               "--node-budget", "20000000"], output / "label", recipe["label_seconds"] + 300,
              [output / "prepared" / name
               for name in ("manifest.json", "states.jsonl", "roots.jsonl")])
        write_json(output / "state.json", dict(status="fitting", scope=recipe["scope"]))
        stage([recipe["python"], "-u", str(HERE / "fit_structural.py"),
               "--prepared", str(output / "prepared"), "--checkpoint", recipe["checkpoint"],
               "--expected-checkpoint-sha256", recipe["checkpoint_sha256"],
               "--output", str(output / "fit"), "--steps", str(recipe["steps"]),
               "--batch-size", "256", "--lr", "0.001", "--device", "cpu", "--threads", "1",
               "--seed", str(recipe["seed"]), "--development-with-pending-exclusions"],
              output / "fitting", recipe["fit_seconds"], [output / "fit"])
    write_json(output / "state.json", dict(status="complete", scope=recipe["scope"]))


def main() -> None:
    """Create or resume a frozen queue without requiring the GPU machine."""
    def terminate(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt("SIGTERM: stopping active stage and preserving completed work")

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--supervisor", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO / "work/structural-value")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        output = args.resume.resolve()
        recipe = json.loads((output / "recipe.json").read_text())
    else:
        if args.collection is None or args.supervisor is None:
            parser.error("--collection and --supervisor are required for a fresh queue")
        output, recipe = initialize(args)
    print(f"Queue: {output}\nResume: bash {output / 'resume.sh'}", flush=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(output, recipe)
        except BaseException as error:
            write_json(output / "state.json", dict(status="stopped",
                       error_type=type(error).__name__, error=str(error), scope=recipe["scope"]))
            raise
        finally:
            temporary = output / "results.tmp.tgz"
            with tarfile.open(temporary, "w:gz") as archive:
                for path in sorted(output.rglob("*")):
                    if path.is_file() and path.suffix not in (".tgz", ".npz", ".lock"):
                        archive.add(path, arcname=str(path.relative_to(output)))
            temporary.replace(output / "results.tgz")
            print(f"Results: {output / 'results.tgz'}", flush=True)


if __name__ == "__main__":
    main()
