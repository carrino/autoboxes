"""Short matched Qdab-policy learning screen, with no games or promotion."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import io
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
from pathlib import Path, PurePosixPath
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
BASE = REPO / "experiments/2026-10-08_14-42-boxes-3x3-loop"
CHECKPOINT_SHA = "c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f"


def read_json(path: Path) -> Any:
    """Read a saved receipt."""
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    """Publish a finite JSON receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    """Hash immutable input and output bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    """Inventory regular files and reject source links."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = digest(path)
    return result


def unpack_payload(source: Path | None, destination: Path) -> None:
    """Copy explicit data or safely extract the bundled SHA-verified archive."""
    if source is not None:
        tree_hashes(source)
        shutil.copytree(source, destination)
    else:
        data = SCRIPT.parent / "data"
        encoded = (data / "teacher-screen-payload.tgz.b64").read_text()
        blob = base64.b64decode("".join(encoded.split()), validate=True)
        expected = (data / "teacher-screen-payload.sha256").read_text().split()[0]
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError("Bundled screen payload failed SHA256 verification")
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
            members = archive.getmembers()
            names = set()
            for member in members:
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or not (member.isfile() or member.isdir())
                    or str(name) in names
                ):
                    raise ValueError(f"Unsafe or duplicate payload member: {member.name}")
                names.add(str(name))
            if sum(member.size for member in members) > 100_000_000:
                raise ValueError("Unexpectedly large screen payload")
            destination.mkdir(parents=True)
            archive.extractall(destination, filter="data")
    if not (destination / "manifest.json").is_file():
        raise ValueError("Payload must contain manifest.json at its root")


def runtime() -> dict[str, Any]:
    """Fingerprint Python, native extension and the inference/training device."""
    import platform

    import alpha_go_cpp  # type: ignore[import-not-found]
    import numpy as np
    import torch

    return dict(
        python=sys.version,
        torch=torch.__version__,
        numpy=np.__version__,
        platform=platform.platform(),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        extension_sha256=digest(Path(alpha_go_cpp.__file__)),
    )


def create(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Freeze the exact inputs and fixed-endpoint recipe before fitting."""
    import torch

    from alpha_go.boxes.policy_target_ab import bundle_files

    if not (
        args.steps > 0
        and args.batch_size >= 2
        and 0 < args.teacher_per_batch < args.batch_size
        and math.isfinite(args.lr)
        and args.lr > 0
        and args.seed >= 0
        and args.min_undrawn >= 0
    ):
        raise ValueError("Require positive steps/lr and 0 < teacher-per-batch < batch-size")
    checkpoint, replay = args.checkpoint.resolve(), args.replay.resolve()
    if digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Expected the unchanged ordinary-four cycle-4 checkpoint")
    if not args.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA required; --cpu is an explicit small-smoke override")
    replay_hashes = bundle_files(replay)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="qdab-teacher-screen-", dir=args.output_root)).resolve()
    inputs = output / "inputs"
    inputs.mkdir()
    shutil.copy2(checkpoint, inputs / "reference.pt")
    if digest(inputs / "reference.pt") != CHECKPOINT_SHA:
        raise ValueError("Checkpoint changed while copying")
    unpack_payload(args.payload.resolve() if args.payload else None, inputs / "payload")
    if read_json(inputs / "payload/manifest.json")["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Teaching payload belongs to a different starting checkpoint")
    write_json(inputs / "replay-hashes.json", replay_hashes)
    sources = [
        SCRIPT,
        SCRIPT.with_suffix(".sh"),
        SCRIPT.with_name("fit_teacher_screen.py"),
        SCRIPT.with_name("prepare_teacher_screen.py"),
        SCRIPT.with_name("prepare_teacher.py"),
        *sorted((REPO / "src/alpha_go").rglob("*.py")),
    ]
    settings = {
        key: getattr(args, key)
        for key in ("steps", "batch_size", "teacher_per_batch", "lr", "seed", "cpu", "min_undrawn")
    }
    config = dict(
        schema=1,
        checkpoint_sha256=CHECKPOINT_SHA,
        replay=str(replay),
        settings=settings,
        input_sha256=tree_hashes(inputs),
        runtime=runtime(),
        code_sha256={str(path.relative_to(REPO)): digest(path) for path in sources},
        fit_limit_seconds=5400,
        scope="Fixed short learning screen: reference raw-policy distillation "
        "versus approximate Qdab targets on identical teacher states. "
        "Replay-only value loss; no games, snapshot selection or promotion.",
    )
    write_json(output / "config.json", config)
    (output / "config.sha256").write_text(digest(output / "config.json") + "\n")
    (output / "code-commit.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True)
    )
    command = shlex.join(["bash", str(SCRIPT.with_suffix(".sh")), "--resume", str(output)])
    (output / "resume.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd "
        + shlex.quote(str(REPO))
        + "\n"
        + command
        + "\n"
    )
    return output, config


def verify(output: Path, config: dict[str, Any]) -> None:
    """Reject changed recipe, source, runtime or data on every resume."""
    from alpha_go.boxes.policy_target_ab import bundle_files

    if digest(output / "config.json") != (output / "config.sha256").read_text().strip():
        raise ValueError("Screen configuration changed")
    if (
        config["checkpoint_sha256"] != CHECKPOINT_SHA
        or digest(output / "inputs/reference.pt") != CHECKPOINT_SHA
    ):
        raise ValueError("Frozen reference checkpoint changed")
    if tree_hashes(output / "inputs") != config["input_sha256"]:
        raise ValueError("Frozen screen inputs changed")
    if bundle_files(Path(config["replay"])) != read_json(output / "inputs/replay-hashes.json"):
        raise ValueError("Source replay changed")
    if runtime() != config["runtime"]:
        raise ValueError("Runtime changed; use the same Python, GPU and native extension")
    for name, expected in config["code_sha256"].items():
        if digest(REPO / name) != expected:
            raise ValueError(f"Screen source changed: {name}")


def fit_command(output: Path, config: dict[str, Any]) -> list[str]:
    """Build the fixed fitter invocation from its frozen recipe."""
    command = [
        sys.executable,
        "-u",
        str(SCRIPT.with_name("fit_teacher_screen.py")),
        "--reference",
        str(output / "inputs/reference.pt"),
        "--replay",
        config["replay"],
        "--payload",
        str(output / "inputs/payload"),
        "--output",
        str(output / "fit"),
    ]
    for name, value in config["settings"].items():
        if name == "cpu":
            if value:
                command.append("--cpu")
        else:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    return command


def fit_job(command: list[str], log: Path, cap_seconds: int) -> None:
    """Run a bounded process group and retain completed fitter arms on interruption."""
    print("Running: " + shlex.join(command) + f"\nLog: {log}", flush=True)
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(
            command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= cap_seconds:
                    raise TimeoutError(f"Screen exceeded {cap_seconds / 60:g} minutes: {log}")
                try:
                    process.wait(timeout=min(30, cap_seconds - elapsed))
                except subprocess.TimeoutExpired:
                    lines = log.read_text(errors="replace").splitlines()
                    print(
                        f"fit: {time.monotonic() - started:.0f}s; "
                        + (lines[-1] if lines else "starting"),
                        flush=True,
                    )
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


def run(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Reuse a complete fit or resume only its incomplete arms."""
    completed = output / "fit-completed.json"
    folder = output / "fit"
    if completed.exists():
        receipt = read_json(completed)
        if (
            receipt["config_sha256"] != digest(output / "config.json")
            or tree_hashes(folder) != receipt["artifact_sha256"]
        ):
            raise ValueError("Completed fit changed")
        print("Reusing completed fits and readouts.", flush=True)
    else:
        attempts = output / "attempts"
        attempts.mkdir(exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix="fit-", dir=attempts))
        command = fit_command(output, config)
        write_json(attempt / "command.json", command)
        fit_job(command, attempt / "run.log", config["fit_limit_seconds"])
        result = read_json(folder / "summary.json")
        if result["status"] != "complete":
            raise ValueError("Fitter exited without completing both fixed endpoints")
        write_json(
            completed,
            dict(config_sha256=digest(output / "config.json"), artifact_sha256=tree_hashes(folder)),
        )
    summary = read_json(folder / "summary.json")
    write_json(output / "summary.json", summary)
    print("Learning screen complete. No matches or automatic promotion were run.", flush=True)
    return dict(summary)


def package(output: Path) -> None:
    """Export small reports and readouts while retaining weights and replay locally."""
    temporary = output / "results.tmp.tgz"
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(output.rglob("*")):
            relative = path.relative_to(output)
            if (
                not path.is_file()
                or path.is_symlink()
                or relative.parts[0] == "inputs"
                or path.suffix in {".pt", ".npz", ".npy", ".tgz"}
            ):
                continue
            archive.add(path, arcname=str(relative), recursive=False)
    temporary.replace(output / "results.tgz")


def main() -> None:
    """Start a short screen or resume the original frozen run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, help="explicit unpacked payload; default bundled")
    parser.add_argument(
        "--checkpoint", type=Path, default=BASE / "logs/qdab-match-1yafbn9k/inputs/checkpoint.pt"
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=BASE / "logs/ordinary-four-aqowx5f8/cycle-4-prepare/attempt-tbyadiie/prepared",
    )
    parser.add_argument("--output-root", type=Path, default=BASE / "logs")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teacher-per-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261128)
    parser.add_argument("--min-undrawn", type=int, default=24)
    parser.add_argument("--cpu", action="store_true", help="explicit small CPU smoke only")
    args = parser.parse_args()
    os.chdir(REPO)
    if args.resume:
        output = args.resume.resolve()
        config = read_json(output / "config.json")
    else:
        output, config = create(args)
    print(
        f"Run: {output}\nResume: bash {output / 'resume.sh'}\n"
        f"Two fixed {config['settings']['steps']}-update fits; "
        f"{config['settings']['teacher_per_batch']}/{config['settings']['batch_size']} "
        "teacher rows per batch. No Qdab games.\n"
        "Control uses the frozen network's raw policy on teacher states; treatment uses "
        "Qdab moves. Keep checkout/build unchanged until completion.",
        flush=True,
    )

    def terminate(_signal: int, _frame: Any) -> None:
        raise KeyboardInterrupt("Stopping safely; use the printed resume command")

    signal.signal(signal.SIGTERM, terminate)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify(output, config)
            run(output, config)
            verify(output, config)
        finally:
            package(output)
            print(f"Send this file: {output / 'results.tgz'}", flush=True)


if __name__ == "__main__":
    main()
