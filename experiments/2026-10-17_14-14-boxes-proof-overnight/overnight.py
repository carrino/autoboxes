"""Fresh proof-target self-play with a rebuilt solver and a matched final arena."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
FINAL_SEED = 20261029
BASE_EXP = Path("experiments/2026-10-08_14-42-boxes-3x3-loop")
ITER0_SHA = "46c7264c4a0e9c7efa1cfccbb1b511c1c5faa3824fb69a62705d8cbfb1f953f0"


def file_sha(path: Path) -> str:
    """Hash build inputs without importing the package before its extension is rebuilt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin_runtime() -> None:
    """Use this checkout and this venv in the controller and all uv children."""
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    os.environ.update(PYTHONPATH=str(REPO / "src"), PYTHONNOUSERSITE="1",
                      UV_PROJECT_ENVIRONMENT=sys.prefix,
                      PATH=str(Path(sys.prefix) / "bin") + os.pathsep + os.environ["PATH"],
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def controller() -> Any:
    """Reuse the tested timed controller, including its interruption recovery."""
    path = REPO / "experiments/2026-10-16_21-41-boxes-solver34-loop/overnight.py"
    spec = importlib.util.spec_from_file_location("solver34_controller", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_identity() -> dict[str, Any]:
    """Identify the actual interpreter, extension and imported training modules."""
    from alpha_go.boxes.curriculum import digest

    modules = {}
    for name in ("alpha_go", "alpha_go.gameplay", "alpha_go.boxes.nn_agent",
                 "alpha_go.boxes.dataset", "alpha_go.boxes.model",
                 "alpha_go.boxes.proof_target_audit"):
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError(f"Imported module has no source path: {name}")
        path = Path(module.__file__).resolve()
        relative = name.replace(".", "/")
        expected = REPO / "src" / (relative + ".py")
        if name == "alpha_go":
            expected = REPO / "src/alpha_go/__init__.py"
        if path != expected:
            raise ValueError(f"Wrong checkout imported: {name}: {path}; expected {expected}")
        modules[name] = str(path)
    native = importlib.import_module("alpha_go_cpp")
    assert native.__file__ is not None
    extension = Path(native.__file__).resolve()
    return dict(interpreter=sys.executable, prefix=sys.prefix, python=sys.version,
                modules=modules, extension=str(extension), extension_sha256=digest(extension))


def build_inputs() -> dict[str, str]:
    """Hash tracked native sources and the build/dependency configuration."""
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "src/alpha_go/cpp", "scripts/build_cpp.sh",
         "pyproject.toml", "uv.lock"], text=True).split("\0")
    return {str((REPO / name).resolve()): file_sha(REPO / name) for name in names if name}


def build_and_check(output: Path) -> dict[str, Any]:
    """Build before importing the extension, then test the real target-saving path."""
    if "alpha_go_cpp" in sys.modules:
        raise RuntimeError("Extension already loaded before rebuild; start a fresh process")
    inputs = build_inputs()
    print(f"Rebuilding current C++ solver; log: {output / 'build.log'}", flush=True)
    with (output / "build.log").open("w") as log:
        subprocess.run(["bash", "scripts/build_cpp.sh"], stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    from alpha_go.boxes.curriculum import checked_files, digest, write_json

    checked_files(inputs)
    runtime = runtime_identity()
    binaries = list((REPO / "src/alpha_go/cpp/build").glob("alpha_go_cpp*.so"))
    if len(binaries) != 1 or digest(binaries[0]) != runtime["extension_sha256"]:
        raise ValueError("Imported extension does not match the freshly built solver")
    print("Checking proof targets through game recording, NPZ and dataset loading; "
          f"log: {output / 'checks.log'}", flush=True)
    with (output / "checks.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "pytest", "-q",
                        "tests/test_boxes_proof_policy_targets.py",
                        "tests/test_boxes_solver_canonical_bytes.py"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    checked_files(inputs)
    receipt = dict(runtime=runtime, build_inputs_sha256=inputs,
                   policy_target_semantics="proof_filtered_visits_v1")
    write_json(output / "preflight.json", receipt)
    print("Preflight passed: current solver loaded and corrected policy targets verified.",
          flush=True)
    return receipt


def verify_preflight(config: dict[str, Any]) -> None:
    """Resuming must use the same imported binary and the verified frozen iter0."""
    from alpha_go.boxes.curriculum import checked_files, digest

    expected = config["preflight"]
    checked_files(expected["build_inputs_sha256"])
    if runtime_identity() != expected["runtime"]:
        raise ValueError("Runtime changed since preflight; restore the original venv/extension")
    if digest(Path(config["reference"])) != ITER0_SHA:
        raise ValueError("This experiment requires the verified frozen iter0 checkpoint")


def main() -> None:
    """Start a new eight-hour run, or resume without rebuilding frozen inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--final-games", type=int, default=200)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="CPU pipeline testing only")
    args = parser.parse_args()
    pin_runtime()

    def interrupted(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / "config.json").read_text())
        if Path(config["cwd"]) != REPO:
            raise ValueError("Resume this run in its original checkout")
    else:
        if not 0 < args.hours <= 168 or args.final_games <= 0 or args.final_games % 2:
            raise ValueError("Require 0 < hours <= 168 and positive even final-games")
        import torch
        if not args.cpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no training or build started")
        choices = [REPO / "work/solver34-vs30-Qeqlwh/reference.pt",
                   REPO / "experiments/2026-10-15_20-57-boxes-qdab-curriculum/logs/"
                          "qdab-20261016-044033/reference.pt",
                   REPO / BASE_EXP / "logs/astra-probe-PCcE6W/reference.pt",
                   REPO / BASE_EXP / "checkpoints/5x5-solver34-8pt6ai3d/iter0.pt"]
        reference = args.reference or next((p for p in choices if p.is_file()), None)
        if reference is None or not reference.is_file():
            raise FileNotFoundError("Supply --reference /path/to/frozen/iter0.pt")
        if file_sha(reference) != ITER0_SHA:
            raise ValueError("Reference differs from verified iter0; "
                             "candidate selection is pending")
        args.reference = reference
        (REPO / "work").mkdir(exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="proof-preflight-", dir=REPO / "work"))
        preflight = build_and_check(scratch)
        if args.check_only:
            print(f"Checks saved: {scratch}; no training started", flush=True)
            return
        runner = controller()
        output, config = runner.create(args, tag_prefix="5x5-proof34-",
                                       final_seed=FINAL_SEED, resume_script=SCRIPT)
        config["preflight"] = preflight
        config["environment"].update(PROOF_TARGET_AUDIT="1", PYTHONPATH=str(REPO / "src"),
                                     PYTHONNOUSERSITE="1", UV_PROJECT_ENVIRONMENT=sys.prefix)
        config["scope"] = "Fresh proof-filtered policy targets with exact-equivalent faster solver"
        runner.write_json(output / "config.json", config)
        (output / "config.sha256").write_text(runner.digest(output / "config.json") + "\n")
        shutil.copytree(scratch, output / "preflight")
    runner = controller()
    verify_preflight(config)
    with (output / "overnight.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.run(output, config)


if __name__ == "__main__":
    main()
