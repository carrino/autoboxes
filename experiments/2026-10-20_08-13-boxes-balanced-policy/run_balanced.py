"""Compare replay sampling on the completed 2,000-game exact-policy collection."""
from __future__ import annotations

import argparse
import fcntl
import importlib
import importlib.util
import json
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
BASE = Path("experiments/2026-10-08_14-42-boxes-3x3-loop")
SOURCE = BASE / "logs/fresh35-36-anmyR2"
PILOT = REPO / "experiments/2026-10-18_20-07-boxes-all-policy-pilot/pilot.py"
spec = importlib.util.spec_from_file_location("all_policy_stage_helpers", PILOT)
assert spec and spec.loader
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
helpers = pilot.helpers
digest, checked, write_json = helpers.digest, helpers.checked, helpers.write_json
REFERENCE_SHA = "73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d"
SEED = 20261020
CAPS = {"train": 3600, "control-vs-balanced": 3600, "iter24-vs-balanced": 3600}


def runtime_identity(source: dict[str, Any], cpu: bool) -> dict[str, Any]:
    """Require the collector's search sources and binary, without rebuilding."""
    import torch

    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; --cpu is only for explicit CPU diagnostics")
    original = Path(source["cwd"])
    native_hashes = []
    checked_sources = 0
    for name, expected in source["hashes"].items():
        path = Path(name)
        if path.name.startswith("alpha_go_cpp.") and path.suffix == ".so":
            native_hashes.append(expected)
        if path.is_relative_to(original / "src/alpha_go"):
            checked(REPO / path.relative_to(original), expected)
            checked_sources += 1
    if len(native_hashes) != 1 or checked_sources == 0:
        raise ValueError("Collection config lacks search-source or native-binary provenance")
    modules = {}
    for name in ("alpha_go", "alpha_go.boxes.balanced_policy",
                 "alpha_go.boxes.search_probe", "alpha_go.boxes.probe_report",
                 "alpha_go.boxes.policy_target_ab"):
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError(f"Imported module lacks a source file: {name}")
        actual = Path(module.__file__).resolve()
        relative = "alpha_go/__init__.py" if name == "alpha_go" else name.replace(".", "/") + ".py"
        if actual != REPO / "src" / relative:
            raise ValueError(f"Wrong checkout imported: {name}: {actual}")
        modules[name] = {"path": str(actual), "sha256": digest(actual)}
    native = importlib.import_module("alpha_go_cpp")
    if native.__file__ is None:
        raise ValueError("Imported native extension lacks a file")
    extension = Path(native.__file__).resolve()
    checked(extension, native_hashes[0])
    return dict(interpreter=sys.executable, prefix=sys.prefix, python=sys.version,
                torch=torch.__version__, extension=str(extension),
                extension_sha256=digest(extension), modules=modules,
                device="cpu" if cpu else "cuda")


def create(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Verify the saved data, then freeze this comparison's checkpoint and protocol."""
    from alpha_go.boxes.policy_target_ab import bundle_files

    source_run = args.source_run.resolve()
    source_path = source_run / "config.json"
    source = json.loads(source_path.read_text())
    if source["future_validation"] != {
        "family_fraction": .1, "split_seed": 0, "require_D4_transposition_exclusion": True
    }:
        raise ValueError("Unexpected predeclared collection validation protocol")
    checkpoint = source_run / "inputs/iter24.pt"
    checked(checkpoint, REFERENCE_SHA)
    runtime = runtime_identity(source, args.cpu)
    prepared = source_run / "prepared"
    print("Verifying the saved preparation files, including every NPZ...", flush=True)
    prepared_hashes = bundle_files(prepared)
    metadata = json.loads((prepared / "preparation.json").read_text())
    settings = metadata["settings"]
    if (settings["rows"], settings["cols"], settings["min_undrawn"],
            settings["max_undrawn"]) != (5, 5, 35, 36):
        raise ValueError("Expected the completed 5x5, 35..36 preparation")
    logs = REPO / BASE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="balanced-policy-", dir=logs))
    inputs = output / "inputs"
    inputs.mkdir()
    shutil.copyfile(checkpoint, inputs / "iter24.pt")
    checked(inputs / "iter24.pt", REFERENCE_SHA)
    shutil.copyfile(source_path, inputs / "collection-config.json")
    write_json(inputs / "prepared-hashes.json", prepared_hashes)
    code = [SCRIPT, SCRIPT.with_name("run_balanced.sh"), PILOT, pilot.HELPER,
            *(REPO / "src/alpha_go").rglob("*.py")]
    config = dict(
        schema=1, cwd=str(REPO), source_run=str(source_run), prepared=str(prepared),
        runtime=runtime, cpu=args.cpu, seed=SEED, caps_seconds=CAPS,
        training={"steps": 2048, "batch_size": 256, "lr": .0001, "root_fraction": .25},
        input_sha256={p.name: digest(p) for p in inputs.iterdir()},
        code_sha256={str(p.relative_to(REPO)): digest(p) for p in code},
        scope="Fixed ordinary-vs-balanced replay comparison. Same exact dataset and checkpoint, "
              "2,048 optimizer updates per arm. Fresh family-held-out validation after D4 "
              "exclusion and deduplication. Final checkpoints are primary; intermediate reads "
              "are diagnostic. This is development selection, not an arena promotion.")
    write_json(output / "config.json", config)
    (output / "config.sha256").write_text(digest(output / "config.json") + "\n")
    (output / "code-commit.txt").write_text(subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True))
    (output / "resume.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(REPO)) + "\n" +
        shlex.join([sys.executable, "-u", str(SCRIPT), "--resume", str(output)]) + "\n")
    return output, config


def verify(output: Path, config: dict[str, Any]) -> None:
    """Reject changed sources, runtime or data before resuming completed stages."""
    from alpha_go.boxes.policy_target_ab import bundle_files

    checked(output / "config.json", (output / "config.sha256").read_text().strip())
    if config["cwd"] != str(REPO):
        raise ValueError("Resume in the original checkout")
    for name, expected in config["code_sha256"].items():
        checked(REPO / name, expected)
    for name, expected in config["input_sha256"].items():
        checked(output / "inputs" / name, expected)
    source = json.loads((output / "inputs/collection-config.json").read_text())
    if runtime_identity(source, config["cpu"]) != config["runtime"]:
        raise ValueError("Runtime changed; restore the original interpreter and native extension")
    if bundle_files(Path(config["prepared"])) != json.loads(
            (output / "inputs/prepared-hashes.json").read_text()):
        raise ValueError("Prepared training data changed")


def train_command(output: Path, attempt: Path, config: dict[str, Any]) -> list[str]:
    """Use a fixed optimizer budget, rather than a speed-dependent time budget."""
    settings = config["training"]
    command = [
        sys.executable, "-u", "-m", "alpha_go.boxes.balanced_policy",
        "--reference", str(output / "inputs/iter24.pt"), "--prepared", config["prepared"],
        "--output", str(attempt / "train"), "--steps", str(settings["steps"]),
        "--batch-size", str(settings["batch_size"]), "--lr", str(settings["lr"]),
        "--root-fraction", str(settings["root_fraction"]), "--seed", str(config["seed"])]
    return command + (["--cpu"] if config["cpu"] else [])


def probe_command(attempt: Path, trained: Path, reference: Path, cache: Path,
                  config: dict[str, Any]) -> list[str]:
    """Read all exported fresh validation roots with identical search budgets."""
    command = [
        sys.executable, "-u", "-m", "alpha_go.boxes.search_probe",
        "--reference", str(reference), "--candidate", str(trained / "balanced.pt"),
        "--positions-file", str(trained / "validation-positions.jsonl"),
        "--label-cache", str(cache), "--output", str(attempt / "probe"),
        "--rows", "5", "--cols", "5", "--min-undrawn", "35", "--max-undrawn", "36",
        "--mixes", "reference", "candidate", "--simulations", "400", "--lambdas", "0.5",
        "--solver", "34", "--solver-budget", "50000", "--label-budget", "20000000",
        "--c-puct", "1.5", "--leaf-batch-size", "16", "--policy-temperature", "1.0",
        "--margin-utility-k", "6", "--prove-terminals", "1", "--shape-terminals", "1",
        "--seed", str(config["seed"])]
    return command + (["--cpu"] if config["cpu"] else [])


def run(output: Path, config: dict[str, Any]) -> None:
    """Preserve completed training and probes; package partial results on interruption."""
    from alpha_go.boxes.probe_report import build_report, write_report

    caps = config["caps_seconds"]
    attempt = pilot.stage(output, "train", caps["train"],
                          lambda folder: train_command(output, folder, config))
    trained = attempt / "train"
    summary: dict[str, Any] = dict(
        status="running", scope=config["scope"],
        training=json.loads((trained / "summary.json").read_text()), comparisons={})
    write_json(output / "summary.json", summary)
    cache = trained / "validation-labels.json"
    for name, reference in (("control-vs-balanced", trained / "control.pt"),
                            ("iter24-vs-balanced", output / "inputs/iter24.pt")):
        def command(folder: Path) -> list[str]:
            paths = [cache, *sorted((output / name).glob("attempt-*/probe/labels.json"))]
            merged = folder / "input-labels.json"
            pilot.merge_caches(paths, merged)
            return probe_command(folder, trained, reference, merged, config)

        def finish(folder: Path) -> None:
            report = build_report(folder / "probe", seed=config["seed"], resamples=10_000)
            write_report(report, folder / "report")

        attempt = pilot.stage(output, name, caps[name], command, finish)
        summary["comparisons"][name] = json.loads((attempt / "report/report.json").read_text())
        write_json(output / "summary.json", summary)
        cache = attempt / "probe/labels.json"
    summary["status"] = "complete"
    write_json(output / "summary.json", summary)
    print("Comparison complete. Read control-vs-balanced first, then iter24-vs-balanced. "
          "The incumbent is unchanged; a full-game match is still required.", flush=True)


def main() -> None:
    """Start or resume a bounded comparison on the existing saved labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=REPO / SOURCE)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    helpers.pin_runtime()
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / "config.json").read_text())
    else:
        output, config = create(args)
    print(f"Run: {output}\nResume: bash {output / 'resume.sh'}\n"
          "Reuses completed labels. Two 2,048-step training arms, then two paired "
          "validation probes. Each stage has a one-hour cap, plus setup/reporting.", flush=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def terminate(_signal: int, _frame: Any) -> None:
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        try:
            verify(output, config)
            run(output, config)
            verify(output, config)
        finally:
            pilot.package(output)
            print(f"Send this file: {output / 'results.tgz'}", flush=True)


if __name__ == "__main__":
    main()
