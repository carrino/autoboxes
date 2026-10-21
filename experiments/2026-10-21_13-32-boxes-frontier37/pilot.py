"""Continue iter24 with exact mistakes through 37, then test the fixed last learner."""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import json
import shutil
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
BASE = Path("experiments/2026-10-08_14-42-boxes-3x3-loop")
BALANCED = BASE / "logs/balanced-policy-k3s3hb6v/inputs"
REFERENCE_SHA = "73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d"
SOURCE_SHA = "52aefca8816e44d34bd030412c1cf00f4bb1123b57ff9e255858dd58c6f7b38f"
FINAL_SEED = 20261103


def load(name: str, relative: str) -> Any:
    """Reuse the tested collection, timed-loop and bounded-evaluation controllers."""
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


balanced = load("frontier_balanced_helpers",
                "experiments/2026-10-20_08-13-boxes-balanced-policy/run_balanced.py")
runner = load("frontier_timed_loop",
              "experiments/2026-10-16_21-41-boxes-solver34-loop/overnight.py")
stages, helpers = balanced.pilot, balanced.helpers
digest, checked, write_json = helpers.digest, helpers.checked, helpers.write_json


def create(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Freeze verified iter24, runtime, diagnostic families and a new isolated league."""
    reference = args.reference or REPO / BALANCED / "iter24.pt"
    checked(reference, REFERENCE_SHA)
    source_path = REPO / BALANCED / "collection-config.json"
    checked(source_path, SOURCE_SHA)
    source = json.loads(source_path.read_text())
    runtime = balanced.runtime_identity(source, args.cpu)
    for name, expected in helpers.INPUT_HASHES.items():
        checked(REPO / helpers.INPUTS / name, expected)
    monitors = {}
    excluded = []
    for name, minimum, maximum in (("35-36", 35, 36), ("37", 37, 37)):
        original = "positions-35-36.jsonl" if minimum == 35 else "positions-37-38.jsonl"
        rows = [json.loads(line) for line in
                (REPO / helpers.INPUTS / original).read_text().splitlines() if line.strip()]
        excluded.extend(rows)  # Keep the unused 38 roots out as well.
        selected = [r for r in rows if minimum <= 60-len(r["prefix"]) <= maximum]
        if not selected:
            raise ValueError(f"Empty diagnostic band: {name}")
        monitors[name] = "".join(json.dumps(r) + "\n" for r in selected)
    if any((r.get("rows"), r.get("cols")) != (5, 5)
           or not isinstance(r.get("family"), str) or not r["family"] for r in excluded):
        raise ValueError("Frozen exclusions must identify 5x5 diagnostic families")
    exclusions = "".join(json.dumps(r) + "\n" for r in excluded)
    args.reference = reference
    output, config = runner.create(args, tag_prefix="5x5-frontier37-", final_seed=FINAL_SEED,
                                   resume_script=SCRIPT, reference_label="proof34 iter24")
    inputs = output / "inputs"
    inputs.mkdir()
    shutil.copyfile(source_path, inputs / "collection-config.json")
    shutil.copyfile(REPO / helpers.INPUTS / "labels.json", inputs / "labels.json")
    (inputs / "exclude.jsonl").write_text(exclusions)
    for name, text in monitors.items():
        (inputs / f"positions-{name}.jsonl").write_text(text)
    config["environment"].update(
        POLICY_MIN_UNDRAWN="35", POLICY_MAX_UNDRAWN="37", POLICY_MAX_PER_GAME="3",
        POLICY_EXCLUDE_POSITIONS=str(inputs / "exclude.jsonl"), PROOF_TARGET_AUDIT="1",
        PYTHONPATH=str(REPO / "src"), PYTHONNOUSERSITE="1", UV_PROJECT_ENVIRONMENT=sys.prefix)
    config.update(runtime=runtime, finish_started_iteration=True, final_workers=2,
                  diagnostic_caps_seconds={"35-36": 7200, "37": 7200},
                  input_sha256={p.name: digest(p) for p in inputs.iterdir()},
                  scope="Exploratory continuation, exact mistakes 35..37 with priority to "
                        "win-to-loss moves. Final LAST-vs-iter24 arena is primary; retained "
                        "development bands check regression. No automatic promotion.")
    for path in (SCRIPT, SCRIPT.with_name("run_pilot.sh"), Path(balanced.__file__),
                 stages.SCRIPT, stages.HELPER):
        config["code_sha256"][str(path)] = digest(path)
    write_json(output / "config.json", config)
    (output / "config.sha256").write_text(digest(output / "config.json") + "\n")
    return output, config


def verify(output: Path, config: dict[str, Any]) -> None:
    """Refuse a different runtime, checkpoint, frozen monitor, source or recipe on resume."""
    checked(output / "config.json", (output / "config.sha256").read_text().strip())
    if Path(config["cwd"]) != REPO or config != json.loads((output / "config.json").read_text()):
        raise ValueError("Resume the unchanged configuration in its original checkout")
    if config["reference_sha256"] != REFERENCE_SHA:
        raise ValueError("This pilot must start from verified proof34 iter24")
    checked(Path(config["reference"]), REFERENCE_SHA)
    for name, expected in config["input_sha256"].items():
        checked(output / "inputs" / name, expected)
    runner.checked_files(config["code_sha256"])
    source = json.loads((output / "inputs/collection-config.json").read_text())
    if balanced.runtime_identity(source, config["cpu"]) != config["runtime"]:
        raise ValueError("Runtime changed; restore the original interpreter and extension")


def verify_replay(output: Path, config: dict[str, Any]) -> None:
    """Missing prior policy manifests must never silently fall back to unfiltered games."""
    from alpha_go.boxes.policy_loop_data import receipt

    settings = config["environment"]
    league = json.loads((REPO / BASE / f"league_state-{output.name}.json").read_text())
    completed = max((int(r["iteration"]) for r in league["history"]), default=0)
    data = Path(settings["GAME_DATA_DIR"]) / BASE / output.name
    for iteration in range(max(0, completed-int(settings["WINDOW"])), completed):
        folder = data / f"policy-it{iteration}"
        metadata = json.loads((folder / "preparation.json").read_text())
        if metadata["status"] != "complete":
            raise ValueError(f"Committed iteration has incomplete policy data: {folder}")
        checked(folder / "dataset.txt", metadata["dataset_sha256"])
        sources = [s for s in metadata["identity"]["sources"] if not s["excluded_reason"]]
        for index, source in enumerate(sources):
            bundle = folder / "games" / f"game-{index:07d}"
            checked(bundle / "receipt.json", metadata["completed_receipts"][bundle.name])
            receipt(bundle, source, metadata["identity_sha256"])


def probe_command(output: Path, attempt: Path, candidate: Path, band: str,
                  config: dict[str, Any]) -> list[str]:
    """Read fixed development roots; solver settings match the final full-game test."""
    minimum, maximum = (35, 36) if band == "35-36" else (37, 37)
    caches = [output / "inputs/labels.json",
              *sorted(output.glob("diagnostic-*/attempt-*/probe/labels.json"))]
    stages.merge_caches(caches, attempt / "input-labels.json")
    command = [sys.executable, "-u", "-m", "alpha_go.boxes.search_probe",
        "--reference", config["reference"], "--candidate", str(candidate),
        "--positions-file", str(output / "inputs" / f"positions-{band}.jsonl"),
        "--label-cache", str(attempt / "input-labels.json"), "--output", str(attempt / "probe"),
        "--rows", "5", "--cols", "5", "--min-undrawn", str(minimum),
        "--max-undrawn", str(maximum), "--mixes", "reference", "candidate",
        "--simulations", "400", "--lambdas", "0.5", "--solver", "34",
        "--solver-budget", "50000", "--label-budget", "20000000", "--c-puct", "1.5",
        "--leaf-batch-size", "16", "--policy-temperature", "1", "--margin-utility-k", "6",
        "--prove-terminals", "1", "--shape-terminals", "1", "--seed", "20261104"]
    return command + (["--cpu"] if config["cpu"] else [])


def label_summary(output: Path, config: dict[str, Any]) -> None:
    """Expose whether the run actually obtained complete 37-edge mistake policies."""
    data = Path(config["environment"]["GAME_DATA_DIR"]) / BASE / output.name
    bands: dict[str, Counter[str]] = {}
    iterations = {}
    for folder in sorted(data.glob("policy-it*")):
        metadata = json.loads((folder / "preparation.json").read_text())
        iterations[folder.name] = dict(status=metadata["status"], counts=metadata["counts"])
        for path in sorted(folder.glob("games/*/roots.csv")):
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    band = str(60-int(row["edges"]).bit_count())
                    counts = bands.setdefault(band, Counter())
                    complete = row["complete"].lower() == "true"
                    before = int(row["exact_margin_p1"]) * (1 if int(row["mover"]) == 0 else -1)
                    win_loss = before > 0 and before-int(row["regret"]) < 0
                    counts.update(attempted=1, complete=int(complete),
                                  win_to_loss=int(win_loss),
                                  complete_win_to_loss=int(complete and win_loss))
    write_json(output / "label-coverage.json", dict(iterations=iterations, bands=bands))
    print("Exact policy coverage: " + json.dumps(bands), flush=True)


def diagnostics(output: Path, config: dict[str, Any]) -> None:
    """Keep the final match even if a bounded post-match band is interrupted."""
    from alpha_go.boxes.probe_report import build_report, write_report

    state = json.loads((output / "state.json").read_text())
    if state["phase"] != "complete":
        raise ValueError("Diagnostics require the completed fixed final match")
    candidate = Path(config["checkpoint_dir"]) / f"iter{state['completed_iterations']}.pt"
    result: dict[str, Any] = dict(complete=False, bands={})
    for band, cap in config["diagnostic_caps_seconds"].items():
        def finish(attempt: Path) -> None:
            report = build_report(attempt / "probe", seed=20261104, resamples=10000)
            write_report(report, attempt / "report")
        try:
            attempt = stages.stage(output, f"diagnostic-{band}", cap,
                lambda path: probe_command(output, path, candidate, band, config), finish)
            result["bands"][band] = dict(complete=True, attempt=str(attempt.relative_to(output)))
        except (TimeoutError, subprocess.CalledProcessError) as error:
            result["bands"][band] = dict(complete=False, error=str(error))
            print(f"Diagnostic {band} incomplete: {error}; final match is preserved", flush=True)
        write_json(output / "diagnostics.json", result)
    result["complete"] = all(r["complete"] for r in result["bands"].values())
    write_json(output / "diagnostics.json", result)


def run(output: Path, config: dict[str, Any]) -> None:
    """Run the guarded continuing loop, final arena, then resumable regression reads."""
    original_job = runner.job

    def guarded_job(command: list[str], *args: Any, **kwargs: Any) -> None:
        if command[:2] == ["bash", str(BASE / "run_iteration_local.sh")]:
            verify_replay(output, config)
        original_job(command, *args, **kwargs)

    runner.job = guarded_job
    try:
        verify(output, config)
        verify_replay(output, config)
        runner.run(output, config)
        verify_replay(output, config)
        label_summary(output, config)
        diagnostics(output, config)
        verify(output, config)
    finally:
        runner.job = original_job
        stages.package(output)
        print(f"Send this file: {output / 'results.tgz'}", flush=True)


def main() -> None:
    """Start a four-work-hour pilot or resume every completed stage without retraining."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=4)
    parser.add_argument("--final-games", type=int, default=200)
    parser.add_argument("--reference", type=Path, help="alternate path to the same verified iter24")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true", help="explicit CPU smoke tests only")
    args = parser.parse_args()
    helpers.pin_runtime()
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / "config.json").read_text())
    else:
        output, config = create(args)
    print(f"Run: {output}\nResume: bash {output / 'resume.sh'}\n"
          f"{config['hours']:g} work hours, finish the current iteration, then "
          f"{config['final_games']} games of LAST learner vs proof34 iter24.\n"
          "Exact mistakes through 37; solver remains 34. Two bounded band diagnostics follow.\n"
          "Do not change checkout or rebuild during this run.", flush=True)
    def interrupted(_signal: int, _frame: Any) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    with (output / "overnight.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(output, config)


if __name__ == "__main__":
    main()
