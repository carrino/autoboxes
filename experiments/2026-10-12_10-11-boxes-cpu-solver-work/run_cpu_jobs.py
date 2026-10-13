"""Run the unchanged experiment commands with separate logs and wall-clock records."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPORT = Path(__file__).resolve().parent
REPO = REPORT.parents[1]
EXP = Path("experiments/2026-10-08_14-42-boxes-3x3-loop")


def record(event: dict) -> None:
    """Append an execution event to the command audit."""
    event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with (REPORT / "logs/resumed-commands.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def run_one(command: list[str], name: str) -> int:
    """Run one measurement command, preserving its complete output."""
    log = REPORT / "logs" / f"{name}.txt"
    record({"event": "start", "command": shlex.join(command), "cwd": str(REPO),
            "GAME_DATA_DIR": os.environ["GAME_DATA_DIR"], "log": str(log)})
    start = time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    outcome = {"event": "finished", "name": name, "exit_code": result.returncode,
               "wall_seconds": time.monotonic() - start}
    record(outcome)
    print(json.dumps(outcome), flush=True)
    print(log.read_text(), flush=True)
    return result.returncode


def job1() -> int:
    """Measure holdout positions; preserve the CSV before its next overwrite."""
    commands = [
        ("region-29-36", ["uv", "run", str(EXP / "region_stats.py"),
                          "--positions-tag", "5x5-mid", "--min-undrawn", "29",
                          "--max-undrawn", "36"]),
        ("region-nodes-30-32", ["uv", "run", str(EXP / "region_stats.py"),
                                "--positions-tag", "5x5-mid", "--min-undrawn", "30",
                                "--max-undrawn", "32", "--num-positions", "1500",
                                "--solver-nodes", "30", "32", "--save-masks"]),
        ("solver-bench", ["uv", "run", str(EXP / "solver_bench.py"), "--tag", "5x5-mid",
                           "--undrawn", "30", "32", "34", "--selfplay-only"]),
    ]
    for name, command in commands:
        code = run_one(command, name)
        if code:
            return code
        if name.startswith("region"):
            band = "29-36" if name == "region-29-36" else "30-32"
            shutil.copy2(REPO / EXP / "data/region_stats-5x5-mid.csv",
                         REPORT / "data" / f"region_stats-5x5-mid-{band}.csv")
        if name == "region-nodes-30-32":
            shutil.copy2(REPO / EXP / "data/positions-5x5-mid-30-32.txt", REPORT / "data")
        if name == "solver-bench":
            shutil.copy2(REPO / EXP / "data/solver_bench-5x5-mid.csv", REPORT / "data")
    return 0


def label(args: argparse.Namespace) -> int:
    """Run disjoint shards from 5x5-chains only, stopping the group on failure."""
    output = Path(os.environ["GAME_DATA_DIR"]) / EXP / args.save_name
    assert not output.exists() or not list(output.iterdir()), f"output is not empty: {output}"
    start = time.monotonic()
    processes = []
    results = []
    for shard in range(args.shards):
        command = ["uv", "run", str(EXP / "solver_label.py"), "--positions-tag", "5x5-chains",
                   "--min-undrawn", "31", "--max-undrawn", "36", "--num-positions",
                   str(args.positions), "--node-budget", str(args.node_budget), "--save-name",
                   args.save_name, "--shard", str(shard), "--num-shards", str(args.shards),
                   "--seed", str(args.seed)]
        log = REPORT / "logs" / f"{args.phase}-shard-{shard}.txt"
        record({"event": "start", "command": shlex.join(command), "cwd": str(REPO),
                "GAME_DATA_DIR": os.environ["GAME_DATA_DIR"], "log": str(log)})
        stream = log.open("x")
        process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((shard, process, stream, log, time.monotonic()))
    failure = False
    while processes:
        finished = [item for item in processes if item[1].poll() is not None]
        for shard, process, stream, log, started in finished:
            stream.close()
            text = log.read_text()
            sampled = re.search(r"^(\d+) positions with", text, re.MULTILINE)
            summary = re.search(r"(\d+) positions labelled \((\d+) skipped on budget\) in "
                                r"(\d+)s; (\d+) files", text)
            result = {"shard": shard, "exit_code": process.returncode,
                      "wall_seconds": time.monotonic() - started, "log": str(log),
                      "sampled": int(sampled[1]) if sampled else None,
                      "labelled": int(summary[1]) if summary else None,
                      "skipped": int(summary[2]) if summary else None,
                      "solver_seconds_rounded": int(summary[3]) if summary else None,
                      "files": int(summary[4]) if summary else None}
            results.append(result)
            record({"event": "finished", "phase": args.phase, **result})
            print(json.dumps(result), flush=True)
            processes.remove((shard, process, stream, log, started))
            failure |= process.returncode != 0 or summary is None
        if failure:
            for _, process, _, _, _ in processes:
                process.terminate()
        status = {"phase": args.phase, "requested": args.positions, "seed": args.seed,
                  "node_budget": args.node_budget, "shards": args.shards,
                  "output": str(output), "wall_seconds": time.monotonic() - start,
                  "running_shards": [item[0] for item in processes], "failed": failure,
                  "results": sorted(results, key=lambda item: item["shard"])}
        status_path = REPORT / "data" / f"{args.phase}-status.json"
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)
        if processes:
            time.sleep(1)
    return int(failure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="job", required=True)
    sub.add_parser("job1")
    labeller = sub.add_parser("label")
    labeller.add_argument("--phase", required=True)
    labeller.add_argument("--positions", type=int, required=True)
    labeller.add_argument("--node-budget", type=int, required=True)
    labeller.add_argument("--shards", type=int, required=True)
    labeller.add_argument("--seed", type=int, required=True)
    labeller.add_argument("--save-name", required=True)
    args = parser.parse_args()
    (REPORT / "logs").mkdir(exist_ok=True)
    (REPORT / "data").mkdir(exist_ok=True)
    raise SystemExit(job1() if args.job == "job1" else label(args))


if __name__ == "__main__":
    main()
