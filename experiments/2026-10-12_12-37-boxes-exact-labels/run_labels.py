"""Run the specified four-shard pilot, size a 12-hour run, and verify/archive its labels.

All source positions are from 5x5-chains. The clean evaluation set is never opened.
Logs and a machine-readable stage manifest survive interruption; an existing run directory
is rejected so a rerun cannot silently mix old and new label files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

EXP = "experiments/2026-10-08_14-42-boxes-3x3-loop"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUMMARY = re.compile(r"(\d+) positions labelled \((\d+) skipped on budget\) in (\d+)s; (\d+) files")


def write_json(path: Path, value: Any) -> None:
    """Replace a manifest atomically after a completed stage."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_stage(name: str, positions: int, budget: int, seed: int, workers: int,
              game_data: Path) -> dict[str, Any]:
    """Run disjoint shards of one deterministic sample with separate logs."""
    output = game_data / EXP / name
    assert not output.exists(), f"refusing to mix a prior run into {output}"
    output.mkdir(parents=True)
    logs = HERE / "logs" / name
    logs.mkdir(parents=True, exist_ok=False)
    processes = []
    handles = []
    commands = []
    env = {**os.environ, "GAME_DATA_DIR": str(game_data), "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    start = time.monotonic()
    for shard in range(workers):
        command = [sys.executable, f"{EXP}/solver_label.py", "--positions-tag", "5x5-chains",
                   "--min-undrawn", "31", "--max-undrawn", "36", "--num-positions",
                   str(positions), "--node-budget", str(budget), "--save-name", name,
                   "--shard", str(shard), "--num-shards", str(workers), "--seed", str(seed)]
        commands.append(command)
        handle = (logs / f"shard-{shard}.log").open("w")
        handles.append(handle)
        processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                          stderr=subprocess.STDOUT))
    write_json(logs / "manifest.json", {"commands": commands, "pids": [p.pid for p in processes],
                                      "started_unix": time.time(), "state": "running"})
    codes = [process.wait() for process in processes]
    for handle in handles:
        handle.close()
    assert all(code == 0 for code in codes), f"shard failures: {codes}; inspect {logs}"
    shards = []
    for shard in range(workers):
        text = (logs / f"shard-{shard}.log").read_text()
        match = SUMMARY.search(text)
        assert match is not None, f"missing completion summary for shard {shard}"
        labelled, skipped, seconds, files = map(int, match.groups())
        shards.append({"shard": shard, "labelled": labelled, "skipped": skipped,
                       "solve_seconds": seconds, "files": files})
    result = {"name": name, "requested": positions, "budget": budget, "seed": seed,
              "workers": workers, "wall_seconds": time.monotonic() - start,
              "labelled": sum(s["labelled"] for s in shards),
              "skipped": sum(s["skipped"] for s in shards), "shards": shards,
              "commands": commands, "output": str(output), "state": "complete"}
    write_json(logs / "manifest.json", result)
    print(json.dumps(result), flush=True)
    return result


def verify(result: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Validate every label, per-shard totals, legal uniform policy, and outcome sign."""
    folder = Path(result["output"])
    counts: Counter[int] = Counter()
    margins: Counter[int] = Counter()
    per_shard: Counter[int] = Counter()
    files = []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            boards = data["boards"]
            n = len(boards)
            assert int(data["num_moves"]) == n
            assert boards.shape == (n, 11, 11)
            visits = data["mcts_visits"]
            assert visits.shape == (n, 60)
            assert np.all((visits == 0) | (visits == 1))
            sums = visits.sum(axis=1)
            assert np.all(sums > 0)
            policies = visits.astype(np.float64) / sums[:, None]
            assert np.allclose(policies.sum(axis=1), 1.0, rtol=0, atol=1e-12)
            edges = np.concatenate([boards[:, 0::2, 1::2].reshape(n, 30),
                                    boards[:, 1::2, 0::2].reshape(n, 30)], axis=1)
            assert np.all(visits[edges != 0] == 0), "a labelled move is already drawn"
            undrawn = 60 - (edges == 1).sum(axis=1)
            assert np.all((31 <= undrawn) & (undrawn <= 36))
            assert np.all(np.diff(undrawn) <= 0)
            players = data["to_play"]
            assert players.shape == (n,) and np.all((players == 0) | (players == 1))
            outcome = str(data["result"])
            margin = 0 if outcome == "Draw" else int(float(outcome[2:]))
            if outcome.startswith("W+"):
                margin = -margin
            assert abs(margin) <= 25 and margin % 2 == 1
            mover_margin = np.where(players == 0, margin, -margin)
            assert np.array_equal(data["mcts_root_values"], (mover_margin > 0).astype(np.float32))
            coords = [(2 * r, 2 * c + 1) for r in range(6) for c in range(5)]
            coords += [(2 * r + 1, 2 * c) for r in range(5) for c in range(6)]
            by_coord = {coord: i for i, coord in enumerate(coords)}
            for i, move in enumerate(data["moves"]):
                coord = (int(move[0]), int(move[1]))
                assert visits[i, by_coord[coord]] == 1
            shard_match = re.search(r"-s(\d+)-", path.name)
            assert shard_match is not None
            shard = int(shard_match.group(1))
            per_shard[shard] += n
            counts.update(map(int, undrawn))
            margins[margin] += n
            files.append({"file": path.name, "positions": n, "margin_p1": margin,
                          "shard": shard, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert sum(counts.values()) == result["labelled"]
    assert len(files) == sum(s["files"] for s in result["shards"])
    for shard in result["shards"]:
        assert per_shard[shard["shard"]] == shard["labelled"]
    undrawn_rows = [{"undrawn": n, "positions": k} for n, k in sorted(counts.items())]
    margin_rows = [{"margin_p1": n, "positions": k} for n, k in sorted(margins.items())]
    for stem, rows in (("files", files), ("undrawn", undrawn_rows), ("margins", margin_rows)):
        if rows:
            with (destination / f"{result['name']}-{stem}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return {"positions": sum(counts.values()), "files": len(files),
            "undrawn": dict(sorted(counts.items())), "margins_p1": dict(sorted(margins.items()))}


def main() -> None:
    """Complete the prescribed pilot/retry/full run, or validate a finished manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-data", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--full-workers", type=int, default=None,
                        help="default: pilot worker count; fewer leave cores for other tasks")
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.workers <= max(1, len(os.sched_getaffinity(0)) - 1)
    full_workers = args.full_workers or args.workers
    assert 1 <= full_workers <= args.workers
    data = HERE / "data"
    data.mkdir(parents=True, exist_ok=True)
    pilot = run_stage("5x5-oracle-31-36-pilot", 100 * args.workers, 20_000_000,
                      0, args.workers, args.game_data)
    verify(pilot, data)
    if pilot["skipped"] / (pilot["labelled"] + pilot["skipped"]) > 0.2:
        pilot = run_stage("5x5-oracle-31-36-pilot-40m", 100 * args.workers, 40_000_000,
                          0, args.workers, args.game_data)
        verify(pilot, data)
    attempted = pilot["labelled"] + pilot["skipped"]
    throughput = attempted / pilot["wall_seconds"] * full_workers / args.workers
    positions = min(60_000, max(full_workers, int(args.hours * 3600 * throughput)))
    sizing = {"pilot": pilot, "hours": args.hours, "positions": positions,
              "full_workers": full_workers, "estimated_wall_seconds": positions / throughput}
    write_json(data / "sizing.json", sizing)
    full = run_stage("5x5-oracle-31-36", positions, pilot["budget"], 1,
                     full_workers, args.game_data)
    full["verification"] = verify(full, data)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as tar:
        tar.add(full["output"], arcname="5x5-oracle-31-36")
    full["archive"] = str(args.archive)
    full["archive_bytes"] = args.archive.stat().st_size
    full["archive_sha256"] = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    write_json(data / "completed.json", full)
    print(json.dumps(full), flush=True)


if __name__ == "__main__":
    main()
