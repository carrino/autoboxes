"""Measure external engines on frozen, fully solved Boxes decision positions.

This diagnostic replays recorded prefixes and scores a single selected move. It does
not play complete games or solve new labels. Full-game evaluations belong in gameplay.
Engine workers read one JSON request from stdin and write one JSON response to stdout.
Each decision starts a fresh worker so no engine inherits another position's cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alpha_go.boxes.rules import BoxesBoard

TABLES = ("child-diagnostic-positions.jsonl", "child-diagnostic-roots.csv",
          "child-diagnostic-children.csv")


def digest(path: Path) -> str:
    """Hash an immutable experiment input or engine artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Publish completed results atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class FrozenRoot:
    """A replayable position with every legal child's exact root-mover margin."""

    index: int
    rows: int
    cols: int
    prefix: list[int]
    family: str
    exact: dict[int, int]
    root_final: int

    def board(self) -> BoxesBoard:
        """Replay the history, retaining captured boxes and extra turns."""
        board = BoxesBoard(self.rows, self.cols)
        for edge in self.prefix:
            if type(edge) is not int or edge not in board.get_legal_moves_flat():
                raise ValueError(f"Position {self.index}: illegal replay edge {edge}")
            board.play_edge(edge)
        return board

    def request(self, seconds: float, seed: int) -> dict[str, Any]:
        """Give a worker both the history and independently reconstructed state."""
        board = self.board()
        return dict(schema=1, position=self.index, rows=self.rows, cols=self.cols,
                    prefix=self.prefix, seconds=seconds, seed=seed,
                    to_play=board.to_play(), scores=list(board.boxes),
                    board=board.to_numpy().tolist())


def read_roots(tables: Path) -> list[FrozenRoot]:
    """Reject partial, contradictory, duplicated or mismatched exact tables."""
    positions = [json.loads(line) for line in (tables / TABLES[0]).read_text().splitlines()
                 if line.strip()]
    with (tables / TABLES[1]).open() as stream:
        raw = list(csv.DictReader(stream))
    roots = {int(row["position"]): row for row in raw}
    if len(roots) != len(raw):
        raise ValueError("Duplicate root labels")
    children: dict[int, dict[int, int]] = {}
    with (tables / TABLES[2]).open() as stream:
        for row in csv.DictReader(stream):
            index, edge = int(row["position"]), int(row["edge"])
            group = children.setdefault(index, {})
            if edge in group:
                raise ValueError("Duplicate child label")
            group[edge] = int(row["final_margin_for_root_mover"])
    indices = [p["position"] for p in positions]
    if (not positions or len(set(indices)) != len(indices)
            or set(indices) != set(roots) or set(indices) != set(children)):
        raise ValueError("Position/root/child indices disagree or are empty")
    result = []
    states: set[tuple[int, int, int, int, int]] = set()
    for p in positions:
        index = p["position"]
        root = FrozenRoot(index, p["rows"], p["cols"], p["prefix"], p["family"],
                          children[index], int(roots[index]["root_final"]))
        board = root.board()
        legal = board.get_legal_moves_flat()
        if not legal or set(root.exact) != set(legal):
            raise ValueError(f"Position {index}: exact children do not cover legal moves")
        if root.root_final != max(root.exact.values()):
            raise ValueError(f"Position {index}: root value contradicts its children")
        if any(abs(v) > root.rows * root.cols or (v-root.rows*root.cols) % 2
               for v in root.exact.values()):
            raise ValueError(f"Position {index}: impossible final score margin")
        if len(legal) != int(roots[index]["undrawn"]):
            raise ValueError(f"Position {index}: undrawn count disagrees")
        state = (root.rows, root.cols, board.edges, board.to_play(), board.margin())
        if state in states:
            raise ValueError("Duplicate board state")
        states.add(state)
        result.append(root)
    return result


def score_move(root: FrozenRoot, edge: int) -> dict[str, Any]:
    """Separate score-margin optimality from preserving a game-theoretic win."""
    if type(edge) is not int or edge not in root.exact:
        raise ValueError(f"Position {root.index}: engine returned illegal edge {edge}")
    value = root.exact[edge]
    return dict(position=root.index, family=root.family, undrawn=len(root.exact),
                edge=edge, root_final=root.root_final, chosen_final=value,
                optimal=int(value == root.root_final),
                margin_regret=root.root_final-value,
                winning_root=int(root.root_final > 0),
                win_preserved=int(root.root_final > 0 and value > 0),
                win_to_loss=int(root.root_final > 0 and value < 0),
                win_to_draw=int(root.root_final > 0 and value == 0))


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize complete decisions; absent results never count as poor moves."""
    wins = sum(r["winning_root"] for r in records)
    preserved = sum(r["win_preserved"] for r in records)
    return dict(positions=len(records), winning_roots=wins,
                optimal_rate=float(np.mean([r["optimal"] for r in records])) if records else None,
                mean_margin_regret=float(np.mean([r["margin_regret"] for r in records]))
                if records else None,
                wins_preserved=preserved, win_preservation_rate=preserved/wins if wins else None,
                win_to_loss=sum(r["win_to_loss"] for r in records),
                win_to_draw=sum(r["win_to_draw"] for r in records),
                total_wall_seconds=sum(r.get("wall_seconds", 0.0) for r in records))


def compare(records: list[dict[str, Any]], baseline: list[dict[str, Any]],
            seed: int = 20261015, resamples: int = 10000) -> dict[str, Any]:
    """Pair engines on common positions and resample whole source families."""
    other = {r["position"]: r for r in baseline}
    common = [r for r in records if r["position"] in other]
    if not common:
        return dict(common_positions=0)
    for r in common:
        if any(r[k] != other[r["position"]][k]
               for k in ("family", "root_final", "undrawn")):
            raise ValueError("Comparison positions differ")
    families = sorted({r["family"] for r in common})
    counts = np.array([sum(r["family"] == f for r in common) for f in families])
    rng = np.random.default_rng(seed)
    sample = rng.integers(len(families), size=(resamples, len(families)))
    metrics: dict[str, Any] = {}
    for key in ("optimal", "margin_regret", "win_preserved", "win_to_loss"):
        sums = np.array([sum(r[key]-other[r["position"]][key] for r in common
                             if r["family"] == f) for f in families])
        distribution = sums[sample].sum(axis=1)/counts[sample].sum(axis=1)
        metrics[key] = dict(delta=float(sums.sum()/counts.sum()),
                            ci95=np.quantile(distribution, [.025, .975]).tolist())
    return dict(common_positions=len(common), families=len(families),
                direction="engine minus baseline; win metrics use all common roots",
                bootstrap="paired source-family percentile, position-weighted",
                resamples=resamples, seed=seed, metrics=metrics)


def engine_config(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Load an explicitly selected worker and verify its recorded artifacts."""
    config = json.loads(path.read_text())
    if set(config) != {"name", "command", "artifacts", "provenance"}:
        raise ValueError("Engine config needs name, command, artifacts and provenance")
    command = config["command"]
    if (not isinstance(command, list) or not command
            or not all(isinstance(x, str) and x for x in command)):
        raise ValueError("Engine command must be a nonempty argv list")
    if not config["artifacts"] or not config["provenance"]:
        raise ValueError("Record engine artifacts and provenance before comparing")
    for name, expected in config["artifacts"].items():
        artifact = (path.parent / name).resolve()
        if digest(artifact) != expected:
            raise ValueError(f"Engine artifact changed: {artifact}")
    executable = (path.parent / command[0]).resolve()
    if not executable.is_file():
        raise ValueError("Engine command[0] must name a file relative to its config")
    return config, [str(executable), *command[1:]]


def call_worker(command: list[str], request: dict[str, Any], cwd: Path,
                timeout: float) -> subprocess.CompletedProcess[str]:
    """Reap the worker and its children if a decision is interrupted or times out."""
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, cwd=cwd,
                          start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(json.dumps(request)+"\n", timeout=timeout)
        except BaseException:
            # Exception handling is cleanup, not a substitute move or success result.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run/resume bounded individual decisions and persist progress after each root."""
    if (not math.isfinite(args.seconds) or args.seconds <= 0 or args.limit < 0
            or not math.isfinite(args.timeout_slack) or args.timeout_slack <= 0):
        raise ValueError("Positive finite seconds/slack and nonnegative limit required")
    roots = read_roots(args.tables)
    if args.limit:
        roots = roots[:args.limit]
    config, command = engine_config(args.engine_config.resolve())
    manifest = dict(schema=1, engine=config, seconds=args.seconds, seed=args.seed,
                    timeout_slack=args.timeout_slack,
                    positions=[r.index for r in roots],
                    tables_sha256={name: digest(args.tables/name) for name in TABLES},
                    harness_sha256=digest(Path(__file__)), platform=platform.platform(),
                    processor=platform.processor())
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / "settings.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != manifest:
            raise ValueError("Resume settings, engine, code or positions changed")
    else:
        if list(args.output.iterdir()):
            raise ValueError("Output is not empty and has no settings receipt")
        write_json(receipt, manifest)
    baseline = []
    if args.baseline:
        previous = json.loads(args.baseline.read_text())
        if previous["tables_sha256"] != manifest["tables_sha256"]:
            raise ValueError("Baseline used different frozen position/label tables")
        baseline = previous["records"]
    records = []
    for root in roots:
        request = root.request(args.seconds, args.seed+root.index)
        destination = args.output / f"position-{root.index:04d}.json"
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved["request"] != request:
                raise ValueError("Saved request differs from the frozen position")
        else:
            started = time.monotonic()
            process = call_worker(command, request, args.engine_config.resolve().parent,
                                  args.seconds+args.timeout_slack)
            elapsed = time.monotonic()-started
            (args.output / f"position-{root.index:04d}.stderr").write_text(process.stderr)
            process.check_returncode()
            response = json.loads(process.stdout)
            score_move(root, response["edge"])
            saved = dict(request=request, response=response, wall_seconds=elapsed)
            write_json(destination, saved)
        record = score_move(root, saved["response"]["edge"])
        record["wall_seconds"] = saved["wall_seconds"]
        record["engine_details"] = saved["response"]
        records.append(record)
        report: dict[str, Any] = dict(
            complete=len(records) == len(roots), engine=config["name"],
            tables_sha256=manifest["tables_sha256"],
            scope="Frozen development diagnostic; not match strength or equal-time GPU comparison",
            regret_units="final score-margin points; transferring one box changes margin by two",
            timing="fresh worker per decision; wall includes startup; engine timer in details",
            aggregate=aggregate(records),
            bands={band: aggregate([r for r in records if lo <= r["undrawn"] <= hi])
                   for band, lo, hi in (("35..36", 35, 36), ("37..38", 37, 38))},
            records=records)
        if baseline:
            report["comparison"] = compare(records, baseline)
        write_json(args.output / "summary.json", report)
        print(f"{config['name']}: {len(records)}/{len(roots)}; "
              f"optimal {report['aggregate']['optimal_rate']:.3f}; "
              f"margin regret {report['aggregate']['mean_margin_regret']:.3f}", flush=True)
    return report


def main() -> None:
    """Run one external engine at a documented time budget on exact-labelled states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--timeout-slack", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--baseline", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
