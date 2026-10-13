"""Snapshot a replay window, compare original/exact values, and evaluate both candidates.

Run from the repository root with ``uv run <this file>``. Defaults target the stalled
5x5-s1600/iter20 checkpoint and its actual dataset manifest. Source NPZs and checkpoints
are read only; every run needs a fresh tag. CPU mode and small budgets support smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import torch
from numpy.typing import NDArray
from oracle_eval import late_positions, raw_net

from alpha_go.boxes.dataset import undrawn_edges
from alpha_go.boxes.nn_agent import load_boxes_net, pick_device
from alpha_go.boxes.relabel import EXACT_UNKNOWN, relabel_file

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
DATA_ROOT = Path(
    os.environ.get("GAME_DATA_DIR", str(Path.home() / "autoboxes-data/game_data_root"))
)
SEARCH = [
    "--policy_temperature",
    "1.0",
    "--prove_terminals",
    "1",
    "--shape_terminals",
    "1",
    "--margin_utility_lambda",
    "0.5",
]


def manifest_directories(manifest: Path, data_root: Path) -> list[Path]:
    """Read and validate every source directory before creating run outputs."""
    assert manifest.is_file(), f"missing replay manifest: {manifest}; pass --manifest explicitly"
    directories = [
        (data_root / line.strip()).resolve()
        for line in manifest.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert directories and len(directories) == len(set(directories)), "empty/duplicate manifest"
    for directory in directories:
        assert directory.is_dir() and any(directory.rglob("*.npz")), f"no games in {directory}"
    return directories


def snapshot(
    directories: list[Path], destination: Path
) -> tuple[dict[str, Path], list[Path], list[dict[str, Any]]]:
    """Copy matching A/B data; strip prior relabels only from original-outcome game copies.

    Solver-labelled collections already have exact ``result`` targets and are preserved
    in both arms. They are not sequential games and must never be replayed by the relabeller.
    """
    destination.mkdir(parents=True, exist_ok=False)
    arms = {arm: destination / arm for arm in ("control", "exact")}
    for directory in arms.values():
        directory.mkdir()
    relabel_paths, inputs = [], []
    for index, source in enumerate(directories):
        for path in sorted(source.rglob("*.npz")):
            raw = path.read_bytes()
            with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
                game = dict(loaded)
            assert len(game["moves"]) > 0 and game["boards"].ndim == 3, path
            labelled = str(game.get("termination", "")) == "solver"
            if not labelled:
                game.pop("exact_margin_p1", None)
            relative = Path(str(index)) / path.relative_to(source)
            for directory in arms.values():
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, **game)
            if not labelled:
                relabel_paths.append(arms["exact"] / relative)
            inputs.append(
                {
                    "source": str(path),
                    "copy": str(relative),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "labelled": labelled,
                }
            )
    assert relabel_paths, "the manifest contains no sequential games to relabel"
    return arms, relabel_paths, inputs


def relabel_games(
    paths: list[Path], rows: int, cols: int, reach: int, workers: int, budget: int
) -> dict[str, dict[str, int]]:
    """Relabel only copied games and report exact coverage separately at every edge count."""
    local = threading.local()

    def one(path: Path) -> tuple[NDArray[np.int16], NDArray[np.int64]]:
        if not hasattr(local, "solver"):
            local.solver = alpha_go_cpp.BoxesSolver(rows, cols, 1 << 22)
        exact = relabel_file(path, rows, cols, reach, budget, local.solver)
        with np.load(path) as game:
            undrawn = undrawn_edges(game["boards"], rows * (cols + 1) + cols * (rows + 1))
        return exact, undrawn

    counts: dict[int, dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (exact, undrawn) in enumerate(pool.map(one, paths), 1):
            for value, u in zip(exact, undrawn):
                if u <= reach:
                    band = counts.setdefault(int(u), {"known": 0, "unknown": 0})
                    band["unknown" if value == EXACT_UNKNOWN else "known"] += 1
            if i % 10 == 0 or i == len(paths):
                print(f"relabelled {i}/{len(paths)} games", flush=True)
    return {str(k): v for k, v in sorted(counts.items())}


def run_logged(command: list[str], log: Path, environment: dict[str, str]) -> dict[str, Any]:
    """Run a checked child command with a durable log and extract its result JSON."""
    print(f"running {command[1]} -> {log}", flush=True)
    with log.open("w") as output:
        subprocess.run(
            command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT, check=True
        )
    lines = log.read_text().split("===RESULT===")[-1].strip().splitlines()
    assert lines, f"no result JSON in {log}"
    return dict(json.loads(lines[0]))


def value_bands(
    checkpoints: dict[str, Path],
    source: Path,
    rows: int,
    cols: int,
    bands: list[tuple[int, int]],
    count: int,
    seed: int,
    budget: int,
    cpu: bool,
    destination: Path,
) -> list[dict[str, Any]]:
    """Solve root values only; expensive all-child optimal-policy labelling is unnecessary.

    Every model sees the identical solved subset. Budget failures remain in the coverage
    report. These are descriptive band reads, not independent-game confidence intervals.
    """
    device = pick_device("cpu" if cpu else None)
    solver = alpha_go_cpp.BoxesSolver(rows, cols, 1 << 22)
    results = []
    for low, high in bands:
        sampled = late_positions(source, rows, cols, low, high, count, random.Random(seed))
        boards, final = [], []
        for i, board in enumerate(sampled, 1):
            value = solver.value_within(board.edges(), budget)
            if value is not None:
                boards.append(board)
                final.append(board.margin() + value)
            if i % 10 == 0 or i == len(sampled):
                print(
                    f"evaluation {low}..{high}: solved {len(boards)}/{i} sampled roots", flush=True
                )
        assert boards, f"no exact evaluation labels in {low}..{high} within budget"
        truth = np.asarray(final)
        decided = truth != 0
        records = []
        for name, checkpoint in checkpoints.items():
            model = load_boxes_net(checkpoint, device)
            # Keep validation forwards small enough for the 8 GB GPU.
            wins, margins = [], []
            for start in range(0, len(boards), 64):
                _, win, margin = raw_net(model, device, boards[start : start + 64])
                wins.extend(win.tolist())
                margins.extend(margin.tolist())
            win, margin = np.asarray(wins), np.asarray(margins)
            row = {
                "model": name,
                "band": f"{low}..{high}",
                "sampled": len(sampled),
                "solved": len(boards),
                "unknown": len(sampled) - len(boards),
                "value_sign": float(np.mean((win[decided] > 0.5) == (truth[decided] > 0)))
                if decided.any()
                else None,
                "margin_mae": float(np.abs(margin - truth).mean()),
                "brier": float(np.mean((win - ((truth > 0) + 0.5 * (truth == 0))) ** 2)),
            }
            results.append(row)
            for b, target, p, m in zip(boards, truth, win, margin):
                records.append(
                    {
                        "model": name,
                        "mask": int(b.edges()),
                        "player": int(b.player()),
                        "current_margin": int(b.margin()),
                        "exact_margin": int(target),
                        "win_probability": float(p),
                        "expected_margin": float(m),
                    }
                )
            del model
        with (destination / f"value-{low}-{high}.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return results


def main() -> None:
    """Run a reproducible A/B probe without altering the source run or its league."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-tag", default="5x5-s1600")
    p.add_argument("--source-iteration", type=int, default=20)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--tag", default=None)
    p.add_argument("--reach", type=int, default=38)
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    p.add_argument("--node-budget", type=int, default=100_000_000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--positions-tag", default="5x5-mid")
    p.add_argument("--positions", type=int, default=300, help="roots per band; 0 skips value reads")
    p.add_argument("--bands", nargs="+", default=["31:34", "35:38"])
    p.add_argument(
        "--arena-games", type=int, default=200, help="games per model vs classic; 0 skips"
    )
    p.add_argument("--arena-workers", type=int, default=1)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--solver", type=int, default=30)
    p.add_argument("--solver-budget", type=int, default=50_000)
    p.add_argument("--baseline", default="boxes-classic")
    p.add_argument("--cpu", action="store_true", help="tiny-board smoke tests only")
    args = p.parse_args()
    assert args.cpu or torch.cuda.is_available(), "CUDA unavailable; run this on the 3070"
    assert args.workers > 0 and args.arena_workers > 0 and args.epochs > 0
    assert args.arena_games >= 0 and args.arena_games % 2 == 0 and args.positions >= 0
    assert args.reach > 0 and args.node_budget > 0 and args.batch_size > 0
    bands = []
    for band in args.bands:
        low, high = map(int, band.split(":"))
        assert 0 <= low <= high, band
        bands.append((low, high))
    tag = args.tag or "exact-ab-" + datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
        "%Y%m%d-%H%M%S"
    )
    assert Path(tag).name == tag and tag not in (".", ".."), "tag must be a single directory name"
    original = EXP_DIR / "checkpoints" / args.source_tag / f"iter{args.source_iteration}.pt"
    assert original.is_file(), f"missing source checkpoint: {original}"
    manifest = args.manifest or EXP_DIR / f"dataset-{args.source_tag}-it{args.source_iteration}.txt"
    directories = manifest_directories(manifest, DATA_ROOT)
    positions = DATA_ROOT / "experiments" / EXP_DIR.name / args.positions_tag
    assert not args.positions or (positions.is_dir() and any(positions.rglob("*.npz"))), (
        f"missing clean evaluation games: {positions}"
    )
    assert not args.positions or not any(
        positions.resolve() == d
        or positions.resolve() in d.parents
        or d in positions.resolve().parents
        for d in directories
    ), "evaluation directory overlaps the training manifest"
    logs = EXP_DIR / "logs" / tag
    data = DATA_ROOT / "experiments" / EXP_DIR.name / (tag + "-data")
    for target in [
        logs,
        data,
        *[EXP_DIR / "checkpoints" / f"{tag}-{arm}" for arm in ("control", "exact")],
    ]:
        assert not target.exists(), f"refusing to overwrite {target}; choose a new --tag"
    logs.mkdir(parents=True)
    reference = logs / "reference.pt"
    shutil.copy2(original, reference)
    config = torch.load(reference, map_location="cpu", weights_only=False)["config"]
    rows, cols = int(config["rows"]), int(config["cols"])
    arms, to_relabel, inputs = snapshot(directories, data)
    settings = {
        **vars(args),
        "tag": tag,
        "manifest": str(manifest.resolve()),
        "source_checkpoint": str(original),
        "checkpoint_config": config,
        "checkpoint_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "search_flags": SEARCH,
        "inputs": inputs,
    }
    (logs / "settings.json").write_text(json.dumps(settings, indent=2))
    print(
        f"Run {tag}: {len(inputs)} copied files; source data stays untouched. Logs: {logs}",
        flush=True,
    )
    report: dict[str, Any] = {
        "tag": tag,
        "logs": str(logs),
        "training": {},
        "arena": {},
        "validation_scope": "replay diagnostic; warm-start may have seen these games",
    }

    def save_report() -> None:
        (logs / "summary.json").write_text(json.dumps(report, indent=2))

    report["relabel_coverage"] = relabel_games(
        to_relabel, rows, cols, args.reach, args.workers, args.node_budget
    )
    save_report()
    environment = {**os.environ, "GAME_DATA_DIR": str(DATA_ROOT.resolve())}
    checkpoints = {"reference": reference}
    for arm, directory in arms.items():
        arm_manifest = logs / f"{arm}-manifest.txt"
        arm_manifest.write_text(str(directory.resolve()) + "\n")
        command = [
            sys.executable,
            str(EXP_DIR / "train.py"),
            "--dataset-txt",
            str(arm_manifest),
            "--iteration",
            "1",
            "--tag",
            f"{tag}-{arm}",
            "--resume-from",
            str(reference),
            "--rows",
            str(rows),
            "--cols",
            str(cols),
            "--features",
            config.get("features", "basic"),
            "--max-epochs",
            str(args.epochs),
            "--time-budget",
            "0",
            "--q-mix",
            "0",
            "--seed",
            str(args.seed),
            "--split-seed",
            str(args.split_seed),
            "--batch-size",
            str(args.batch_size),
            "--min-undrawn",
            str(max(0, args.solver - 6)),
        ]
        if args.cpu:
            command.append("--cpu")
        report["training"][arm] = run_logged(command, logs / f"train-{arm}.log", environment)
        checkpoints[arm] = EXP_DIR / "checkpoints" / f"{tag}-{arm}" / "iter1.pt"
        save_report()
    assert (
        report["training"]["control"]["steps_completed"]
        == report["training"]["exact"]["steps_completed"]
    )
    if args.positions:
        report["value_bands"] = value_bands(
            checkpoints,
            positions,
            rows,
            cols,
            bands,
            args.positions,
            args.seed,
            args.node_budget,
            args.cpu,
            logs,
        )
        save_report()
    if args.arena_games:
        for name, checkpoint in checkpoints.items():
            command = [
                sys.executable,
                "-m",
                "alpha_go.boxes.arena",
                "--rows",
                str(rows),
                "--cols",
                str(cols),
                "--a",
                f"ckpt:{checkpoint}",
                "--b",
                args.baseline,
                "--num_games",
                str(args.arena_games),
                "--num_workers",
                str(args.arena_workers),
                "--seed",
                str(args.seed),
                "--opening_moves",
                "4",
                "--sims",
                str(args.sims),
                "--solver",
                str(args.solver),
                "--solver_budget",
                str(args.solver_budget),
                "--save_games",
                str(logs / f"arena-{name}.jsonl"),
                *SEARCH,
            ]
            if args.cpu:
                command.append("--cpu")
            report["arena"][name] = run_logged(command, logs / f"arena-{name}.log", environment)
            save_report()
    report["completed"] = True
    save_report()
    print("===RESULT===")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
