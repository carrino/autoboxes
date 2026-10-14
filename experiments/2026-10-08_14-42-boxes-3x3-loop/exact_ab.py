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
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    directories: list[Path], destination: Path, reuse_reach: int | None = None
) -> tuple[dict[str, Path], list[Path], list[dict[str, Any]]]:
    """Copy matching A/B data; strip prior relabels only from original-outcome game copies.

    Solver-labelled collections already have exact ``result`` targets and are preserved
    in both arms. They are not sequential games and must never be replayed by the relabeller.
    With reuse_reach, preserve existing exact arrays within that reach in the exact arm.
    Games without arrays are scheduled for solving; existing budget failures stay unknown.
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
            exact = None
            if not labelled and reuse_reach is not None and "exact_margin_p1" in game:
                exact = np.asarray(game["exact_margin_p1"])
                assert exact.shape == (len(game["moves"]),), f"invalid exact label shape: {path}"
                assert np.issubdtype(exact.dtype, np.integer), f"non-integer exact labels: {path}"
                _, height, width = game["boards"].shape
                rows, cols = (height - 1) // 2, (width - 1) // 2
                assert np.all(
                    (exact == EXACT_UNKNOWN) | (np.abs(exact.astype(np.int64)) <= rows * cols)
                ), path
                undrawn = undrawn_edges(game["boards"], rows * (cols + 1) + cols * (rows + 1))
                exact = np.where(undrawn <= reuse_reach, exact, EXACT_UNKNOWN).astype(np.int16)
            if not labelled:
                game.pop("exact_margin_p1", None)
            relative = Path(str(index)) / path.relative_to(source)
            for arm, directory in arms.items():
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = (
                    {**game, "exact_margin_p1": exact}
                    if arm == "exact" and exact is not None
                    else game
                )
                np.savez_compressed(target, **payload)
            if not labelled:
                relabel_paths.append(arms["exact"] / relative)
            inputs.append(
                {
                    "source": str(path),
                    "source_directory": str(source),
                    "copy": str(relative),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "labelled": labelled,
                    "label_action": "static"
                    if labelled
                    else "reuse"
                    if exact is not None
                    else "solve",
                }
            )
    assert relabel_paths, "the manifest contains no sequential games to relabel"
    return arms, relabel_paths, inputs


def relabel_games(
    paths: list[Path],
    rows: int,
    cols: int,
    reach: int,
    workers: int,
    budget: int,
    reuse: bool = False,
) -> dict[str, dict[str, int]]:
    """Reuse stored arrays when requested and solve only files without them.

    Existing unknown entries are not retried. Completion-order progress and a 30-second
    heartbeat remain visible even when the first game is unusually expensive.
    """
    local = threading.local()

    def one(path: Path) -> tuple[NDArray[np.int16], NDArray[np.int64], bool]:
        with np.load(path) as game:
            exact = (
                game["exact_margin_p1"].astype(np.int16)
                if reuse and "exact_margin_p1" in game
                else None
            )
            undrawn = undrawn_edges(game["boards"], rows * (cols + 1) + cols * (rows + 1))
        reused = exact is not None
        if exact is None:
            if not hasattr(local, "solver"):
                local.solver = alpha_go_cpp.BoxesSolver(rows, cols, 1 << 22)
            exact = relabel_file(path, rows, cols, reach, budget, local.solver)
        return exact, undrawn, reused

    counts: dict[int, dict[str, int]] = {}
    reused_count = solved_count = reported = 0
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one, path) for path in paths}
        next_report = start + 30
        try:
            while pending:
                done, pending = wait(
                    pending,
                    timeout=max(0, next_report - time.monotonic()),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    exact, undrawn, reused = future.result()
                    reused_count += int(reused)
                    solved_count += int(not reused)
                    for value, u in zip(exact, undrawn):
                        if u <= reach:
                            band = counts.setdefault(int(u), {"known": 0, "unknown": 0})
                            band["unknown" if value == EXACT_UNKNOWN else "known"] += 1
                now = time.monotonic()
                completed = reused_count + solved_count
                if now >= next_report or completed - reported >= 10 or not pending:
                    print(
                        f"labels ready for {completed}/{len(paths)} games: "
                        f"{reused_count} reused, {solved_count} relabelled; "
                        f"{now - start:.0f}s elapsed",
                        flush=True,
                    )
                    next_report, reported = now + 30, completed
        finally:
            for future in pending:
                future.cancel()
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
    p.add_argument(
        "--reuse-exact",
        "--reuse-labels",
        action="store_true",
        help="reuse stored exact arrays and solve only games without them",
    )
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
    arms, to_relabel, inputs = snapshot(directories, data, args.reach if args.reuse_exact else None)
    manifests: dict[str, Path] = {}
    for arm, directory in arms.items():
        manifests[arm] = logs / f"{arm}-manifest.txt"
        manifests[arm].write_text(str(directory.resolve()) + "\n")
    label_plan = {
        action: sum(item["label_action"] == action for item in inputs)
        for action in ("reuse", "solve", "static")
    }
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
    print(
        f"Label plan: {label_plan['reuse']} games reuse saved arrays; "
        f"{label_plan['solve']} games need solving; "
        f"{label_plan['static']} static collections stay unchanged",
        flush=True,
    )
    for directory in directories:
        missing = sum(
            item["source_directory"] == str(directory) and item["label_action"] == "solve"
            for item in inputs
        )
        if missing:
            print(f"  to solve: {directory} ({missing} games)", flush=True)
    report: dict[str, Any] = {
        "tag": tag,
        "logs": str(logs),
        "training": {},
        "arena": {},
        "reused_exact_labels": args.reuse_exact,
        "label_plan": label_plan,
        "validation_scope": "replay diagnostic; warm-start may have seen these games",
    }

    def save_report() -> None:
        (logs / "summary.json").write_text(json.dumps(report, indent=2))

    save_report()
    report["relabel_coverage"] = relabel_games(
        to_relabel, rows, cols, args.reach, args.workers, args.node_budget, reuse=args.reuse_exact
    )
    save_report()
    assert sum(band["known"] for band in report["relabel_coverage"].values()), (
        "no known exact labels within reach; the value-target comparison would be empty"
    )
    environment = {**os.environ, "GAME_DATA_DIR": str(DATA_ROOT.resolve())}
    checkpoints = {"reference": reference}
    for arm, arm_manifest in manifests.items():
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
