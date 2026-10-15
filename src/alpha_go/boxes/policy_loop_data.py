"""Resumable, bounded exact-policy preparation for one ongoing training iteration.

Each source game publishes an atomic bundle of its overridden trajectory and value-only
children. Hashed receipts make complete games reusable after interruption; incomplete
private staging directories are rebuilt. A portable dataset manifest appears only after
all included sources are complete. Source games and their replay-family split stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any

from alpha_go.boxes.policy_ab_data import (
    filter_sources,
    sha256,
    solve_game,
    write_arms,
    write_json,
)
from alpha_go.boxes.policy_labels import LabelSettings, source_paths
from alpha_go.boxes.rules import geometry


def receipt(bundle: Path, source: dict[str, Any], identity_hash: str) -> dict[str, Any]:
    """Verify a published game's inputs and all output bytes before reusing it."""
    manifest = bundle / "receipt.json"
    checksum = bundle / "receipt.sha256"
    assert manifest.is_file() and checksum.is_file(), f"incomplete published bundle: {bundle}"
    assert sha256(manifest) == checksum.read_text().strip(), f"receipt changed: {bundle}"
    saved: dict[str, Any] = json.loads(manifest.read_text())
    assert saved["identity_sha256"] == identity_hash and saved["source"] == source, (
        f"bundle input identity changed: {bundle}")
    assert saved["status"] == "complete", f"incomplete receipt: {bundle}"
    files = saved["artifact_sha256"]
    actual = {str(path.relative_to(bundle)) for path in bundle.rglob("*") if path.is_file()}
    assert actual == set(files) | {"receipt.json", "receipt.sha256"}, (
        f"unrecorded or missing bundle artifacts: {bundle}")
    for name, digest in files.items():
        path = (bundle / name).resolve()
        assert path.is_relative_to(bundle.resolve()), f"artifact escapes bundle: {name}"
        assert sha256(path) == digest, f"published artifact changed: {path}"
    return saved


def publish_game(output: Path, index: int, source: dict[str, Any], settings: LabelSettings,
                 identity_hash: str) -> dict[str, Any]:
    """Solve privately and publish one self-contained game directory in one rename."""
    name = f"game-{index:07d}"
    bundle = output / "games" / name
    assert not bundle.exists(), f"refusing to replace published bundle: {bundle}"
    stage = output / f".partial-{name}"
    if stage.exists():
        assert stage.is_dir() and not stage.is_symlink(), f"invalid staging directory: {stage}"
        shutil.rmtree(stage)
    for folder in [stage / "exact", stage / "children"]:
        folder.mkdir(parents=True)
    source_path = Path(source["path"])
    assert sha256(source_path) == source["sha256"], f"source changed: {source_path}"
    game, labels, coverage = solve_game(source_path, settings)
    assert sha256(source_path) == source["sha256"], f"source changed during solve: {source_path}"
    coverage.update(write_arms(stage, index, source_path, game, labels, source["sha256"],
                               geometry(settings.rows, settings.cols), write_control=False))
    saved = dict(status="complete", identity_sha256=identity_hash, source=source,
                 counts=dict(coverage), artifact_sha256={
                     str(path.relative_to(stage)): sha256(path)
                     for path in sorted(stage.rglob("*")) if path.is_file()})
    write_json(stage / "receipt.json", saved)
    (stage / "receipt.sha256").write_text(sha256(stage / "receipt.json") + "\n")
    stage.rename(bundle)
    return saved


def prepare_iteration(paths: list[Path], output: Path, settings: LabelSettings, *,
                      exclude_positions: list[Path] | None = None,
                      workers: int = 1) -> dict[str, Any]:
    """Prepare or resume one immutable source set with at most ``workers`` live solves."""
    settings.validate()
    assert paths and workers > 0
    output = output.resolve()
    excluded = sorted({path.resolve() for path in (exclude_positions or [])})
    fresh, _, records = filter_sources(paths, [], excluded, geometry(settings.rows, settings.cols))
    assert fresh, "no training games remain after diagnostic exclusions"
    included = [record for record in records if not record["excluded_reason"]]
    code_hashes = {name: sha256(Path(__file__).parent / name) for name in (
        "policy_loop_data.py", "policy_ab_data.py", "policy_labels.py", "forced.py")}
    identity = dict(schema=1, settings=asdict(settings), sources=records,
                    exclusions={str(path): sha256(path) for path in excluded},
                    code_sha256=code_hashes)
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "preparation.json"
    previous: dict[str, Any] = {}
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        assert previous["identity"] == identity, "preparation inputs/settings/exclusions changed"
        for name, digest in previous.get("completed_receipts", {}).items():
            path = output / "games" / name / "receipt.json"
            assert path.is_file() and sha256(path) == digest, f"published receipt changed: {name}"
        if previous["status"] == "complete":
            assert sha256(output / "dataset.txt") == previous["dataset_sha256"], (
                "completed dataset manifest changed")
    else:
        assert all(p.name == "preparation.tmp.json" for p in output.iterdir()), (
            "existing output is not a policy-loop preparation")
    game_root = output / "games"
    game_root.mkdir(exist_ok=True)
    names = {f"game-{index:07d}" for index in range(len(included))}
    assert all(path.is_dir() and path.name in names for path in game_root.iterdir()), (
        "unexpected published game bundle")
    complete: dict[int, dict[str, Any]] = {}
    missing = []
    for index, source in enumerate(included):
        bundle = game_root / f"game-{index:07d}"
        if bundle.exists():
            complete[index] = receipt(bundle, source, identity_hash)
        else:
            missing.append(index)
    reused = len(complete)
    started = time.monotonic()
    summary: dict[str, Any] = dict(
        schema=1, status="running", identity=identity, identity_sha256=identity_hash,
        workers=workers, source_files=len(records), included_files=len(included),
        excluded_files=len(records) - len(included), reused_games=reused, dataset="dataset.txt",
        code_sha256=code_hashes)

    def update_status() -> None:
        counts: Counter[str] = Counter()
        for saved in complete.values():
            counts.update(saved["counts"])
        summary.update(completed_games=len(complete), counts=dict(counts),
                       elapsed_seconds=round(time.monotonic() - started, 2),
                       completed_receipts={f"game-{index:07d}": sha256(
                           game_root / f"game-{index:07d}" / "receipt.json")
                           for index in sorted(complete)})
        write_json(metadata, summary)

    update_status()
    print(f"Policy labels: {reused}/{len(included)} complete games reused; {len(missing)} "
          f"to solve; {workers} workers, {settings.table_entries * 16 / 1e6:.1f} MB/table; "
          f"line/child budgets {settings.node_budget:,}/{settings.child_node_budget:,}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict[Future[dict[str, Any]], int] = {}
        todo = iter(missing)

        def submit() -> None:
            index = next(todo, None)
            if index is not None:
                pending[pool.submit(publish_game, output, index, included[index], settings,
                                    identity_hash)] = index

        for _ in range(min(workers, len(missing))):
            submit()
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                complete[pending.pop(future)] = future.result()
                update_status()
                submit()
            print(f"Policy labels: {len(complete)}/{len(included)} games ready, "
                  f"{summary['counts'].get('replaced_roots', 0)} exact policies; "
                  f"{time.monotonic() - started:.0f}s", flush=True)
    # Recheck source inputs and receipts before publishing a complete training manifest.
    for source in records:
        assert sha256(Path(source["path"])) == source["sha256"], "source changed during preparation"
    assert identity["exclusions"] == {str(path): sha256(path) for path in excluded}, (
        "exclusions changed during preparation")
    directories = []
    for index, source in enumerate(included):
        bundle = game_root / f"game-{index:07d}"
        saved = receipt(bundle, source, identity_hash)
        directories.append(str((bundle / "exact").relative_to(output)))
        if saved["counts"].get("child_collections", 0):
            directories.append(str((bundle / "children").relative_to(output)))
    temporary = output / "dataset.tmp.txt"
    temporary.write_text("".join(name + "\n" for name in directories))
    temporary.replace(output / "dataset.txt")
    summary["status"] = "complete"
    summary["dataset_sha256"] = sha256(output / "dataset.txt")
    update_status()
    print(f"Policy-labelled dataset ready: {output / 'dataset.txt'}", flush=True)
    return summary


def main() -> None:
    """Command-line loop preparation; repeat the same command to resume safely."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--min-undrawn", type=int, default=35)
    parser.add_argument("--max-undrawn", type=int, default=38)
    parser.add_argument("--node-budget", type=int, default=20_000_000)
    parser.add_argument("--child-node-budget", type=int, default=20_000_000)
    parser.add_argument("--table-entries", type=int, default=1 << 22)
    parser.add_argument("--max-per-game", type=int, default=2)
    parser.add_argument("--min-regret", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--exclude-positions", type=Path, action="append", default=[])
    args = parser.parse_args()
    settings = LabelSettings(args.rows, args.cols or args.rows, args.min_undrawn,
                             args.max_undrawn, args.node_budget, args.child_node_budget,
                             args.table_entries, args.max_per_game, args.min_regret)
    paths = source_paths(args.games, [], Path.cwd())
    prepare_iteration(paths, args.output, settings, exclude_positions=args.exclude_positions,
                      workers=args.workers)


if __name__ == "__main__":
    main()
