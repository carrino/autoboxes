"""Fresh-table exact solver benchmark, with per-position reproducibility checks."""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np

from alpha_go.boxes.rules import geometry


def largest_region(mask: int) -> int:
    """Count undrawn edges in the largest connected component of open boxes."""
    geo = geometry(5, 5)
    parent = list(range(geo.num_boxes))

    def find(box: int) -> int:
        while parent[box] != box:
            parent[box] = parent[parent[box]]
            box = parent[box]
        return box

    legal = [e for e in range(geo.num_edges) if not (mask >> e) & 1]
    for edge in legal:
        boxes = geo.edge_boxes[edge]
        parent[find(boxes[0])] = find(boxes[-1])
    counts: dict[int, int] = {}
    for edge in legal:
        root = find(geo.edge_boxes[edge][0])
        counts[root] = counts.get(root, 0) + 1
    return max(counts.values(), default=0)


def measure(item: tuple[int, int, int]) -> dict[str, int | float]:
    """Solve once unbounded and twice budgeted, each with an independently fresh solver."""
    index, undrawn, mask = item
    result: dict[str, int | float] = {
        "index": index, "undrawn": undrawn, "mask": mask,
        "largest_region": largest_region(mask),
    }
    for label, budget in (("exact", 0), ("100k", 100_000), ("1m", 1_000_000)):
        solver = alpha_go_cpp.BoxesSolver(5, 5)
        start = time.perf_counter_ns()
        value = solver.value(mask) if budget == 0 else solver.value_within(mask, budget)
        result[f"{label}_ms"] = (time.perf_counter_ns() - start) / 1e6
        result[f"{label}_nodes"] = int(solver.nodes())
        if budget == 0:
            result["value"] = int(value)
            result["table_bytes"] = int(solver.table_bytes())
        else:
            result[f"{label}_solved"] = int(value is not None)
            assert value is None or value == result["value"], (mask, value, result)
        del solver
    return result


def summarize(rows: list[dict[str, int | float]], output: Path) -> None:
    """Write per-undrawn node/time quantiles and both budget success rates."""
    summary = []
    for undrawn in sorted({int(row["undrawn"]) for row in rows}):
        group = [r for r in rows if r["undrawn"] == undrawn]
        entry: dict[str, int | float] = {"undrawn": undrawn, "positions": len(group)}
        for metric in ("exact_nodes", "exact_ms"):
            values = [r[metric] for r in group]
            for suffix, quantile in (("median", 0.5), ("p90", 0.9), ("max", 1.0)):
                entry[f"{metric}_{suffix}"] = float(np.quantile(values, quantile))
        for label in ("100k", "1m"):
            entry[f"{label}_solved"] = float(np.mean([r[f"{label}_solved"] for r in group]))
        summary.append(entry)
        print(json.dumps(entry), flush=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    """Run a mask file in stable input order and fail on any baseline-value mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("masks", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    masks = [tuple(map(int, line.split())) for line in args.masks.read_text().splitlines()]
    masks = masks[:args.limit]
    expected: dict[tuple[int, int], int] = {}
    if args.reference:
        with args.reference.open() as handle:
            expected = {(int(r["index"]), int(r["mask"])): int(r["value"])
                        for r in csv.DictReader(handle)}
        assert all((i, mask) in expected for i, (_, mask) in enumerate(masks)), (
            "reference does not cover the requested positions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    rows: list[dict[str, int | float]] = []
    if args.resume and args.output.exists():
        with args.output.open() as prior:
            rows = [{key: float(value) if key.endswith("_ms") else int(value)
                     for key, value in row.items()} for row in csv.DictReader(prior)]
        assert len({row["index"] for row in rows}) == len(rows)
        for row in rows:
            assert masks[int(row["index"])] == (row["undrawn"], row["mask"])
            if expected:
                assert row["value"] == expected[(int(row["index"]), int(row["mask"]))], row
    done = {int(row["index"]) for row in rows}
    with ProcessPoolExecutor(max_workers=args.workers) as pool, args.output.open(
        "a" if rows else "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])) if rows else None
        futures = [pool.submit(measure, (i, n, m))
                   for i, (n, m) in enumerate(masks) if i not in done]
        for future in as_completed(futures):
            error = future.exception()
            if error is not None:
                print(f"FATAL: position measurement failed: {error}", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                raise error
            row = future.result()
            if expected:
                if row["value"] != expected[(int(row["index"]), int(row["mask"]))]:
                    print(f"FATAL: baseline value mismatch: {row}", flush=True)
                    pool.shutdown(wait=False, cancel_futures=True)
                assert row["value"] == expected[(int(row["index"]), int(row["mask"]))], row
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if len(rows) % 100 == 0:
                print(f"{len(rows)}/{len(masks)} positions, {time.monotonic() - start:.1f}s",
                      flush=True)
    summarize(rows, args.output.with_name(args.output.stem + "-summary.csv"))
    print(f"validated {len(rows)} values; wall_seconds={time.monotonic() - start:.3f}", flush=True)


if __name__ == "__main__":
    main()
