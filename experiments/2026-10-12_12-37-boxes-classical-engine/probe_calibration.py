"""Measure fresh exact-probe coverage on development positions, without arena tuning."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np

HERE = Path(__file__).resolve().parent
BUDGETS = (1024, 4096, 16384)


def main() -> None:
    """Check every successful probe against an unrestricted fresh-table exact solve."""
    masks = [tuple(map(int, line.split()))
             for line in (HERE / "data/probe-masks-24-30.txt").read_text().splitlines()]
    rows: list[dict[str, int | float]] = []
    with (HERE / "data/probe-calibration.csv").open("w", newline="") as handle:
        writer = None
        for index, (undrawn, mask) in enumerate(masks):
            oracle = alpha_go_cpp.BoxesSolver(5, 5)
            start = time.perf_counter_ns()
            value = int(oracle.value(mask))
            exact_ms = (time.perf_counter_ns() - start) / 1e6
            exact_nodes = int(oracle.nodes())
            del oracle
            for budget in BUDGETS:
                solver = alpha_go_cpp.BoxesSolver(5, 5)
                start = time.perf_counter_ns()
                result = solver.value_within(mask, budget)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                assert result is None or int(result) == value
                row = {"index": index, "undrawn": undrawn, "mask": mask,
                       "budget": budget, "value": value, "solved": int(result is not None),
                       "nodes": int(solver.nodes()), "ms": elapsed,
                       "exact_nodes": exact_nodes, "exact_ms": exact_ms}
                del solver
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
            handle.flush()
            if (index + 1) % 100 == 0:
                print(f"{index + 1}/{len(masks)} development positions", flush=True)
    summaries = []
    for undrawn in range(24, 31):
        for budget in BUDGETS:
            group = [row for row in rows
                     if row["undrawn"] == undrawn and row["budget"] == budget]
            summaries.append({"undrawn": undrawn, "budget": budget, "positions": len(group),
                              "solved_share": float(np.mean([r["solved"] for r in group])),
                              "median_ms": float(np.median([r["ms"] for r in group])),
                              "p90_ms": float(np.quantile([r["ms"] for r in group], 0.9)),
                              "max_ms": max(r["ms"] for r in group),
                              "median_exact_nodes": float(np.median(
                                  [r["exact_nodes"] for r in group]))})
    with (HERE / "data/probe-calibration-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    # Prefer the existing short probe. Lower its eligibility threshold to at most
    # 26, requiring >=90% fresh-table coverage throughout the sampled lower band.
    choices = [n for n in range(24, 27)
               if all(row["solved_share"] >= 0.9 for row in summaries
                      if row["budget"] == 1024 and row["undrawn"] <= n)]
    recommendation = {"budget": 1024, "threshold": max(choices) if choices else None,
                      "rule": "largest threshold24..26 with >=90% solved at each sampled "
                              "undrawn count at or below it; development only, seed73",
                      "positions": len(masks), "fresh_tables": True}
    (HERE / "data/probe-single-choice.json").write_text(
        json.dumps(recommendation, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "recommendation": recommendation}, indent=2))


if __name__ == "__main__":
    main()
