"""Calibrate larger cumulative probe budgets with a clock-check opportunity every 1024 nodes."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np

HERE = Path(__file__).resolve().parent
BUDGETS = (4096, 16384, 65536)


def main() -> None:
    """Retain each probe's TT between slices and validate against the frozen exact values."""
    with (HERE / "data/probe-calibration.csv").open() as handle:
        inputs = [row for row in csv.DictReader(handle) if int(row["budget"]) == 1024]
    assert len(inputs) == 1400
    rows: list[dict[str, int | float]] = []
    with (HERE / "data/probe-slices.csv").open("w", newline="") as handle:
        writer = None
        for old in inputs:
            for budget in BUDGETS:
                solver = alpha_go_cpp.BoxesSolver(5, 5)
                spent = calls = 0
                result = None
                start = time.perf_counter_ns()
                while result is None and spent < budget:
                    result = solver.value_within(int(old["mask"]), min(1024, budget - spent))
                    spent += int(solver.nodes())
                    calls += 1
                elapsed = (time.perf_counter_ns() - start) / 1e6
                assert result is None or int(result) == int(old["value"])
                row = {"index": int(old["index"]), "undrawn": int(old["undrawn"]),
                       "mask": int(old["mask"]), "value": int(old["value"]), "budget": budget,
                       "solved": int(result is not None), "nodes": spent, "calls": calls,
                       "ms": elapsed}
                del solver
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
            handle.flush()
            if (int(old["index"]) + 1) % 100 == 0:
                print(f"{int(old['index']) + 1}/1400 development positions", flush=True)
    summaries = []
    for undrawn in range(24, 31):
        for budget in BUDGETS:
            group = [r for r in rows if r["undrawn"] == undrawn and r["budget"] == budget]
            summaries.append({"undrawn": undrawn, "budget": budget, "positions": len(group),
                              "solved_share": float(np.mean([r["solved"] for r in group])),
                              "median_ms": float(np.median([r["ms"] for r in group])),
                              "p90_ms": float(np.quantile([r["ms"] for r in group], 0.9)),
                              "max_ms": max(r["ms"] for r in group)})
    with (HERE / "data/probe-slices-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    choices = [(n, budget) for n in (26, 25, 24) for budget in BUDGETS
               if all(row["solved_share"] >= 0.9 for row in summaries
                      if row["budget"] == budget and row["undrawn"] <= n)]
    assert choices, "No tested setting established adequate fresh-table probe coverage"
    threshold, budget = choices[0]
    choice = {"threshold": threshold, "budget": budget, "slice_nodes": 1024,
              "rule": "prefer threshold26 (then25,24), smallest cumulative budget4096/16384/65536 "
                      "with >=90% solved at each sampled count24..threshold; development only",
              "positions": len(inputs), "seed": 73, "fresh_tables": True}
    (HERE / "data/probe-choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "choice": choice}, indent=2))


if __name__ == "__main__":
    main()
