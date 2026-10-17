"""Summarize saved cost counters and target discrepancies without running search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profile = [sum(row["solver_profile"][i] for row in rows) for i in range(7)]
    return {
        "roots": len(rows),
        "searched": sum(row["searched"] for row in rows),
        "proven_in_mcts": sum(row["root_proven"] for row in rows),
        "changed_targets": sum(row["discarded_target_mass"] > 0 for row in rows),
        "search_wall_seconds": sum(row["seconds"] for row in rows),
        "neural_wall_seconds": sum(row["evaluator_seconds"] for row in rows),
        "bounded_calls": profile[0],
        "bounded_nodes": profile[1],
        "aborted_calls": profile[2],
        "bounded_solver_wall_seconds": profile[3] / 1e9,
        "unbounded_calls": profile[4],
        "unbounded_nodes": profile[5],
        "unbounded_solver_wall_seconds": profile[6] / 1e9,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    summary = {}
    for reach in [30, 34]:
        rows = [json.loads(line) for line in
                (args.data / f"solver{reach}.jsonl").read_text().splitlines()]
        summary[str(reach)] = {
            "all": aggregate(rows),
            "by_undrawn": {
                str(n): aggregate([row for row in rows if row["undrawn"] == n])
                for n in sorted({row["undrawn"] for row in rows})
            },
            "changed_targets": [
                {key: row[key] for key in ["id", "family", "undrawn", "discarded_target_mass"]}
                for row in rows if row["discarded_target_mass"] > 0
            ],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
