"""Measure repeated solver failures and later recovery without changing search."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import alpha_go_cpp as cpp


def analyze(rows: list[dict[str, Any]], canonical: bool) -> dict[str, Any]:
    solver = cpp.BoxesSolver(5, 5)
    calls = nodes = retries = retry_nodes = successes = 0
    groups_total = failed_groups = recovered_groups = 0
    details = []
    for row in rows:
        groups: dict[int, list[list[int]]] = defaultdict(list)
        for mask, n, aborted in row["probe_trace"]:
            key = int(solver.canonical(mask)) if canonical else mask
            previous = groups[key]
            calls += 1
            nodes += n
            # Retrying an as-yet-unsolved mask after at least one budget exhaustion.
            retry = bool(previous) and all(x[2] for x in previous)
            if retry:
                retries += 1
                retry_nodes += n
                successes += not aborted
            previous.append([mask, n, aborted])
        recovered = [probes for probes in groups.values()
                     if probes[0][2] and any(not p[2] for p in probes[1:])]
        groups_total += len(groups)
        failed_groups += sum(probes[0][2] for probes in groups.values())
        recovered_groups += len(recovered)
        details.append({"id": row["id"], "undrawn": row["undrawn"],
                        "calls": len(row["probe_trace"]), "distinct_masks": len(groups),
                        "recovered_masks": len(recovered)})
    return {"roots": len(rows), "calls": calls, "nodes": nodes,
            "distinct_root_mask_groups": groups_total,
            "groups_first_call_aborted": failed_groups,
            "retries_before_first_success": retries,
            "nodes_in_retries_before_first_success": retry_nodes,
            "successful_retries": successes, "recovered_groups": recovered_groups,
            "roots_detail": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    baseline = {row["id"]: row for row in
                map(json.loads, args.baseline.read_text().splitlines())}
    for row in rows:
        for field in ["visits", "selection_policy", "chosen", "root_proven"]:
            assert row[field] == baseline[row["id"]][field], (row["id"], field)
        for i in [0, 1, 2, 4, 5]:  # Work counts; wall times intentionally excluded.
            assert row["solver_profile"][i] == baseline[row["id"]]["solver_profile"][i]
    result = {"parity_verified_roots": len(rows), "raw_masks": analyze(rows, False),
              "symmetry_canonical_masks": analyze(rows, True)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
