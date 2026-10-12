"""Build the run ledger and paired, position-bootstrap held-out comparisons."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def records(name: str) -> list[dict[str, str]]:
    """Read one completed run's position metrics."""
    with (HERE / "data" / f"{name}.csv").open() as stream:
        return list(csv.DictReader(stream))


def paired(base: str, candidate: str) -> dict[str, Any]:
    """Resample paired positions, keeping each configuration's comparison aligned."""
    a, b = records(base), records(candidate)
    assert [r["mask"] for r in a] == [r["mask"] for r in b]
    delta = np.array([int(y["optimal"]) - int(x["optimal"]) for x, y in zip(a, b)])
    rng = np.random.default_rng(20261012)
    means = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {
        "base": base, "candidate": candidate, "n": len(delta),
        "gain": float(delta.mean()), "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "improved": int((delta == 1).sum()), "regressed": int((delta == -1).sum()),
    }


def main() -> None:
    """Summarize all finished runs and both frozen held-out checkpoint comparisons."""
    rows = []
    for path in sorted((HERE / "data").glob("*.json")):
        run = json.loads(path.read_text())
        if "search_optimal" not in run:
            continue
        per_position = records(run["name"])
        above = [r for r in per_position if int(r["undrawn"]) > 30]
        rows.append({
            "name": run["name"], "iteration": run["iteration"],
            "positions_tag": run["positions_tag"], "n": run["num_positions"],
            "correct": run["correct"], "search_optimal": run["search_optimal"],
            "n_31_34": len(above),
            "search_optimal_31_34": sum(int(r["optimal"]) for r in above) / len(above),
            "seconds": run["seconds"], "oracle_seconds": run["oracle_seconds"],
            "root_solved": sum(int(r["root_solved"]) for r in per_position),
            "root_proven": sum(int(r["root_proven"]) for r in per_position),
            **run["search_flags"],
        })
    if rows:
        with (HERE / "data" / "runs.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    comparisons = []
    for iteration in (54, 15):
        base, candidate = f"test-baseline{iteration}", f"test-best{iteration}"
        if all((HERE / "data" / f"{name}.csv").exists() for name in (base, candidate)):
            comparisons.append(paired(base, candidate))
    (HERE / "data" / "paired.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    for row in rows:
        print(f"{row['name']:25s} {row['correct']:3d}/{row['n']} "
              f"{100 * row['search_optimal']:.1f}% {row['seconds']:.1f}s")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
