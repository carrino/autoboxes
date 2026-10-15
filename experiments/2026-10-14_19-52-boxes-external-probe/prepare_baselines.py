"""Rescore the recorded 3070 search choices using all 3,640 exact child values."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from alpha_go.boxes.external_probe import TABLES, aggregate, digest, read_roots, score_move


def main() -> None:
    """Reproduce the baseline JSON files without inference or solving."""
    data = Path(__file__).resolve().parent / "data"
    roots = {r.index: r for r in read_roots(data)}
    source = json.loads((data / "probe-summary.json").read_text())
    with (data / "probe-positions.csv").open() as stream:
        decisions = list(csv.DictReader(stream))
    for mix in ("reference", "candidate"):
        records = []
        for row in decisions:
            if row["case"] != f"{mix}:s400:lambda0.5":
                continue
            root = roots[int(row["position"])]
            assert root.family == row["source_family"]
            record = score_move(root, int(row["search_edge"]))
            record["wall_seconds"] = float(row["search_seconds"])
            records.append(record)
        assert len(records) == len(roots) == len({r["position"] for r in records})
        report = dict(
            complete=True, engine=f"autoboxes-{mix}-s400",
            tables_sha256={name: digest(data / name) for name in TABLES},
            scope="Previously recorded RTX 3070 decisions; not equal hardware or equal-time "
                  "vs cloud CPU engines",
            regret_units="final score-margin points",
            checkpoint_sha256=source["checkpoint_sha256"][mix],
            source_csv_sha256=digest(data / "probe-positions.csv"),
            search_arguments=source["arguments"], aggregate=aggregate(records), records=records,
        )
        (data / f"autoboxes-{mix}-s400.json").write_text(json.dumps(report, indent=2) + "\n")
        print(mix, report["aggregate"])


if __name__ == "__main__":
    main()
