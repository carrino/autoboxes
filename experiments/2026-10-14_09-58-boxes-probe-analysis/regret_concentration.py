"""Summarize hindsight concentration of already-solved arena mistakes; never train."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    """Read the previous replay CSV and emit provenance plus per-arm concentrations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    with args.positions.open() as stream:
        records = list(csv.DictReader(stream))
    summaries = []
    for arm in sorted({r["arm"] for r in records}):
        games: dict[int, list[float]] = defaultdict(list)
        known = unknown = 0
        for row in records:
            if row["arm"] != arm or row["mover"] != "A" or not 35 <= int(row["undrawn"]) <= 38:
                continue
            # This historical column was misnamed: its values are score-margin points.
            if row["regret_boxes"] == "":
                unknown += 1
                continue
            regret = float(row["regret_boxes"])
            if regret < 0:
                raise ValueError("Exact regret must be nonnegative")
            games[int(row["index"])].append(regret)
            known += 1
        total = sum(sum(values) for values in games.values())
        top1 = sum(sum(sorted(values, reverse=True)[:1]) for values in games.values())
        top2 = sum(sum(sorted(values, reverse=True)[:2]) for values in games.values())
        summaries.append(dict(
            arm=arm, known_moves=known, unknown_moves=unknown,
            total_margin_regret=total,
            top1_regret_share=top1 / total if total else None,
            top2_regret_share=top2 / total if total else None,
            positive_regret_moves=sum(v > 0 for values in games.values() for v in values),
        ))
    with (args.output / "concentration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    document = dict(
        source_sha256=hashlib.sha256(args.positions.read_bytes()).hexdigest(),
        units="final-score margin points; transferring a box shifts margin by two",
        scope="hindsight description of arena games; not training data or a strength test",
        arms=summaries,
    )
    (args.output / "concentration.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
