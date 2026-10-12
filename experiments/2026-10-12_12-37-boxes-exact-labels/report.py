"""Render a results report from completed and validated label manifests."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Refuse to report completion before the archive and final validation exist."""
    result = json.loads((HERE / "data/completed.json").read_text())
    sizing = json.loads((HERE / "data/sizing.json").read_text())
    archive = Path(result["archive"])
    assert archive.is_file() and archive.stat().st_size == result["archive_bytes"]
    validation = result["verification"]
    assert validation["positions"] == result["labelled"]
    attempted = result["labelled"] + result["skipped"]
    lines = ["# Exact labels above the solver's reach", "",
             f"Generated **{result['labelled']:,} exact positions** with 31–36 undrawn edges "
             f"from 5x5-chains; {result['skipped']:,} of {attempted:,} attempted positions "
             f"were skipped ({result['skipped'] / attempted:.1%}) at the node limit.", "",
             "Only development games were replayed. No 5x5-mid positions were labeled, "
             "trained on, or used for sizing or tuning.", "",
             "## Pilot and sizing", "",
             "| Stage | Processes | Node budget | Attempted | Labeled | Skipped | Wall minutes |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    pilots = sorted((HERE / "logs").glob("5x5-oracle-31-36-pilot*/manifest.json"))
    for path in pilots:
        pilot = json.loads(path.read_text())
        assert pilot["state"] == "complete"
        n = pilot["labelled"] + pilot["skipped"]
        lines.append(f"| {pilot['name']} | {pilot['workers']} | {pilot['budget']:,} | {n:,} | "
                     f"{pilot['labelled']:,} | {pilot['skipped']:,} | "
                     f"{pilot['wall_seconds'] / 60:.2f} |")
    lines += ["", f"The production request was {sizing['positions']:,} positions, seed 1, "
              f"{result['budget']:,} nodes per solver call, {result['workers']} processes. "
              f"Measured pilot throughput was scaled to {sizing['hours']:.0f} hours and capped "
              "at 60,000 positions. Two production processes left CPU capacity for tasks 1 and 3.",
              "", f"Actual production wall time: **{result['wall_seconds'] / 3600:.2f} hours**.",
              "", "| Production shard | Labeled | Skipped | Solve seconds | Files |",
              "|---:|---:|---:|---:|---:|"]
    for shard in result["shards"]:
        lines.append(f"| {shard['shard']} | {shard['labelled']:,} | {shard['skipped']:,} | "
                     f"{shard['solve_seconds']:,} | {shard['files']} |")
    lines += ["", "## Distribution", "", "| Undrawn edges | Positions |", "|---:|---:|"]
    for count, n in sorted(validation["undrawn"].items(), key=lambda item: int(item[0])):
        lines.append(f"| {count} | {n:,} |")
    lines += ["", "Final margins are from player 1's perspective. The stored `to_play` "
              "field supplies the training reader's mover perspective.", "",
              "| Final margin | Positions |", "|---:|---:|"]
    for margin, n in sorted(validation["margins_p1"].items(), key=lambda item: int(item[0])):
        lines.append(f"| {int(margin):+d} | {n:,} |")
    lines += ["", "## Verification and artifact", "",
              f"Validated all {validation['files']} files: counts equal the per-shard logs; "
              "every policy has at least one optimal legal edge and sums to 1 after normalization; "
              "stored moves belong to that set. Board shapes, player fields, undrawn counts, "
              "ordering, final margins, and exact root outcomes also pass. The CSVs contain "
              "per-file counts, per-file SHA256 digests, and both distributions.", "",
              f"Archive: `{archive}` ({result['archive_bytes']:,} bytes).", "",
              f"SHA256: `{result['archive_sha256']}`.", "",
              "The NPZ data and tar archive stay out of Git. Extract the folder under the "
              "GPU machine's game-data experiment directory, then add "
              "`experiments/2026-10-08_14-42-boxes-3x3-loop/5x5-oracle-31-36` to `EXTRA_DATA`.", "",
              "## Interpretation", "",
              "These are exact outcomes and uniform-optimal move targets, not self-play outcomes. "
              "Budget skips select harder positions out of the dataset, so its accuracy as a "
              "training target does not make it an unbiased evaluation set. The solver is the "
              "pinned main version, independent of task 1's candidate changes. The clean "
              "evaluation games remain available for measuring any downstream net improvement.", ""]
    (HERE / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
