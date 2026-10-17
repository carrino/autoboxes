"""Check paired traces and preserve compact timing artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def timings(paths: list[Path]) -> list[float]:
    return [sum(row["cpu_seconds"] for row in rows(path)) for path in paths]


def comparison(folder: Path, mode: str) -> dict:
    paths = {arm: sorted(folder.glob(f"{mode}-*-{arm}.jsonl"))
             for arm in ("baseline", "candidate")}
    expected = None
    for arm in paths.values():
        for path in arm:
            trace = [{k: v for k, v in row.items() if k != "cpu_seconds"} for row in rows(path)]
            if expected is None:
                expected = trace
            assert trace == expected, path
    values = {arm: timings(files) for arm, files in paths.items()}
    mean = {arm: statistics.mean(times) for arm, times in values.items()}
    return dict(cpu_seconds=values, mean_cpu_seconds=mean,
                speedup=mean["baseline"] / mean["candidate"],
                time_reduction=1 - mean["candidate"] / mean["baseline"],
                exact_trace_parity=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--mcts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {mode: comparison(args.native, mode) for mode in ("canonical", "solve")}
    result["mcts"] = comparison(args.mcts, "mcts")
    probes = rows(args.native / "solve-0-baseline.jsonl")
    result["solve"].update(calls=len(probes), nodes=sum(row["nodes"] for row in probes),
                           exact=sum(row["value"] is not None for row in probes),
                           unknown=sum(row["value"] is None for row in probes))
    result["mcts"]["roots"] = [r["id"] for r in rows(args.mcts / "mcts-0-baseline.jsonl")]
    result["construct"] = {
        str(entries): {
            arm: [next(r["cpu_seconds"] for r in rows(path) if r["entries"] == entries)
                  for path in sorted(args.native.glob(f"construct-*-{arm}.jsonl"))]
            for arm in ("baseline", "candidate")}
        for entries in (2, 1048576)}
    for path in args.native.glob("*.jsonl"):
        with (args.output / path.with_suffix(".csv").name).open("w") as file:
            records = rows(path)
            writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    for path in args.mcts.glob("*.jsonl"):
        shutil.copyfile(path, args.output / path.name)
    for name, source in (("native", args.native), ("mcts", args.mcts)):
        shutil.copyfile(source / "provenance.json", args.output / f"{name}-provenance.json")
    text = json.dumps(result, indent=2) + "\n"
    (args.output / "summary.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
