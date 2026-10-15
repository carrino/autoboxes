"""Score saved native Windows decisions without inference or new solver work."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from alpha_go.boxes.external_probe import (
    TABLES,
    aggregate,
    compare,
    digest,
    read_roots,
    score_move,
    write_json,
)

ENGINE_HASHES = {
    "prsboxes": "2f603782446e0df640b832966ac7b19e5ad9e8781aa6bc0c279fa415901b9d55",
    "dabble": "7e88069a551edae2fc8a404eca9fcb0366b3ba729738e0f4f5bea399a7704c32",
}
BANDS = (("35..36", 35, 36), ("37..38", 37, 38))


def read_json(path: Path) -> Any:
    """Accept both Python UTF-8 receipts and Windows PowerShell UTF-8 BOM files."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def score_results(results: Path, tables: Path) -> dict[str, Any]:
    """Reject mismatched inputs and score only atomically completed decisions."""
    roots = {root.index: root for root in read_roots(tables)}
    hashes = {name: digest(tables / name) for name in TABLES}
    runtime = read_json(results / "runtime-provenance.json")
    if (runtime["input_sha256"]["positions.jsonl"] != hashes[TABLES[0]]
            or digest(results / "all-positions.jsonl") != hashes[TABLES[0]]
            or digest(results / "positions.jsonl") != runtime["selected_positions_sha256"]):
        raise ValueError("Windows inputs differ from the frozen positions")
    selected = [json.loads(line) for line in
                (results / "positions.jsonl").read_text(encoding="utf-8-sig").splitlines()
                if line.strip()]
    indices = [p["position"] for p in selected]
    if (not indices or len(set(indices)) != len(indices)
            or indices != runtime["selected_position_ids"] or not set(indices) <= roots.keys()):
        raise ValueError("Invalid or duplicate selected positions")
    for position in selected:
        root = roots[position["position"]]
        if any(position[k] != getattr(root, k) for k in ("rows", "cols", "prefix", "family")):
            raise ValueError("Selected position differs from frozen state")
    seconds = runtime["requested_seconds"]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid requested time budget")
    selected_by_id = {p["position"]: p for p in selected}
    baselines = {}
    for name in ("reference", "candidate"):
        baseline = read_json(tables / f"autoboxes-{name}-s400.json")
        if baseline["tables_sha256"] != hashes:
            raise ValueError("Baseline labels differ from these exact tables")
        baselines[name] = baseline["records"]

    engines = {}
    for name, expected_hash in ENGINE_HASHES.items():
        directory = results / name
        status_path = directory / "exit-status.txt"
        status = int(status_path.read_text(encoding="utf-8-sig")) if status_path.exists() else None
        paths = sorted((directory / "roots").glob("root-*/result.json"))
        provenance_path = directory / "engine-provenance.json"
        provenance = read_json(provenance_path) if provenance_path.exists() else None
        if (paths or provenance) and (
            not provenance or provenance["engine_sha256"] != expected_hash
        ):
            raise ValueError(f"{name}: unexpected engine provenance")
        records = []
        seen: set[int] = set()
        for path in paths:
            saved = read_json(path)
            request, response = saved["request"], saved["response"]
            index = request["position"]
            if index not in selected_by_id or index in seen:
                raise ValueError(f"{name}: unexpected or duplicate result position")
            seen.add(index)
            if (any(request[k] != v for k, v in selected_by_id[index].items())
                    or response["position"] != index or request["seconds"] != seconds
                    or response["requested_seconds"] != seconds):
                raise ValueError(f"{name}: result request differs from frozen input or budget")
            if name == "prsboxes":
                if (response["native_root_roundtrip_verified"] is not True
                        or response["mode"] != "normal-gui-computer-move"):
                    raise ValueError("PRsBoxes did not verify its native board roundtrip")
            else:
                board = roots[index].board()
                if (response["executable_sha256"] != expected_hash
                        or response["input_to_play"] != board.to_play()
                        or response["input_scores"] != list(board.boxes)):
                    raise ValueError("Dabble state or executable differs")
            wall_seconds = saved["wall_seconds"]
            if not math.isfinite(wall_seconds) or wall_seconds < 0:
                raise ValueError("Invalid decision wall time")
            record = score_move(roots[index], response["edge"])
            record.update(wall_seconds=wall_seconds, engine_details=response)
            records.append(record)
        engines[name] = dict(
            complete=status == 0 and len(records) == len(indices), exit_status=status,
            missing_positions=[index for index in indices if index not in seen],
            provenance=provenance, aggregate=aggregate(records),
            bands={band: aggregate([r for r in records if lo <= r["undrawn"] <= hi])
                   for band, lo, hi in BANDS},
            comparisons={label: dict(
                overall=compare(records, baseline),
                bands={band: compare([r for r in records if lo <= r["undrawn"] <= hi], baseline)
                       for band, lo, hi in BANDS}) for label, baseline in baselines.items()},
            records=records,
        )
    report = dict(
        complete=all(engine["complete"] for engine in engines.values()),
        scope="Frozen development positions; not full games or equal-time GPU comparisons",
        regret_units="final score-margin points; transferring one box changes margin by two",
        timing="wall_seconds includes startup and adapter work; raw timing in engine_details",
        failures="Missing decisions are reported separately; paired comparisons use common roots",
        tables_sha256=hashes, runtime=runtime, engines=engines,
    )
    write_json(results / "summary.json", report)
    return report


def main() -> None:
    """Print a compact read and save complete per-position scores and intervals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args()
    report = score_results(args.results, args.tables)
    for name, engine in report["engines"].items():
        score = engine["aggregate"]
        print(f"{name}: {score['positions']} decisions; optimal {score['optimal_rate']}; "
              f"mean margin regret {score['mean_margin_regret']}; wins preserved "
              f"{score['wins_preserved']}/{score['winning_roots']}; "
              f"missing {len(engine['missing_positions'])}", flush=True)
    print(f"Scored results: {args.results / 'summary.json'}", flush=True)
    if not report["complete"]:
        raise SystemExit("Partial run: completed decisions saved; inspect engine error logs.")


if __name__ == "__main__":
    main()
