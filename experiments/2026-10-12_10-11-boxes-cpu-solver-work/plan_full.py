"""Size the full run from a verified pilot's aggregate attempted-position rate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
phase = sys.argv[1]
assert phase in {"pilot20", "pilot40"}
pilot = json.loads((root / "data" / f"{phase}-verification.json").read_text())
status = json.loads((root / "data" / f"{phase}-status.json").read_text())
assert pilot["verification_passed"]
assert phase == "pilot40" or pilot["skipped"] * 5 <= pilot["sampled"], "40M retry is required"
shards = status["shards"]
rate = pilot["sampled"] / pilot["wall_seconds"]
positions = min(60_000, int(43_200 * rate / shards) * shards)
assert positions >= shards
plan = {
    "pilot_phase": phase,
    "pilot_sampled": pilot["sampled"],
    "pilot_labelled": pilot["positions"],
    "pilot_skipped": pilot["skipped"],
    "pilot_wall_seconds": pilot["wall_seconds"],
    "pilot_attempted_positions_per_second": rate,
    "target_wall_seconds": 43_200,
    "cap_positions": 60_000,
    "requested_positions": positions,
    "shards": shards,
    "positions_per_shard": positions // shards,
    "node_budget": pilot["node_budget"],
    "seed": 1,
    "save_name": "5x5-oracle-31-36",
    "estimated_wall_seconds": positions / rate,
    "formula": "min(60000, floor(43200 * pilot_sampled / pilot_wall_seconds / shards) * shards)",
    "rate_definition": "All attempted positions (labelled plus skipped) divided by concurrent-group wall time, including sampling and output.",
}
(root / "data/full-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
print(json.dumps(plan, indent=2))
