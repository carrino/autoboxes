"""Validate fresh self-play proof-filter targets before exact labelling or training."""
# ruff: noqa: N806 (dimension-suffixed arrays)
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from alpha_go.boxes.curriculum import digest, write_json
from alpha_go.boxes.dataset import undrawn_edges

SEMANTICS = "proof_filtered_visits_v1"
BANDS = ((0, 23), (24, 30), (31, 34), (35, 36), (37, 38), (39, 60), (61, 64))


def audit_collection(games: Path, expected_games: int) -> dict[str, Any]:
    """Check saved count masks, allowing unsearched rows and exact solved-root targets.

    This checks fresh collection files, before policy labelling replaces selected rows.
    It verifies visit accounting, not the mathematical validity of search proofs.
    """
    if expected_games <= 0 or not games.is_dir():
        raise ValueError("Require a games directory and positive expected-games")
    paths = sorted(games.rglob("*.npz"))
    if any(path.name.endswith(".tmp.npz") for path in paths):
        raise ValueError(f"Unfinished temporary NPZ in collection: {games}")
    if len(paths) != expected_games:
        raise ValueError(f"Expected {expected_games} finished games, found {len(paths)}: {games}")
    bands: dict[str, dict[str, Any]] = {
        f"{low}..{high}": dict(positions=0, searched_positions=0, changed_positions=0,
                               removed_mass_sum=0.0)
        for low, high in BANDS
    }
    files: dict[str, str] = {}
    for path in paths:
        before = digest(path)
        with np.load(path, allow_pickle=False) as game:
            semantics = np.asarray(game.get("policy_target_semantics", ""))
            if semantics.shape != () or str(semantics.item()) != SEMANTICS:
                raise ValueError(f"Missing or stale policy_target_semantics: {path}")
            required = ("boards", "mcts_visits", "mcts_raw_visits")
            if any(name not in game for name in required):
                raise ValueError(f"Missing boards or visit arrays: {path}")
            boards_NHW = game["boards"]
            if (boards_NHW.ndim != 3 or boards_NHW.shape[0] == 0
                    or min(boards_NHW.shape[1:]) < 3
                    or any(size % 2 == 0 for size in boards_NHW.shape[1:])):
                raise ValueError(f"Invalid board lattice shape: {path}")
            rows, cols = ((size - 1) // 2 for size in boards_NHW.shape[1:])
            edges = rows * (cols + 1) + (rows + 1) * cols
            if edges > 64:
                raise ValueError(f"Unsupported board with more than 64 edges: {path}")
            raw_NE, filtered_NE = game["mcts_raw_visits"], game["mcts_visits"]
            shape = (len(boards_NHW), edges)
            if raw_NE.shape != shape or filtered_NE.shape != shape:
                raise ValueError(f"Visit arrays must both have shape {shape}: {path}")
            for counts in (raw_NE, filtered_NE):
                if (counts.dtype.kind not in "iuf" or not np.isfinite(counts).all()
                        or np.any(counts < 0)):
                    raise ValueError(f"Visit counts must be finite nonnegative numbers: {path}")
            if np.any((filtered_NE != 0) & (filtered_NE != raw_NE)):
                raise ValueError(f"Filtered targets must retain or zero each raw count: {path}")
            raw_N = raw_NE.sum(axis=1, dtype=np.float64)
            filtered_N = filtered_NE.sum(axis=1, dtype=np.float64)
            searched_N = raw_N > 0
            if np.any(searched_N & (filtered_N <= 0)):
                raise ValueError(f"A searched row lost every training visit: {path}")
            changed_N = filtered_N != raw_N
            removed_N = np.divide(raw_N - filtered_N, raw_N,
                                  out=np.zeros_like(raw_N), where=searched_N)
            undrawn_N = undrawn_edges(boards_NHW, edges)
            for low, high in BANDS:
                selected_N = (undrawn_N >= low) & (undrawn_N <= high)
                band = bands[f"{low}..{high}"]
                band["positions"] += int(selected_N.sum())
                band["searched_positions"] += int((selected_N & searched_N).sum())
                band["changed_positions"] += int((selected_N & changed_N).sum())
                band["removed_mass_sum"] += float(removed_N[selected_N].sum())
        if digest(path) != before:
            raise ValueError(f"Collection file changed during audit: {path}")
        files[str(path.resolve())] = before
    if sorted(games.rglob("*.npz")) != paths:
        raise ValueError("Collection file set changed during audit")
    totals = {key: sum(band[key] for band in bands.values()) for key in
              ("positions", "searched_positions", "changed_positions", "removed_mass_sum")}
    for band in [*bands.values(), totals]:
        count = band["searched_positions"]
        band["unsearched_positions"] = band["positions"] - count
        removed = band.pop("removed_mass_sum")
        band["mean_raw_mass_removed"] = removed / count if count else None
    return dict(status="complete", policy_target_semantics=SEMANTICS, games=len(paths),
                scope="Fresh collection visit accounting before exact-policy overrides",
                mean_raw_mass_removed_denominator="all searched rows, including unchanged rows",
                totals=totals, bands=bands, files_sha256=files)


def main() -> None:
    """Write an immutable-input audit report; never modify the collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_collection(args.games, args.expected_games)
    if args.output.resolve() in {Path(path) for path in report["files_sha256"]}:
        raise ValueError("Audit output must not overwrite a source game")
    write_json(args.output, report)
    print(json.dumps(dict(games=report["games"], **report["totals"])), flush=True)


if __name__ == "__main__":
    main()
