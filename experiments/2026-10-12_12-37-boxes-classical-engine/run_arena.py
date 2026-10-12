"""Reproducible paired arena through the repository's play_game/arena implementation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np

from alpha_go.boxes.agents import BoxesClassicAgent
from alpha_go.boxes.arena import MatchResult, _one_game, random_opening


def main() -> None:
    """Checkpoint individual games, then aggregate exactly as the paired arena does."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="boxes-classic")
    parser.add_argument("--b", required=True)
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.num_games > 0 and args.num_games % 2 == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configuration = {"a": args.a, "b": args.b, "rows": 5, "cols": 5,
                     "games": args.num_games, "seed": args.seed, "opening_moves": 4,
                     "classic_settings": {
                         "solver_max_undrawn": BoxesClassicAgent.solver_max_undrawn,
                         "probe_node_budget": alpha_go_cpp.BoxesClassic.probe_node_budget(),
                         "probe_slice_nodes": alpha_go_cpp.BoxesClassic.probe_slice_nodes(),
                         "seconds": {"boxes-classic": 1.0, "boxes-classic-fast": 0.1,
                                     "boxes-classic-tiny": 0.02}},
                     "extension_sha256": hashlib.sha256(
                         Path(alpha_go_cpp.__file__).read_bytes()).hexdigest()}
    settings = args.output.with_suffix(".settings.json")
    if settings.exists():
        assert json.loads(settings.read_text()) == configuration
    else:
        settings.write_text(json.dumps(configuration, indent=2) + "\n")
    records: dict[int, tuple[int, float, int, float, int]] = {}
    if args.output.exists():
        with args.output.open() as handle:
            for row in csv.DictReader(handle):
                index = int(row["game"])
                assert index not in records
                records[index] = (int(row["margin"]), float(row["a_seconds"]),
                                  int(row["a_moves"]), float(row["b_seconds"]), int(row["b_moves"]))
    rng = np.random.default_rng(args.seed)
    openings = [random_opening(5, 5, 4, rng) for _ in range(args.num_games // 2)]
    start = time.monotonic()
    fields = ["game", "margin", "a_seconds", "a_moves", "b_seconds", "b_moves"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.output.open(
        "a" if records else "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        if not records:
            writer.writerow(fields)
        futures = {pool.submit(_one_game, args.a, args.b, 5, 5, i, args.seed, openings[i // 2]): i
                   for i in range(args.num_games) if i not in records}
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            records[index] = result
            writer.writerow((index, *result))
            handle.flush()
            if len(records) % 10 == 0:
                print(f"{args.a} vs {args.b}: {len(records)}/{args.num_games}, "
                      f"wall_seconds={time.monotonic() - start:.1f}", flush=True)
    results = [records[i] for i in range(args.num_games)]
    margins = [r[0] for r in results]
    match = MatchResult(
        a=args.a, b=args.b, rows=5, cols=5, games=args.num_games,
        a_wins=sum(m > 0 for m in margins), b_wins=sum(m < 0 for m in margins),
        draws=sum(m == 0 for m in margins), margins=margins,
        a_seconds_per_move=sum(r[1] for r in results) / max(1, sum(r[2] for r in results)),
        b_seconds_per_move=sum(r[3] for r in results) / max(1, sum(r[4] for r in results)),
    )
    summary = match.summary()
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with args.output.with_suffix(".pairs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "margin_a_first", "margin_a_second", "sum_margin"])
        for i, margin in enumerate(match.pair_margins):
            writer.writerow([i, margins[2 * i], margins[2 * i + 1], margin])
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
