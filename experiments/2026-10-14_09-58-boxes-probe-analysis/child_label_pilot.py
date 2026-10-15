"""Bounded exact child values for frozen development positions; diagnostic data only."""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from alpha_go.boxes.search_probe import (
    ExactLabels,
    Position,
    load_positions,
    selected_final,
    sha256,
    write_json,
)


def solve_position(index: int, position: Position, output: Path, cache: Path,
                   budget: int, deadline: float) -> dict[str, Any]:
    """Persist each completed child solve; never treat incomplete coverage as optimality."""
    started = time.monotonic()
    board = position.board()
    legal = board.get_legal_moves_flat()
    result: dict[str, Any] = dict(
        position=index, frozen=asdict(position), undrawn=len(legal),
        root_final=None, children=[], legal_edges=legal,
        finished_all_children=False, complete_known=False,
    )
    target = output / f"position-{index:03d}.json"
    if started >= deadline:
        result.update(status="not_started_before_deadline", elapsed_seconds=0)
        write_json(target, result)
        return result
    labels = ExactLabels(position.rows, position.cols, budget, 1 << 22, cache)
    result["root_final"] = labels.final(board, int(board.to_play()))
    for edge in legal:
        if time.monotonic() >= deadline:
            break
        value = selected_final(labels, board, edge)
        root = result["root_final"]
        if root is not None and value is not None:
            assert value <= root, "exact child exceeds minimax root"
        result["children"].append(dict(edge=edge, final_margin_for_root_mover=value))
        result.update(elapsed_seconds=time.monotonic() - started,
                      solver_seconds=labels.seconds, solver_calls=labels.calls,
                      cache_hits=labels.hits)
        write_json(target, result)
        if len(result["children"]) % 10 == 0:
            print(f"position {index}: {len(result['children'])}/{len(legal)} children",
                  flush=True)
    result["finished_all_children"] = len(result["children"]) == len(legal)
    result["complete_known"] = (result["root_final"] is not None
                                and result["finished_all_children"]
                                and all(c["final_margin_for_root_mover"] is not None
                                        for c in result["children"]))
    if result["complete_known"]:
        assert max(c["final_margin_for_root_mover"] for c in result["children"]) == result[
            "root_final"], "complete child set disagrees with exact root"
    result.update(status="finished" if result["finished_all_children"] else "deadline",
                  elapsed_seconds=time.monotonic() - started,
                  solver_seconds=labels.seconds, solver_calls=labels.calls,
                  cache_hits=labels.hits)
    labels.save(output / f"labels-{index:03d}.json")
    write_json(target, result)
    print(f"position {index}: {result['status']}; complete={result['complete_known']}",
          flush=True)
    return result


def main() -> None:
    """Randomly select equal counts from each band without consulting model errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-band", type=int, default=6)
    parser.add_argument("--all-positions", action="store_true")
    parser.add_argument("--label-cache", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--node-budget", type=int, default=20_000_000)
    parser.add_argument("--seconds", type=int, default=360)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    assert min(args.per_band, args.workers, args.node_budget, args.seconds) > 0
    assert not args.output.exists(), "use a fresh output directory"
    source = args.input / "positions.jsonl"
    cache = args.label_cache or args.input / "labels.json"
    positions = load_positions(source, 5, 5, 35, 38)
    rng = random.Random(args.seed)
    bands = [[i for i, p in enumerate(positions) if low <= 60 - len(p.prefix) <= high]
             for low, high in ((35, 36), (37, 38))]
    for band in bands:
        rng.shuffle(band)
    chosen = (list(range(len(positions))) if args.all_positions else
              [i for pair in zip(*(b[:args.per_band] for b in bands)) for i in pair])
    assert args.all_positions or len(chosen) == 2 * args.per_band, (
        "not enough positions in each band")
    args.output.mkdir(parents=True)
    write_json(args.output / "settings.json", dict(
        arguments=vars(args), positions_sha256=sha256(source), cache_sha256=sha256(cache),
        script_sha256=sha256(Path(__file__)), selected_position_indices=chosen,
        scope="Development diagnostic; not independent evaluation or a training dataset",
        sampling=("All frozen positions" if args.all_positions else
                  "Equal per-band random sample, without using model-error information"),
    ))
    started = time.monotonic()
    deadline = started + args.seconds
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(solve_position, i, positions[i], args.output, cache,
                               args.node_budget, deadline) for i in chosen]
        results = [future.result() for future in futures]
    summary = dict(
        elapsed_seconds=time.monotonic() - started, positions=len(results),
        complete_known=sum(r["complete_known"] for r in results),
        attempted_children=sum(len(r["children"]) for r in results),
        known_children=sum(c["final_margin_for_root_mover"] is not None
                           for r in results for c in r["children"]),
        aggregate_solver_seconds=sum(r.get("solver_seconds", 0) for r in results),
    )
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
