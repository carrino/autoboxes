"""Give every stored game exact value targets from the solver's reach down.

Adds `exact_margin_p1` to each game NPZ under --games (`alpha_go.boxes.relabel`): the exact
final margin from every position with at most --max-undrawn undrawn edges, in place. The
loop runs it on each self-play iteration (`RELABEL_N`), so the value head trains on exact
values where the games are decided instead of on outcomes both sides misplayed. Prints how
far the outcomes were from the exact values by undrawn band: the noise the relabelling
removes. Progress is flushed every five completed games or 30 seconds, including while
all workers are still solving their current games.

  uv run relabel_exact.py --games $GAME_DATA_DIR/experiments/<exp>/5x5-s1600/selfplay-it20 \\
      --rows 5 --max-undrawn 38 --num-workers 4
"""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
from numpy.typing import NDArray

from alpha_go.boxes.dataset import undrawn_edges
from alpha_go.boxes.relabel import EXACT_UNKNOWN, relabel_file
from alpha_go.boxes.rules import geometry
from alpha_go.self_play import parse_score_from_result

_thread_local = threading.local()
PROGRESS_SECONDS = 30


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True, help="directory of game NPZs (searched recursively)")
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--max-undrawn", type=int, default=38)
    p.add_argument("--node-budget", type=int, default=100_000_000)
    p.add_argument("--table-entries", type=int, default=1 << 22)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--bands", type=int, nargs="*", default=[25, 31, 35],
                   help="lower edges of the undrawn bands in the report")
    args = p.parse_args()
    cols = args.cols or args.rows
    geo = geometry(args.rows, cols)
    paths = sorted(Path(args.games).rglob("*.npz"))
    assert paths, f"no .npz games under {args.games}"

    def solver() -> alpha_go_cpp.BoxesSolver:  # one table per thread, carried across games
        if not hasattr(_thread_local, "solver"):
            _thread_local.solver = alpha_go_cpp.BoxesSolver(args.rows, cols, args.table_entries)
        return _thread_local.solver

    def one(path: Path) -> tuple[NDArray[np.int16], NDArray[np.int64], int]:
        exact = relabel_file(path, args.rows, cols, args.max_undrawn, args.node_budget, solver())
        game = np.load(path)
        return exact, undrawn_edges(game["boards"], geo.num_edges), int(
            parse_score_from_result(str(game["result"])))

    start = time.monotonic()
    print(f"Relabelling {len(paths)} games under {Path(args.games).name}: "
          f"{args.num_workers} workers, reach {args.max_undrawn}, "
          f"{args.node_budget:,} nodes per position", flush=True)
    results: list[tuple[NDArray[np.int16], NDArray[np.int64], int]] = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        pending = {pool.submit(one, path) for path in paths}
        next_report, reported = start + PROGRESS_SECONDS, 0
        try:
            while pending:
                done, pending = wait(pending, timeout=max(0, next_report - time.monotonic()),
                                     return_when=FIRST_COMPLETED)
                results.extend(future.result() for future in done)
                now = time.monotonic()
                if now >= next_report or len(results) - reported >= 5 or not pending:
                    print(f"relabelled {len(results)}/{len(paths)} games; "
                          f"{now - start:.0f}s elapsed, {len(pending)} unfinished", flush=True)
                    next_report, reported = now + PROGRESS_SECONDS, len(results)
        finally:
            # An interrupt/error must not leave the entire queued batch running on exit.
            for future in pending:
                future.cancel()
    edges = [*sorted(b for b in args.bands if b <= args.max_undrawn), args.max_undrawn + 1]
    edges = edges if len(edges) > 1 else [0, args.max_undrawn + 1]
    labels = [f"{lo}..{hi - 1}" for lo, hi in zip(edges, edges[1:])]
    diffs: dict[str, list[tuple[int, bool]]] = {lab: [] for lab in labels}
    known = unknown = 0
    for exact, undrawn, outcome in results:
        for value, u in zip(exact, undrawn):
            if u > args.max_undrawn:
                continue
            if value == EXACT_UNKNOWN:
                unknown += 1
                continue
            known += 1
            if u >= edges[0]:
                diffs[labels[int(np.searchsorted(edges[1:], u, side="right"))]].append(
                    (abs(int(value) - outcome), (int(value) > 0) != (outcome > 0)))
    print(f"{len(paths)} games under {Path(args.games).name}: {known} positions labelled within "
          f"{args.max_undrawn} undrawn edges, {unknown} over the {args.node_budget:,}-node budget, "
          f"{time.monotonic() - start:.0f}s on {args.num_workers} workers")
    print("undrawn | positions | mean |outcome - exact| | outcome sign wrong")
    for lab in labels:
        d = np.array(diffs[lab])
        if len(d):
            print(f"{lab} | {len(d)} | {d[:, 0].mean():.2f} | {d[:, 1].mean():.2f}")


if __name__ == "__main__":
    main()
