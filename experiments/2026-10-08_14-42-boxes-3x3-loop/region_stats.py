"""How often the board has split into independent regions, by undrawn edges.

A region is a set of open boxes joined through the undrawn edges between them; every
undrawn edge belongs to exactly one region (an edge on the rim belongs to its one box).
Regions interact only through whose turn it is, so an exact solver can work on them one at
a time (Nimstring decomposition, PLAN.md). For positions sampled from stored games the way
oracle_eval.py samples them, this reports how many regions there are and how many undrawn
edges the largest one holds: decomposition pays off when the largest region is within the
solver's cheap reach while the whole board is not. With --solver-nodes it also solves every
position at the given undrawn counts (a fresh table each, capped at --node-cap nodes) and
reports the solver's nodes by largest-region size: if they already fall as the largest
region shrinks, the solver exploits the split on its own and decomposition adds little.

  uv run region_stats.py --positions-tag 5x5-mid --min-undrawn 29 --max-undrawn 36
  uv run region_stats.py --positions-tag 5x5-mid --min-undrawn 30 --max-undrawn 32 \
      --solver-nodes 30 32
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np

from alpha_go.boxes.rules import BoxesGeometry, geometry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_eval import EXP_NAME, GAME_DATA_DIR, late_positions  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent


def regions(drawn_mask: int, geo: BoxesGeometry) -> list[int]:
    """Undrawn-edge count of every region, largest first (union-find over open boxes)."""
    parent = list(range(geo.num_boxes))

    def find(b: int) -> int:
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    undrawn = [e for e in range(geo.num_edges) if not (drawn_mask >> e) & 1]
    for e in undrawn:
        boxes = geo.edge_boxes[e]
        parent[find(boxes[0])] = find(boxes[-1])
    sizes: dict[int, int] = {}
    for e in undrawn:
        root = find(geo.edge_boxes[e][0])
        sizes[root] = sizes.get(root, 0) + 1
    return sorted(sizes.values(), reverse=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--positions-tag", required=True, help="run whose games supply the positions")
    p.add_argument("--min-undrawn", type=int, default=29)
    p.add_argument("--max-undrawn", type=int, default=36)
    p.add_argument("--num-positions", type=int, default=4000)
    p.add_argument("--reach", type=int, nargs="*", default=[20, 24, 28],
                   help="report the share of positions whose largest region has <= N edges")
    p.add_argument("--solver-nodes", type=int, nargs="*", default=[],
                   help="undrawn counts at which to report solver nodes by largest region")
    p.add_argument("--node-cap", type=int, default=2_000_000)
    p.add_argument("--save-masks", action="store_true",
                   help="also write the sampled positions' drawn-edge masks, one per line with "
                        "their undrawn count, to data/positions-<tag>-<min>-<max>.txt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    size = re.match(r"(\d+)x(\d+)", args.positions_tag)
    assert size, "positions tag must start with <rows>x<cols>"
    rows, cols = int(size.group(1)), int(size.group(2))
    geo = geometry(rows, cols)
    boards = late_positions(GAME_DATA_DIR / "experiments" / EXP_NAME / args.positions_tag, rows,
                            cols, args.min_undrawn, args.max_undrawn, args.num_positions,
                            random.Random(args.seed))
    by_undrawn: dict[int, list[tuple[object, list[int]]]] = {}
    for b in boards:
        by_undrawn.setdefault(b.num_edges() - b.move_count(), []).append(
            (b, regions(b.edges(), geo)))

    reach_cols = [f"largest <= {n}" for n in args.reach]
    print(f"{len(boards)} positions from {args.positions_tag}, "
          f"{args.min_undrawn}..{args.max_undrawn} undrawn edges\n")
    header = ["undrawn", "positions", "split", "regions", "largest (median)", *reach_cols]
    print(" | ".join(header))
    rows_out = []
    for undrawn in sorted(by_undrawn):
        found = [r for _, r in by_undrawn[undrawn]]
        largest = np.array([r[0] for r in found])
        row = {
            "undrawn": undrawn, "positions": len(found),
            "split": round(float(np.mean([len(r) > 1 for r in found])), 3),
            "regions": round(float(np.mean([len(r) for r in found])), 2),
            "largest (median)": int(np.median(largest)),
            **{c: round(float(np.mean(largest <= n)), 3) for c, n in zip(reach_cols, args.reach)},
        }
        rows_out.append(row)
        print(" | ".join(str(row[h]) for h in header))
    out = EXP_DIR / "data" / f"region_stats-{args.positions_tag}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nwrote {out}")
    if args.save_masks:
        masks = EXP_DIR / "data" / (f"positions-{args.positions_tag}-{args.min_undrawn}-"
                                    f"{args.max_undrawn}.txt")
        masks.write_text("".join(f"{b.num_edges() - b.move_count()} {b.edges()}\n"
                                 for b in boards))
        print(f"wrote {masks} ({len(boards)} positions)")

    for undrawn in args.solver_nodes:
        nodes: dict[int, list[int]] = {}
        for b, found_regions in by_undrawn.get(undrawn, []):
            solver = alpha_go_cpp.BoxesSolver(rows, cols, 1 << 20)
            solver.value_within(b.edges(), args.node_cap)  # type: ignore[attr-defined]
            nodes.setdefault(found_regions[0], []).append(solver.nodes())
        print(f"\nsolver nodes at {undrawn} undrawn, by largest region (cap {args.node_cap}):")
        print("largest | positions | nodes median | nodes p90 | capped")
        for largest in sorted(nodes):
            n = np.array(nodes[largest])
            print(f"{largest} | {len(n)} | {int(np.median(n))} | {int(np.percentile(n, 90))} | "
                  f"{float(np.mean(n >= args.node_cap)):.2f}")


if __name__ == "__main__":
    main()
