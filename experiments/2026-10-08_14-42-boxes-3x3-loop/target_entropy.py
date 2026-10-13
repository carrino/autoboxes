"""Are the policy targets getting flatter from one iteration to the next?

Self-play trains the policy on the search's visit distribution (Dirichlet noise at the root
included). With few simulations a flat prior spreads the visits, the net trained on them
gets flatter, and the next search spreads further: a loop that does not strengthen the net.
This reads every searched position of each self-play iteration of a run and prints, by
iteration and undrawn band, the entropy of the visit distribution (nats), the share of visits
on the top move and the mean number of legal moves (the entropy of a uniform target is
ln(legal)). Solver-played positions (one visit per optimal edge) are left out via --min-undrawn.

  uv run target_entropy.py --tag 5x5-s1600 [--iterations 0 5 10 15 20] [--bands 31 35 41]
"""
# ruff: noqa: N806
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np

from alpha_go.boxes.dataset import undrawn_edges
from alpha_go.boxes.rules import geometry

EXP_NAME = Path(__file__).resolve().parent.name
GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")).resolve()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="run whose self-play games to read")
    p.add_argument("--iterations", type=int, nargs="*", default=None,
                   help="self-play iterations (default: every selfplay-it* directory)")
    p.add_argument("--min-undrawn", type=int, default=31,
                   help="skip positions with fewer undrawn edges (the solver's)")
    p.add_argument("--bands", type=int, nargs="*", default=[31, 35, 41],
                   help="lower edges of the undrawn bands to report")
    args = p.parse_args()
    root = GAME_DATA_DIR / "experiments" / EXP_NAME / args.tag
    dirs = sorted(root.glob("selfplay-it*"), key=lambda d: int(d.name[11:]))
    if args.iterations is not None:
        dirs = [d for d in dirs if int(d.name[11:]) in args.iterations]
    assert dirs, f"no selfplay-it* directories under {root}"
    size = re.match(r"(\d+)x(\d+)", args.tag)
    assert size, "tag must start with <rows>x<cols>"
    geo = geometry(int(size.group(1)), int(size.group(2)))
    bands = sorted(args.bands)
    edges = [*bands, geo.num_edges + 1]
    labels = [f"{lo}..{hi - 1}" for lo, hi in zip(edges, edges[1:])]
    print(f"{args.tag}: visit-distribution targets by self-play iteration and undrawn band "
          f"(entropy in nats / top-move share / legal moves)\n")
    print("iter | games | " + " | ".join(labels))
    for d in dirs:
        rows: dict[str, list[tuple[float, float, int]]] = {lab: [] for lab in labels}
        paths = sorted(d.glob("*.npz"))
        for path in paths:
            game = np.load(path)
            if "mcts_visits" not in game:
                continue
            visits_NE = game["mcts_visits"].astype(np.float64)
            undrawn_N = undrawn_edges(game["boards"], geo.num_edges)
            for visits_E, undrawn in zip(visits_NE, undrawn_N):
                total = visits_E.sum()
                if total <= 0 or undrawn < args.min_undrawn:
                    continue
                probs_E = visits_E / total
                nonzero = probs_E[probs_E > 0]
                entropy = float(-(nonzero * np.log(nonzero)).sum())
                band = labels[int(np.searchsorted(edges[1:], undrawn, side="right"))]
                rows[band].append((entropy, float(probs_E.max()), int(undrawn)))
        cells = []
        for lab in labels:
            if rows[lab]:
                e = np.array(rows[lab])
                cells.append(f"{e[:, 0].mean():.2f} / {e[:, 1].mean():.2f} / "
                             f"{e[:, 2].mean():.1f} (n={len(e)})")
            else:
                cells.append("-")
        print(f"{d.name[11:]} | {len(paths)} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
