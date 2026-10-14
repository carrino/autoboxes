"""Does the net fit its exact value labels? Value sign and margin error on the training
positions that carry one, by undrawn band.

train.py's accuracy is an aggregate over every position, where exact labels are a small
minority, so it cannot say whether the net learned them. This reads the same manifest,
keeps only the positions whose value target is exact (`exact_margin_p1`, relabel.py or
the solver-labelled sets), and scores each checkpoint on them: a net that fits the labels
here but not on clean positions generalises badly; one that fits neither cannot represent
them (capacity or inputs), whatever the data.

  uv run exact_fit.py --tag quick-exact --iterations 2 3 --dataset-txt dataset-5x5-s1600-it21.txt
"""
# ruff: noqa: N806
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from alpha_go.boxes.dataset import BoxesDataset, undrawn_edges
from alpha_go.boxes.encode import encode_grid
from alpha_go.boxes.nn_agent import load_boxes_net, pick_device
from alpha_go.boxes.relabel import EXACT_UNKNOWN

EXP_DIR = Path(__file__).resolve().parent
GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")).resolve()


def exact_positions(dataset: BoxesDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Planes, exact margin for the side to move and undrawn count of every kept position
    with an exact value target."""
    planes, margins, undrawn = [], [], []
    for game in dataset.games:
        if "exact_margin_p1" not in game:
            continue
        exact = game["exact_margin_p1"]
        indices = dataset.kept_indices(game)
        counts = undrawn_edges(game["boards"], dataset.geo.num_edges)
        for k in indices[exact[indices] != EXACT_UNKNOWN]:
            to_play = int(game["to_play"][k]) + 1
            planes.append(encode_grid(game["boards"][k], to_play, dataset.features))
            margins.append(int(exact[k]) if to_play == 1 else -int(exact[k]))
            undrawn.append(int(counts[k]))
    assert planes, "no positions with exact value targets in this manifest"
    return np.stack(planes), np.array(margins), np.array(undrawn)


@torch.no_grad()
def predict(model: torch.nn.Module, device: torch.device, planes: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    wins, expected = [], []
    for start in range(0, len(planes), 256):
        batch = torch.from_numpy(planes[start:start + 256]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            _, margin_BM = model(batch)
        wins.append(model.win_prob(margin_BM).float().cpu().numpy())
        expected.append(model.expected_margin(margin_BM).float().cpu().numpy())
    return np.concatenate(wins), np.concatenate(expected)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="checkpoint directory under checkpoints/")
    p.add_argument("--iterations", type=int, nargs="+", required=True)
    p.add_argument("--dataset-txt", required=True, help="manifest, relative to this folder")
    p.add_argument("--min-undrawn", type=int, default=24)
    p.add_argument("--bands", type=int, nargs="*", default=[25, 31, 35])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = pick_device("cpu" if args.cpu else None)
    dirs = [GAME_DATA_DIR / line.strip() for line in (EXP_DIR / args.dataset_txt).read_text()
            .splitlines() if line.strip() and not line.startswith("#")]
    first = load_boxes_net(EXP_DIR / "checkpoints" / args.tag / f"iter{args.iterations[0]}.pt",
                           device)
    dataset = BoxesDataset(dirs, min_undrawn=args.min_undrawn, features=first.features)
    planes, margins, undrawn = exact_positions(dataset)
    edges = [*sorted(args.bands), int(undrawn.max()) + 1]
    bands = [(lo, hi - 1, (undrawn >= lo) & (undrawn < hi)) for lo, hi in zip(edges, edges[1:])]
    print(f"{len(margins)} positions with exact value targets (>= {args.min_undrawn} undrawn) "
          f"from {len(dataset.games)} games; value sign / margin MAE by undrawn band\n")
    print("checkpoint | " + " | ".join(f"{lo}..{hi} (n={int(m.sum())})" for lo, hi, m in bands)
          + " | all")
    for it in args.iterations:
        model = load_boxes_net(EXP_DIR / "checkpoints" / args.tag / f"iter{it}.pt", device)
        win, expected = predict(model, device, planes)
        decided = margins != 0

        def cell(mask: np.ndarray) -> str:
            m = mask & decided
            sign = float(np.mean((win[m] > 0.5) == (margins[m] > 0))) if m.any() else float("nan")
            mae = (float(np.mean(np.abs(expected[mask] - margins[mask]))) if mask.any()
                   else float("nan"))
            return f"{sign:.3f} / {mae:.2f}"

        print(f"iter{it} | " + " | ".join(cell(m) for _, _, m in bands)
              + f" | {cell(np.ones_like(decided))}")


if __name__ == "__main__":
    main()
