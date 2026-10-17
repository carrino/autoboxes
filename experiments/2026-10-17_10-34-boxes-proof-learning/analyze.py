"""Score the frozen endpoint, excluding feasibility-pilot overlap from heldout reads."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    data = args.data
    roots = [json.loads(s) for s in (data / "roots.jsonl").read_text().splitlines()]
    feasibility = json.loads((Path(__file__).parent / "data/feasibility.json").read_text())
    excluded = set(feasibility["families"])
    config = json.loads((data / "config.json").read_text())
    seeds = config["seeds"]
    output, bootstrap = [], {}
    all_arrays = {}
    for split in ("train", "heldout"):
        group = [r for r in roots if r["split"] == split]
        keep = np.array([split == "train" or r["family"] not in excluded for r in group])
        for seed in seeds:
            for arm in ("raw", "filtered"):
                metrics = json.loads((data / f"per-root-{seed}-{arm}-{split}.json").read_text())
                arrays = {k: np.asarray(v)[keep] for k, v in metrics.items()}
                all_arrays[seed, arm, split] = arrays
                output.append(dict(seed=seed, arm=arm, split=split, positions=int(keep.sum()),
                                   **{k: float(v.mean()) for k, v in arrays.items()}))
        if split == "heldout":
            rng = np.random.default_rng(20261101)
            # One root per canonical family; average across the three fixed model
            # seeds first. Root bootstrap is conditional on these fitted nets.
            sample = rng.integers(int(keep.sum()), size=(10000, int(keep.sum())))
            for metric in arrays:
                delta = np.mean([all_arrays[s, "filtered", split][metric]
                                 - all_arrays[s, "raw", split][metric] for s in seeds], axis=0)
                bootstrap[metric] = dict(filtered_minus_raw=float(delta.mean()),
                                        ci95=np.quantile(delta[sample].mean(axis=1),
                                                         [0.025, 0.975]).tolist())
    with (data / "primary-metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    targets = {}
    for split in ("train", "heldout"):
        group = [r for r in roots if r["split"] == split
                 and (split == "train" or r["family"] not in excluded)]
        target_rows = []
        for row in group:
            best = max(row["children"].values())
            target_rows.append(dict(
                changed=int(row["raw"] != row["filtered"]),
                raw_inferior_mass=1 - sum(row["raw"][e] for e in row["optimum"])
                / sum(row["raw"]),
                filtered_inferior_mass=1 - sum(row["filtered"][e] for e in row["optimum"])
                / sum(row["filtered"]),
                search_regret=best - row["children"][str(row["selected"])],
                search_optimal=int(row["selected"] in row["optimum"]),
                uniform_optimal_mass=len(row["optimum"]) / len(row["children"])))
        targets[split] = dict(positions=len(group),
                             **{k: float(np.mean([r[k] for r in target_rows]))
                                for k in target_rows[0]})
    excluded_heldout = [dict(id=r["id"], family=r["family"]) for r in roots
                       if r["split"] == "heldout" and r["family"] in excluded]
    game_seeds = sorted(r["seed"] for r in roots)
    summary = dict(
        seeds=seeds, training_positions=sum(r["split"] == "train" for r in roots),
        original_heldout_positions=sum(r["split"] == "heldout" for r in roots),
        minimum_game_seed_gap=min(b - a for a, b in zip(game_seeds, game_seeds[1:])),
        configured_seed_stride=config.get("seed_stride", 1),
        regret_units="score-margin points",
        primary_heldout_positions=sum(r["split"] == "heldout" and r["family"] not in excluded
                                      for r in roots),
        excluded_heldout=excluded_heldout,
        exclusion_reason=("Canonical masks seen in feasibility pilot; no replacement or retuning"
                          if excluded_heldout else "None; no pilot-family overlap"),
        training_pilot_overlap=sum(r["split"] == "train" and r["family"] in excluded
                                  for r in roots),
        bootstrap=bootstrap, targets=targets,
        model_means={split: {arm: {metric: float(np.mean([
            r[metric] for r in output if r["split"] == split and r["arm"] == arm]))
            for metric in bootstrap} for arm in ("raw", "filtered")}
            for split in ("train", "heldout")})
    (data / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
