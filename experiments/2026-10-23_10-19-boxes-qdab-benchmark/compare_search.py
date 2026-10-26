"""Compare saved 400/1600-simulation matches on identical opening pairs.

Run verify_benchmark.py separately for each archive before using this comparison.
Usage: python compare_search.py OLD_RUN NEW_RUN OUTPUT_DIRECTORY
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    """Resample complete opening clusters, including both seats and both runs."""
    old, new, output = map(Path, sys.argv[1:])
    configs = [json.loads((path / "config.json").read_text()) for path in (old, new)]
    source_changes = {
        key: [configs[0]["code_sha256"][key], configs[1]["code_sha256"].get(key)]
        for key in configs[0]["code_sha256"]
        if configs[0]["code_sha256"][key] != configs[1]["code_sha256"].get(key)
    }
    assert not source_changes, source_changes
    for key in configs[0]:
        if key not in {"checkpoint", "source_checkpoint", "code_commit", "code_sha256", "search"}:
            assert configs[0][key] == configs[1][key], key
    assert configs[1]["search"] == {**configs[0]["search"], "num_simulations": 1600}
    assert configs[0]["search"]["num_simulations"] == 400
    games = [
        [json.loads(path.read_text()) for path in sorted((run / "games").glob("*.json"))]
        for run in (old, new)
    ]
    assert len(games[0]) == len(games[1]) == 120
    assert [game["slot"] for game in games[0]] == [game["slot"] for game in games[1]]
    wins = np.array(
        [
            [int(a["a_margin"] > 0), int(b["a_margin"] > 0)]
            for a, b in zip(games[0][:100], games[1][:100])
        ]
    ).reshape(50, 2, 2)
    pair_scores = wins.mean(axis=1)
    seed, resamples = 20261024, 100000
    indices = np.random.default_rng(seed).integers(0, 50, (resamples, 50))
    samples = pair_scores[indices].mean(axis=1)
    delta_samples = samples[:, 1] - samples[:, 0]
    result = {
        "old_run": str(old),
        "new_run": str(new),
        "checkpoint_sha256": configs[0]["checkpoint_sha256"],
        "config_sha256": [
            hashlib.sha256((p / "config.json").read_bytes()).hexdigest() for p in (old, new)
        ],
        "source_changes": source_changes,
        "source_additions": sorted(
            configs[1]["code_sha256"].keys() - configs[0]["code_sha256"].keys()
        ),
        "opening_pairs": 50,
        "bootstrap_seed": seed,
        "resamples": resamples,
        "wins": wins.sum(axis=(0, 1)).tolist(),
        "score_delta": float((pair_scores[:, 1] - pair_scores[:, 0]).mean()),
        "score_delta_ci95": np.quantile(delta_samples, [0.025, 0.975]).tolist(),
        "transitions": {
            f"{i}->{j}": int(((wins[:, :, 0] == i) & (wins[:, :, 1] == j)).sum())
            for i in (0, 1)
            for j in (0, 1)
        },
        "seats": {},
        "scope": "Matched opening-family percentile bootstrap, four games per cluster. "
        "Observed configured-strength difference, not equal-time or a learned gain. "
        "Qdab's uncontrolled randomness and timing vary between runs. "
        "One realization per setting/opening does not isolate opponent variation. "
        "Empty-board repeats excluded; seat results are exploratory.",
    }
    for seat, label in enumerate(("first", "second")):
        values = wins[:, seat, :]
        seat_samples = values[indices].mean(axis=1)
        result["seats"][label] = {
            "wins": values.sum(axis=0).tolist(),
            "games_each": 50,
            "delta": float((values[:, 1] - values[:, 0]).mean()),
            "delta_ci95": np.quantile(
                seat_samples[:, 1] - seat_samples[:, 0], [0.025, 0.975]
            ).tolist(),
        }
    rows = []
    for pair in range(50):
        old_pair, new_pair = games[0][2 * pair : 2 * pair + 2], games[1][2 * pair : 2 * pair + 2]
        assert all(g["slot"]["pair"] == pair for g in old_pair + new_pair)
        rows.append(
            {
                "pair": pair,
                "old_first_margin": old_pair[0]["a_margin"],
                "old_second_margin": old_pair[1]["a_margin"],
                "new_first_margin": new_pair[0]["a_margin"],
                "new_second_margin": new_pair[1]["a_margin"],
                "old_score": pair_scores[pair, 0],
                "new_score": pair_scores[pair, 1],
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "search-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "matched-opening-pairs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
