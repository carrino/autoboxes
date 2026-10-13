"""Verify each output NPZ against shard logs and summarize exact label distributions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def verify(status_path: Path) -> dict:
    """Check counts, policy normalization, legality, margins, and distinct positions."""
    status = json.loads(status_path.read_text())
    assert not status["failed"] and not status["running_shards"]
    output = Path(status["output"])
    paths = sorted(output.glob("*.npz"))
    expected = {entry["shard"]: entry for entry in status["results"]}
    assert len(expected) == status["shards"]
    assert all(entry["exit_code"] == 0 for entry in expected.values())
    for entry in expected.values():
        text = Path(entry["log"]).read_text()
        match = re.search(r"(\d+) positions labelled \((\d+) skipped on budget\)", text)
        assert match and (int(match[1]), int(match[2])) == (entry["labelled"], entry["skipped"])
        assert entry["sampled"] == entry["labelled"] + entry["skipped"]
    counts = Counter()
    files = Counter()
    undrawn_counts = Counter()
    margins_p1 = Counter()
    margins_mover = Counter()
    position_keys = set()
    manifest = []
    maximum_error = 0.0
    for path in paths:
        shard = int(re.search(r"-s(\d+)-", path.name)[1])
        assert shard in expected
        with np.load(path, allow_pickle=False) as data:
            # Materialize every stored array, not just the NPZ directory.
            arrays = {key: data[key] for key in data.files}
        boards = arrays["boards"]
        n = int(arrays["num_moves"])
        assert n > 0 and boards.shape == (n, 11, 11)
        to_play = arrays["to_play"]
        assert to_play.shape == (n,) and np.isin(to_play, [0, 1]).all()
        assert arrays["moves"].shape == (n, 2)
        assert str(arrays["game"]) == "boxes" and str(arrays["termination"]) == "solver"
        visits = arrays["mcts_visits"].astype(np.float32)
        assert visits.shape == (n, 60) and np.isfinite(visits).all()
        assert np.isin(visits, [0, 1]).all()
        row_sums = visits.sum(axis=1)
        assert (row_sums > 0).all()
        policy = visits / row_sums[:, None]
        assert np.isfinite(policy).all()
        error = float(np.max(np.abs(policy.sum(axis=1) - 1.0)))
        assert np.allclose(policy.sum(axis=1), 1.0, rtol=0, atol=1e-6)
        maximum_error = max(maximum_error, error)
        drawn = np.concatenate([(boards[:, 0::2, 1::2] == 1).reshape(n, -1),
                                (boards[:, 1::2, 0::2] == 1).reshape(n, -1)], axis=1)
        assert (visits[drawn] == 0).all()
        undrawn = 60 - drawn.sum(axis=1)
        assert ((undrawn >= 31) & (undrawn <= 36)).all()
        assert (np.diff(undrawn) <= 0).all()
        result = str(arrays["result"])
        match = re.fullmatch(r"([BW])\+(\d+)\.0", result)
        assert match, result
        margin = int(match[2]) * (1 if match[1] == "B" else -1)
        assert abs(margin) <= 25 and margin % 2 == 1
        mover_margins = np.where(to_play == 0, margin, -margin)
        assert np.array_equal(arrays["mcts_root_values"], (mover_margins > 0).astype(np.float32))
        boxes = boards[:, 1::2, 1::2]
        current_p1_margin = (boxes == 1).sum(axis=(1, 2)) - (boxes == 2).sum(axis=(1, 2))
        for edges, mover, current in zip(drawn, to_play, current_p1_margin):
            key = (np.packbits(edges).tobytes(), int(mover), int(current))
            assert key not in position_keys, "duplicate sampled position across output files"
            position_keys.add(key)
        counts[shard] += n
        files[shard] += 1
        undrawn_counts.update(map(int, undrawn))
        margins_p1[margin] += n
        margins_mover.update(map(int, mover_margins))
        manifest.append({"file": path.name, "shard": shard, "positions": n,
                         "margin_p1": margin, "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "maximum_policy_sum_error": error})
    for shard, entry in expected.items():
        assert counts[shard] == entry["labelled"], (shard, counts[shard], entry)
        assert files[shard] == entry["files"], (shard, files[shard], entry)
    summary = {"output": str(output), "files": len(paths), "positions": sum(counts.values()),
               "skipped": sum(entry["skipped"] for entry in expected.values()),
               "sampled": sum(entry["sampled"] for entry in expected.values()),
               "wall_seconds": status["wall_seconds"], "node_budget": status["node_budget"],
               "positions_per_shard": dict(sorted(counts.items())),
               "positions_per_undrawn": dict(sorted(undrawn_counts.items())),
               "final_margins_player_1": dict(sorted(margins_p1.items())),
               "final_margins_side_to_move": dict(sorted(margins_mover.items())),
               "maximum_policy_sum_error": maximum_error, "verification_passed": True}
    prefix = status_path.parent / status["phase"]
    Path(str(prefix) + "-verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    with Path(str(prefix) + "-files.csv").open("w", newline="") as stream:
        fields = ["file", "shard", "positions", "margin_p1", "bytes", "sha256",
                  "maximum_policy_sum_error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.status), indent=2))
