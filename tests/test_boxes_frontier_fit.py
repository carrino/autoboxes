"""Fixed replay arms preserve held-out families; canaries use training roots only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from alpha_go.boxes.balanced_policy import clean_validation, input_key, root_subset
from alpha_go.boxes.frontier_fit import run, select_canary
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import save_boxes_net
from alpha_go.boxes.policy_ab_data import prepare
from alpha_go.boxes.policy_labels import LabelSettings
from alpha_go.boxes.policy_target_ab import bundle_files, load_arm, sha256
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.search_probe import load_positions
from alpha_go.boxes.validation import replay_group


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("frontier-fit")
    source = root / "source"
    source.mkdir()
    geo = geometry(3, 3)
    paths = []
    for seed in range(32):
        board = BoxesBoard(3, 3)
        rng = np.random.default_rng(seed)
        grids, players, moves, visits = [], [], [], []
        while not board.is_game_over():
            legal = board.get_legal_moves_flat()
            edge = int(rng.choice(legal))
            grids.append(board.to_numpy())
            players.append(board.player())
            moves.append(geo.edge_rc[edge])
            visits.append([int(e in legal) for e in range(geo.num_edges)])
            board.play_edge(edge)
        score = round(board.score())
        path = source / f"{seed:03d}.npz"
        np.savez_compressed(path, boards=np.stack(grids), to_play=np.array(players, np.int8),
            moves=np.array(moves), mcts_visits=np.array(visits, np.int16),
            mcts_root_values=np.full(len(moves), .5, np.float32),
            result=np.array(f"{'B' if score > 0 else 'W'}+{abs(score)}.0"),
            game=np.array("boxes"), num_moves=len(moves))
        paths.append(path)
    output = root / "prepared"
    prepare(paths, output, LabelSettings(3, 3, 9, 12, 100_000, 100_000, 1 << 12, 2, 2))
    return output


def test_canary_selects_only_distinct_training_inputs(prepared: Path) -> None:
    dataset = load_arm(prepared, "exact", "chains", 0)
    training, held_out = dataset.split_stable(.1, 0)
    validation, _ = clean_validation(training, held_out)
    canary, report = select_canary(training, 12, 5, 42)
    again, repeated = select_canary(training, 12, 5, 42)
    assert report == repeated and len(canary) == len(again) == 5
    keys = [input_key(g["boards"][i], int(g["to_play"][i]), canary.geo)
            for g, indices in zip(canary.games, canary.position_indices) for i in indices]
    assert len(set(keys)) == 5
    training_families = {replay_group(g, training.geo).hex() for g in training.games}
    validation_families = {replay_group(g, validation.geo).hex() for g in validation.games}
    selected_families = {r["family"] for r in report["records"]}
    assert selected_families <= training_families
    assert selected_families.isdisjoint(validation_families)
    assert len(root_subset(canary, 12)) == len(canary)


@pytest.mark.parametrize("arm", ["control", "balanced", "canary"])
def test_actual_training_exports_only_held_out_band_and_preserves_inputs(
    prepared: Path, tmp_path: Path, arm: str,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(17)
    reference = tmp_path / "reference.pt"
    save_boxes_net(BoxesNet(3, 3, channels=8, n_blocks=1, value_hidden=8,
                           features="chains"), reference)
    original = sha256(reference)
    inputs = bundle_files(prepared)
    output = tmp_path / arm
    summary = run(reference, prepared, output, arm, steps=2, batch_size=8,
                  lr=.001, seed=9, cpu=True, min_undrawn=0, canary_band=11, canary_roots=4)
    assert summary["status"] == "complete"
    assert summary["arena_eligible"] == (arm != "canary")
    assert summary["reference_sha256"] == original == sha256(reference)
    assert bundle_files(prepared) == inputs
    assert summary["result"]["steps"] == 2
    checkpoint = output / summary["result"]["checkpoint"]
    assert checkpoint.is_file() and sha256(checkpoint) == summary["result"]["checkpoint_sha256"]
    positions = load_positions(output / "validation-positions.jsonl", 3, 3, 11, 11)
    assert len(positions) == summary["search_validation"]["positions"] > 0
    assert all(24-len(p.prefix) == 11 for p in positions)
    assert "11" in summary["reference"]["validation_roots"]
    assert "11" in summary["result"]["final"]["validation_roots"]
    exposed = summary["result"]["exposures"]
    if arm == "balanced":
        assert exposed["exact_roots"] == 4
    if arm == "canary":
        assert exposed["exact_roots"] == exposed["rows"] == 16
        assert summary["training"]["positions"] == summary["training"]["exact_roots"] == 4
        saved = json.loads((output / "canary-selection.json").read_text())
        assert saved == summary["canary"]
        assert summary["result"]["checkpoint"] == "control.pt"
    else:
        assert summary["training"] == summary["full_training"]
    with pytest.raises(AssertionError, match="output must be fresh"):
        run(reference, prepared, output, arm, steps=2, batch_size=8, cpu=True)


def test_empty_canary_band_is_rejected(prepared: Path) -> None:
    training, _ = load_arm(prepared, "exact", "chains", 0).split_stable(.1, 0)
    with pytest.raises(AssertionError, match="no eligible exact training roots"):
        select_canary(training, 24, 5, 0)
