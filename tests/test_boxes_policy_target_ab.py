"""The policy-target experiment changes targets, not data, updates or validation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from alpha_go.boxes.agents import BoxesRandomAgent
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import load_boxes_net, save_boxes_net
from alpha_go.boxes.policy_ab_data import load_game, prepare
from alpha_go.boxes.policy_labels import LabelSettings
from alpha_go.boxes.policy_target_ab import (
    TrainSettings,
    bundle_files,
    evaluate,
    load_arm,
    matched_datasets,
    run,
    sha256,
)
from alpha_go.boxes.rules import geometry
from alpha_go.boxes.validation import replay_group
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("matched-policy")
    (root / "games").mkdir()
    paths = []
    geo = geometry(2, 2)
    for seed in range(2, 14):
        record = play_game(BoxesRandomAgent(), BoxesRandomAgent(), game=get_game("boxes"),
                           board_size=2, seed=seed, render_debug_on_error=False)
        path = save_game_data(record, root / "games", seed, "matched-test")
        game = load_game(path)
        game["mcts_visits"] = np.asarray([
            [int(grid[r, c] == 0) for r, c in geo.edge_rc] for grid in game["boards"]], np.int16)
        game["mcts_root_values"] = np.full(len(game["moves"]), 0.5, np.float32)
        np.savez_compressed(path, **game)
        paths.append(path)
    output = root / "prepared"
    prepare(paths, output, LabelSettings(2, 2, 1, 12, 1_000_000, 1_000_000, 1 << 12, 2, 2))
    return output


def test_integrity_and_relocation(prepared: Path, tmp_path: Path) -> None:
    hashes = bundle_files(prepared)
    relocated = tmp_path / "moved"
    shutil.copytree(prepared, relocated)
    assert bundle_files(relocated) == hashes
    a, b = [load_arm(relocated, name, "chains", 0) for name in ("control", "exact")]
    counts = matched_datasets(a, b)
    assert counts["changed_policy_rows"] > 0 and counts["value_only_rows"] > 0
    extra = relocated / "control/extra.npz"
    shutil.copyfile(next((relocated / "control").glob("*.npz")), extra)
    with pytest.raises(AssertionError, match="unrecorded"):
        bundle_files(relocated)
    extra.unlink()
    manifest = relocated / "control-manifest.txt"
    manifest.write_text("exact\nchildren\n")
    with pytest.raises(AssertionError, match="artifact changed"):
        bundle_files(relocated)


@pytest.mark.parametrize("field", ["exact_margin_p1", "boards", "policy_mask", "train_mask"])
def test_arm_mismatches_rejected(prepared: Path, field: str) -> None:
    a, b = [load_arm(prepared, name, "chains", 0) for name in ("control", "exact")]
    game = b.games[0]
    if field in game:
        game[field] = game[field].copy()
        game[field].flat[0] += 1
    else:
        game[field] = np.ones(len(game["boards"]), bool)
    with pytest.raises(AssertionError, match="arms differ|different field"):
        matched_datasets(a, b)


def test_unmarked_policy_change_rejected(prepared: Path) -> None:
    a, b = [load_arm(prepared, name, "chains", 0) for name in ("control", "exact")]
    game = b.games[0]
    row = int(np.flatnonzero(~game["policy_ab_root_mask"])[0])
    game["mcts_visits"][row] = 0
    game["mcts_visits"][row, 0] = 1
    with pytest.raises(AssertionError, match="outside selected"):
        matched_datasets(a, b)


def test_training_same_updates_and_common_exact_validation(prepared: Path, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    reference = tmp_path / "reference.pt"
    save_boxes_net(BoxesNet(2, 2, channels=8, n_blocks=1, value_hidden=8,
                           features="chains"), reference)
    original = sha256(reference)
    settings = TrainSettings(epochs=2, batch_size=32, val_fraction=0.4, min_undrawn=0)
    output = tmp_path / "result"
    summary = run(reference, prepared, output, settings, cpu=True)
    assert summary["status"] == "complete"
    assert sha256(reference) == original == sha256(output / "reference.pt")
    assert summary["training"]["changed_policy_rows"] > 0
    assert summary["validation"]["exact_roots"] > 0
    a, b = summary["arms"]["control"], summary["arms"]["exact"]
    assert a["steps"] == b["steps"] > 1
    assert a["update_trace_sha256"] == b["update_trace_sha256"]
    dataset = load_arm(prepared, "exact", "chains", 0)
    train, val = dataset.split_stable(settings.val_fraction, settings.split_seed)
    train_groups = {replay_group(g, dataset.geo) for g in train.games}
    val_groups = {replay_group(g, dataset.geo) for g in val.games}
    assert train_groups.isdisjoint(val_groups)
    for name in ["reference", "control", "exact"]:
        model = load_boxes_net(output / f"{name}.pt", torch.device("cpu"))
        metrics = evaluate(model, val, settings.batch_size, torch.device("cpu"))
        saved = summary["reference_validation"] if name == "reference" else (
            summary["arms"][name]["validation"])
        assert metrics == saved
        assert metrics["policy_positions"] < metrics["positions"]
        assert 0 <= metrics["optimal_mass"] <= 1
    assert json.loads((output / "summary.json").read_text()) == summary
    with pytest.raises(AssertionError, match="fresh"):
        run(reference, prepared, output, settings, cpu=True)


def test_invalid_budget_rejected() -> None:
    with pytest.raises(AssertionError):
        TrainSettings(epochs=0).validate()
