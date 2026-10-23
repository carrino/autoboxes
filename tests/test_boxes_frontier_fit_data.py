"""Saved live-loop labels become a portable fitting bundle without solving again."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alpha_go.boxes import policy_loop_data
from alpha_go.boxes.agents import BoxesRandomAgent
from alpha_go.boxes.balanced_policy import export_validation, read_labels, verify_root_targets
from alpha_go.boxes.frontier_fit_data import prepare, verify_sources
from alpha_go.boxes.policy_ab_data import load_game, sha256
from alpha_go.boxes.policy_labels import LabelSettings
from alpha_go.boxes.policy_target_ab import bundle_files, load_arm
from alpha_go.boxes.rules import geometry
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


def saved_iteration(folder: Path, seed: int) -> Path:
    """Prepare real tiny-board exact targets with synthetic legal search visits."""
    (folder / "selfplay").mkdir(parents=True)
    record = play_game(BoxesRandomAgent(), BoxesRandomAgent(), game=get_game("boxes"),
                       board_size=2, seed=seed, render_debug_on_error=False)
    source = save_game_data(record, folder / "selfplay", seed, "fit-data-test")
    game = load_game(source)
    geo = geometry(2, 2)
    game["mcts_visits"] = np.asarray([
        [int(grid[r, c] == 0) for r, c in geo.edge_rc] for grid in game["boards"]], np.int16)
    game["mcts_root_values"] = np.full(len(game["moves"]), .5, np.float32)
    game["train_mask"] = np.ones(len(game["moves"]), bool)
    np.savez_compressed(source, **game)
    output = folder / "policy"
    settings = LabelSettings(2, 2, 1, 12, 1_000_000, 1_000_000, 1 << 12, 2, 2)
    policy_loop_data.prepare_iteration([source], output, settings)
    return output


def test_nested_copy_preserves_targets_and_exports_compatible_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directories = [saved_iteration(tmp_path / str(seed), seed) for seed in (2, 3)]
    original = verify_sources(directories)

    def never_solve(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("saved exact targets must not be solved again")

    monkeypatch.setattr(policy_loop_data, "solve_game", never_solve)
    output = tmp_path / "prepared"
    result = prepare(directories, output)
    assert result["status"] == "complete" and result["counts"]["replaced_roots"] > 0
    assert result["source_sha256"] == original == verify_sources(directories)
    assert (output / "control-manifest.txt").read_bytes() == (
        output / "exact-manifest.txt").read_bytes()
    for index, source in enumerate(directories):
        for path in source.rglob("*.npz"):
            assert sha256(output / "iterations" / f"iteration-{index:04d}"
                          / path.relative_to(source)) == sha256(path)
    # Both source bundles used game-0000000/source-0000000: nested copies must survive.
    assert len(list(output.glob("iterations/*/games/game-0000000/exact/source-0000000.npz"))) == 2
    renamed = tmp_path / "moved"
    output.rename(renamed)
    assert bundle_files(renamed)
    data = load_arm(renamed, "exact", "chains", 0)
    labels, source_rows = read_labels(renamed, geometry(2, 2))
    verify_root_targets(data, labels)
    exported = tmp_path / "export"
    exported.mkdir()
    report = export_validation(data, source_rows, exported)
    assert report["positions"] > 0 and report["cached_states"] > 0
    assert json.loads((exported / "validation-labels.json").read_text())["rows"] == 2


@pytest.mark.parametrize("mutation", [
    "incomplete", "identity", "manifest", "receipt", "artifact", "source",
    "missing_bundle", "extra_file", "unpublished_npz", "counts",
])
def test_rejects_changed_or_incomplete_source_before_output(
    tmp_path: Path, mutation: str,
) -> None:
    source = saved_iteration(tmp_path / "source", 2)
    bundle = source / "games/game-0000000"
    metadata_path = source / "preparation.json"
    metadata = json.loads(metadata_path.read_text())
    if mutation == "incomplete":
        metadata["status"] = "running"
        metadata_path.write_text(json.dumps(metadata))
    elif mutation == "identity":
        metadata["identity"]["settings"]["min_regret"] += 1
        metadata_path.write_text(json.dumps(metadata))
    elif mutation == "manifest":
        (source / "dataset.txt").write_text("unexpected\n")
    elif mutation == "receipt":
        (bundle / "receipt.json").write_text("{}\n")
    elif mutation == "artifact":
        next((bundle / "exact").glob("*.npz")).write_bytes(b"changed")
    elif mutation == "source":
        Path(metadata["identity"]["sources"][0]["path"]).write_bytes(b"changed")
    elif mutation == "missing_bundle":
        shutil.rmtree(bundle)
    elif mutation == "extra_file":
        (bundle / "unexpected.npz").write_bytes(b"changed")
    elif mutation == "unpublished_npz":
        (source / "unexpected.npz").write_bytes(b"changed")
    else:
        metadata["counts"]["replaced_roots"] += 1
        metadata_path.write_text(json.dumps(metadata))
    output = tmp_path / "output"
    with pytest.raises(AssertionError):
        prepare([source], output)
    assert not output.exists()


def test_duplicate_iteration_and_duplicate_source_are_rejected(tmp_path: Path) -> None:
    source = saved_iteration(tmp_path / "source", 2)
    with pytest.raises(AssertionError, match="distinct policy iterations"):
        verify_sources([source, source])
    second = tmp_path / "duplicate"
    shutil.copytree(source, second)
    with pytest.raises(AssertionError, match="duplicate source game"):
        prepare([source, second], tmp_path / "output")
