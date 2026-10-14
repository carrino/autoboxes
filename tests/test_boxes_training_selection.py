"""Mixed games retain replay lineage while selecting only searched learner turns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.relabel import EXACT_UNKNOWN
from alpha_go.boxes.rules import geometry
from alpha_go.boxes.validation import replay_group, validation_game
from tests.test_boxes_relabel import random_game


def test_mask_selects_source_rows_and_exact_targets(tmp_path: Path) -> None:
    game = random_game(2, 3, 4)
    np.savez_compressed(tmp_path / "game.npz", **game)
    original = BoxesDataset([tmp_path], min_undrawn=6)
    count = len(game["moves"])
    mask = np.arange(count) % 3 == 1
    exact = np.full(count, EXACT_UNKNOWN, dtype=np.int16)
    exact[[1, 4, 7, count - 1]] = 0
    np.savez_compressed(tmp_path / "game.npz", **game, train_mask=mask, exact_margin_p1=exact)
    selected = BoxesDataset([tmp_path], min_undrawn=6)
    indices = [i for i in range(len(original)) if mask[i]]
    assert len(selected) == len(indices)
    assert selected.num_exact() == int((exact[indices] != EXACT_UNKNOWN).sum())
    for i, source in enumerate(indices):
        assert torch.equal(selected[i]["planes"], original[source]["planes"])
        assert torch.equal(selected[i]["policy"], original[source]["policy"])
    train, val = selected.split_stable(0.2)
    assert len(train) + len(val) == len(selected)
    assert len(selected.games[0]["moves"]) == count  # complete trajectory survives


@pytest.mark.parametrize("mask", [np.ones(17), np.ones(3, dtype=bool)])
def test_invalid_training_mask_is_rejected(tmp_path: Path, mask: np.ndarray) -> None:
    np.savez_compressed(tmp_path / "game.npz", **random_game(2, 3, 4), train_mask=mask)
    with pytest.raises(ValueError, match="train_mask"):
        BoxesDataset([tmp_path])


def test_derived_policy_labels_keep_original_split() -> None:
    geo = geometry(2, 3)
    parent = random_game(2, 3, 6)
    child = {"source_replay_group": np.array(replay_group(parent, geo).hex())}
    assert replay_group(child, geo) == replay_group(parent, geo)
    byte_child = {"source_replay_group": np.array(replay_group(parent, geo).hex().encode())}
    assert replay_group(byte_child, geo) == replay_group(parent, geo)
    for seed in range(10):
        assert validation_game(child, geo, 0.2, seed) == validation_game(parent, geo, 0.2, seed)


@pytest.mark.parametrize("value", ["", "abc", "xy", "aa bb", np.array(["aa"])])
def test_invalid_inherited_group_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="source_replay_group"):
        replay_group({"source_replay_group": value}, geometry(2, 3))
