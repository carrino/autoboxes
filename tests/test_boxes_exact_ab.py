"""The A/B preparation preserves source files and excludes non-game label collections."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments/2026-10-08_14-42-boxes-3x3-loop"
sys.path.insert(0, str(EXP))
spec = importlib.util.spec_from_file_location("boxes_exact_ab", EXP / "exact_ab.py")
assert spec is not None and spec.loader is not None
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def test_snapshot_preserves_sources_and_policies(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    common = {"moves": np.array([[0, 1]], np.int16), "boards": np.zeros((1, 5, 5), np.int8),
              "mcts_visits": np.arange(12).reshape(1, 12), "result": "B+2.0",
              "to_play": np.array([0]), "exact_margin_p1": np.array([-2], np.int16)}
    np.savez_compressed(source / "game.npz", **common, termination="solved")
    np.savez_compressed(source / "labels.npz", **common, termination="solver")
    before = {p: p.read_bytes() for p in source.glob("*.npz")}
    arms, paths, inputs = ab.snapshot([source], tmp_path / "snapshot")
    assert [p.name for p in paths] == ["game.npz"] and len(inputs) == 2
    for p, contents in before.items():
        assert p.read_bytes() == contents
    for arm in arms.values():
        with np.load(arm / "0/game.npz") as copy:
            assert "exact_margin_p1" not in copy
            assert np.array_equal(copy["mcts_visits"], common["mcts_visits"])
        with np.load(arm / "0/labels.npz") as copy:
            assert copy["exact_margin_p1"][0] == -2
    with pytest.raises(FileExistsError):
        ab.snapshot([source], tmp_path / "snapshot")


def test_missing_manifest_fails_before_creating_outputs(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="missing replay manifest"):
        ab.manifest_directories(tmp_path / "missing.txt", tmp_path)
