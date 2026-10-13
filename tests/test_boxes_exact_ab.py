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


def test_reuse_keeps_exact_targets_only_in_treatment_and_never_solves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    boards = np.zeros((3, 5, 5), dtype=np.int8)
    boards[1:, 0, 1] = boards[1:, 0, 3] = 1
    boards[2, 1, 0] = 1
    visits = np.arange(36).reshape(3, 12)
    path = source / "game.npz"
    np.savez_compressed(path, moves=np.array([[0, 1], [0, 3], [1, 0]]), boards=boards,
                        to_play=np.array([0, 1, 0]), result="B+2.0", mcts_visits=visits,
                        exact_margin_p1=np.array([2, -2, ab.EXACT_UNKNOWN], dtype=np.int16))
    before = path.read_bytes()
    arms, paths, _ = ab.snapshot([source], tmp_path / "snapshot", reuse_reach=10)
    for arm, directory in arms.items():
        with np.load(directory / "0/game.npz") as game:
            assert np.array_equal(game["mcts_visits"], visits)
            if arm == "control":
                assert "exact_margin_p1" not in game
            else:
                assert game["exact_margin_p1"].tolist() == [ab.EXACT_UNKNOWN, -2, ab.EXACT_UNKNOWN]

    def forbidden_solver(*args: object) -> None:
        pytest.fail("reuse must not construct a solver")

    monkeypatch.setattr(ab.alpha_go_cpp, "BoxesSolver", forbidden_solver)
    assert ab.relabel_games(paths, 2, 2, 10, 1, 1, reuse=True) == {
        "9": {"known": 0, "unknown": 1}, "10": {"known": 1, "unknown": 0},
    }
    assert path.read_bytes() == before


@pytest.mark.parametrize("labels", [np.array([1, 2]), np.array([1.5]), np.array([99])])
def test_reuse_rejects_invalid_exact_arrays(tmp_path: Path, labels: object) -> None:
    source = tmp_path / "source"
    source.mkdir()
    np.savez_compressed(source / "game.npz", moves=np.array([[0, 1]]),
                        boards=np.zeros((1, 5, 5), dtype=np.int8), exact_margin_p1=labels)
    with pytest.raises(AssertionError):
        ab.snapshot([source], tmp_path / "snapshot", reuse_reach=12)


def test_reuse_solves_only_missing_games_and_preserves_saved_unknowns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    moves = np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int16)
    boards = np.zeros((4, 3, 3), dtype=np.int8)
    for k, (row, col) in enumerate(moves[:-1]):
        boards[k + 1:, row, col] = 1
    common = {"moves": moves, "boards": boards, "to_play": np.array([0, 1, 0, 1]),
              "result": "W+1.0", "mcts_visits": np.arange(16).reshape(4, 4)}
    saved = np.array([-1, ab.EXACT_UNKNOWN, -1, -1], dtype=np.int16)
    np.savez_compressed(source / "saved.npz", **common, exact_margin_p1=saved)
    np.savez_compressed(source / "missing.npz", **common)
    np.savez_compressed(source / "static.npz", **common, exact_margin_p1=saved,
                        termination="solver")
    before = {p: p.read_bytes() for p in source.glob("*.npz")}
    arms, paths, inputs = ab.snapshot([source], tmp_path / "snapshot", reuse_reach=4)
    assert {Path(item["source"]).name: item["label_action"] for item in inputs} == {
        "saved.npz": "reuse", "missing.npz": "solve", "static.npz": "static",
    }
    saved_copy = arms["exact"] / "0/saved.npz"
    saved_bytes = saved_copy.read_bytes()
    calls = []
    real_relabel = ab.relabel_file

    def record_relabel(path: Path, *args: object) -> object:
        calls.append(path)
        return real_relabel(path, *args)

    monkeypatch.setattr(ab, "relabel_file", record_relabel)
    coverage = ab.relabel_games(paths, 1, 1, 4, 2, 1000, reuse=True)
    assert calls == [arms["exact"] / "0/missing.npz"]
    assert coverage == {
        "1": {"known": 2, "unknown": 0}, "2": {"known": 2, "unknown": 0},
        "3": {"known": 1, "unknown": 1}, "4": {"known": 2, "unknown": 0},
    }
    assert saved_copy.read_bytes() == saved_bytes
    for path, raw in before.items():
        assert path.read_bytes() == raw
    for name in ("saved.npz", "missing.npz"):
        with np.load(arms["control"] / "0" / name) as game:
            assert "exact_margin_p1" not in game
        with np.load(arms["exact"] / "0" / name) as game:
            assert np.array_equal(game["mcts_visits"], common["mcts_visits"])
    for arm in arms.values():
        with np.load(arm / "0/static.npz") as game:
            assert np.array_equal(game["exact_margin_p1"], saved)
