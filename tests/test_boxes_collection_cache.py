"""Completed collection reuse is bound to checkpoint, settings, and NPZ semantics."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "experiments/2026-10-08_14-42-boxes-3x3-loop/collection_cache.py")
spec = importlib.util.spec_from_file_location("boxes_collection_cache", SCRIPT)
assert spec is not None and spec.loader is not None
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def setup(tmp_path: Path) -> tuple[Path, Path, str, int, Path]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    return tmp_path / "games", checkpoint, "rows=5 sims=1600 seed=123", 2, tmp_path / "logs"


def write_game(path: Path, index: int) -> None:
    np.savez_compressed(path, boards=np.full((2, 3, 3), index, dtype=np.int8),
                        moves=np.array([[0, 1], [1, 0]], dtype=np.int16),
                        to_play=np.array([0, 1], dtype=np.int8), result="B+1.0",
                        mcts_visits=np.array([[1., 2.], [3., 4.]], dtype=np.float32))


def complete(args: tuple[Path, Path, str, int, Path]) -> None:
    assert cache.begin(*args) == "collect"
    for index in range(args[3]):
        write_game(cache.staging_path(args[0]) / f"game-{index}.npz", index)
    assert cache.finish(*args) == "complete"


def test_completion_and_reuse_leave_all_files_unchanged(tmp_path: Path) -> None:
    args = setup(tmp_path)
    complete(args)
    assert not cache.staging_path(args[0]).exists()
    before = {str(p.relative_to(tmp_path)): p.read_bytes()
              for p in tmp_path.rglob("*") if p.is_file()}
    assert cache.begin(*args) == "reuse"
    assert before == {str(p.relative_to(tmp_path)): p.read_bytes()
                      for p in tmp_path.rglob("*") if p.is_file()}
    receipt = json.loads((args[0] / cache.RECEIPT).read_text())
    assert receipt["status"] == "complete" and len(receipt["files"]) == 2


def test_interrupted_staging_is_preserved_under_unique_archive(tmp_path: Path) -> None:
    args = setup(tmp_path)
    for index in range(2):
        assert cache.begin(*args) == "collect"
        (cache.staging_path(args[0]) / "partial.txt").write_text(f"unfinished {index}")
    assert cache.begin(*args) == "collect"
    preserved = sorted(p.read_text() for p in args[4].rglob("partial.txt"))
    assert preserved == ["unfinished 0", "unfinished 1"]
    assert sorted(p.name for p in cache.staging_path(args[0]).iterdir()) == [cache.REQUEST]


@pytest.mark.parametrize("changed", ["checkpoint", "signature", "num_games"])
def test_reuse_rejects_changed_source_or_settings(tmp_path: Path, changed: str) -> None:
    args = setup(tmp_path)
    complete(args)
    games, checkpoint, signature, count, archive = args
    if changed == "checkpoint":
        checkpoint.write_bytes(b"new model")
    elif changed == "signature":
        signature += " different search settings"
    else:
        count += 1
    with pytest.raises(ValueError, match="checkpoint or settings differ"):
        cache.begin(games, checkpoint, signature, count, archive)


@pytest.mark.parametrize("changed", ["boards", "moves", "mcts_visits", "result", "new_metadata"])
def test_reuse_rejects_every_other_array_change(tmp_path: Path, changed: str) -> None:
    args = setup(tmp_path)
    complete(args)
    path = args[0] / "game-0.npz"
    with np.load(path) as archive:
        arrays = dict(archive)
    if changed in ("result", "new_metadata"):
        arrays[changed] = np.asarray("changed")
    else:
        arrays[changed] = arrays[changed] + 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="NPZ content differs"):
        cache.begin(*args)


def test_exact_value_addition_changes_and_zip_rewrite_are_allowed(tmp_path: Path) -> None:
    args = setup(tmp_path)
    complete(args)
    path = args[0] / "game-0.npz"
    for value in (1, -1):
        with np.load(path) as archive:
            arrays = dict(archive)
        arrays["exact_margin_p1"] = np.full(2, value, dtype=np.int16)
        # Array order and compression are not part of semantic identity.
        np.savez(path, **dict(reversed(list(arrays.items()))))
        assert cache.begin(*args) == "reuse"


def test_missing_games_cannot_be_published(tmp_path: Path) -> None:
    args = setup(tmp_path)
    cache.begin(*args)
    staging = cache.staging_path(args[0])
    write_game(staging / "game-0.npz", 0)
    with pytest.raises(ValueError, match="1 NPZ games; expected 2"):
        cache.finish(*args)
    assert staging.exists() and not args[0].exists()
    assert not (staging / cache.RECEIPT).exists()
    write_game(staging / "game-1.npz", 1)
    assert cache.finish(*args) == "complete"


@pytest.mark.parametrize("changed", ["checkpoint", "signature", "request"])
def test_finish_rejects_mid_collection_changes(tmp_path: Path, changed: str) -> None:
    args = setup(tmp_path)
    cache.begin(*args)
    games, checkpoint, signature, count, archive = args
    for index in range(count):
        write_game(cache.staging_path(games) / f"game-{index}.npz", index)
    if changed == "checkpoint":
        checkpoint.write_bytes(b"replacement model")
    elif changed == "signature":
        signature += " changed"
    else:
        (cache.staging_path(games) / cache.REQUEST).write_text("{}")
    with pytest.raises(ValueError, match="changed since begin"):
        cache.finish(games, checkpoint, signature, count, archive)
    assert not games.exists()


def test_existing_collection_without_receipt_is_never_replaced(tmp_path: Path) -> None:
    args = setup(tmp_path)
    args[0].mkdir()
    write_game(args[0] / "game.npz", 0)
    with pytest.raises(ValueError, match="no completion receipt"):
        cache.begin(*args)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        cache.finish(*args)
    assert (args[0] / "game.npz").is_file()


def test_semantic_hash_distinguishes_array_shapes(tmp_path: Path) -> None:
    path = tmp_path / "shape.npz"
    np.savez(path, marker=np.array(1, dtype=np.int64))
    scalar = cache.semantic_hash(path)
    np.savez(path, marker=np.array([1], dtype=np.int64))
    assert cache.semantic_hash(path) != scalar


def test_cli_stdout_contains_only_the_action(tmp_path: Path) -> None:
    games, checkpoint, signature, count, archive = setup(tmp_path)
    command = [sys.executable, str(SCRIPT), "begin", "--games", str(games),
               "--checkpoint", str(checkpoint), "--signature", signature,
               "--num-games", str(count), "--archive", str(archive)]
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    second = subprocess.run(command, capture_output=True, text=True, check=True)
    assert first.stdout == second.stdout == "collect\n"
    assert not first.stderr and "Preserved interrupted collection" in second.stderr
