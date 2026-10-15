"""An arena commit survives a missing timing marker without replaying promotion."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1]
          / "experiments/2026-10-08_14-42-boxes-3x3-loop/arena_completed.py")
spec = importlib.util.spec_from_file_location("boxes_arena_completed", SCRIPT)
assert spec is not None and spec.loader is not None
arena = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arena)


def setup(tmp_path: Path) -> tuple[Path, Path, int, Path]:
    checkpoint = tmp_path / "iter1.pt"
    checkpoint.write_bytes(b"candidate checkpoint")
    return tmp_path / "league.json", checkpoint, 1, tmp_path / "timing/it1.json"


def save_commit(args: tuple[Path, Path, int, Path]) -> None:
    state, checkpoint, iteration, _ = args
    entry = {"iteration": iteration,
             "candidate_checkpoint_sha256": arena.checkpoint_hash(checkpoint)}
    state.write_text(json.dumps({"champion": iteration, "history": [entry]}))


def test_missing_league_or_missing_iteration_is_not_complete(tmp_path: Path) -> None:
    args = setup(tmp_path)
    assert not arena.completed(*args)
    args[0].write_text('{"champion": 0, "history": [{"iteration": 0}]}')
    args[1].unlink()  # No new checkpoint is required before training this iteration.
    assert not arena.completed(*args)
    assert not args[3].exists()


def test_committed_arena_recovers_missing_timing_without_touching_league(tmp_path: Path) -> None:
    args = setup(tmp_path)
    save_commit(args)
    before = args[0].read_bytes()
    assert arena.completed(*args)
    assert json.loads(args[3].read_text()) == {"iteration": 1, "recovered_from_league": True}
    assert args[0].read_bytes() == before
    assert not args[3].with_name(args[3].name + ".tmp").exists()


def test_existing_timing_is_preserved(tmp_path: Path) -> None:
    args = setup(tmp_path)
    save_commit(args)
    args[3].parent.mkdir()
    existing = '{"iteration":1,"collect":20,"train":3,"arena":7}\n'
    args[3].write_text(existing)
    assert arena.completed(*args)
    assert args[3].read_text() == existing


def test_changed_checkpoint_is_rejected_even_with_timing(tmp_path: Path) -> None:
    args = setup(tmp_path)
    save_commit(args)
    args[3].parent.mkdir()
    args[3].write_text('{"iteration": 1}')
    args[1].write_bytes(b"retrained replacement")
    with pytest.raises(ValueError, match="checkpoint differs"):
        arena.completed(*args)


def test_duplicate_commits_are_rejected(tmp_path: Path) -> None:
    args = setup(tmp_path)
    save_commit(args)
    league = json.loads(args[0].read_text())
    league["history"] *= 2
    args[0].write_text(json.dumps(league))
    with pytest.raises(ValueError, match="duplicate committed arena"):
        arena.completed(*args)
    assert not args[3].exists()


def test_old_unhashed_commit_cannot_be_used_for_recovery(tmp_path: Path) -> None:
    args = setup(tmp_path)
    args[0].write_text('{"champion": 1, "history": [{"iteration": 1}]}')
    with pytest.raises(ValueError, match="no checkpoint digest"):
        arena.completed(*args)
    assert not args[3].exists()


def test_missing_committed_checkpoint_fails(tmp_path: Path) -> None:
    args = setup(tmp_path)
    save_commit(args)
    args[1].unlink()
    with pytest.raises(FileNotFoundError):
        arena.completed(*args)
    assert not args[3].exists()


def test_cli_prints_only_completion_flag(tmp_path: Path) -> None:
    args = setup(tmp_path)
    state, checkpoint, iteration, timing = args
    command = [sys.executable, str(SCRIPT), "--state", str(state),
               "--checkpoint", str(checkpoint), "--iteration", str(iteration),
               "--timing", str(timing)]
    incomplete = subprocess.run(command, capture_output=True, text=True, check=True)
    save_commit(args)
    complete = subprocess.run(command, capture_output=True, text=True, check=True)
    assert incomplete.stdout == "0\n" and complete.stdout == "1\n"
    assert not incomplete.stderr and not complete.stderr


def test_atomic_json_replaces_old_state(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("old state")
    arena.write_atomic_json(path, {"champion": 1, "history": []})
    assert json.loads(path.read_text()) == {"champion": 1, "history": []}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
