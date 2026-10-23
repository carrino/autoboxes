"""Queue lifecycle checks without playing Qdab or performing model inference."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from queue_structural import collection_lock, collection_ready, digest, stage, write_json


def source(tmp_path: Path) -> tuple[Path, Path, str]:
    collection = tmp_path / "collection"
    slots = [dict(index=0), dict(index=1)]
    write_json(collection / "config.json", dict(schedule=slots))
    config_hash = digest(collection / "config.json")
    for slot in slots:
        write_json(collection / "games" / f"{slot['index']:04d}.json",
                   dict(slot=slot, config_sha256=config_hash))
    supervisor = tmp_path / "job.json"
    write_json(supervisor, dict(status="running", supervisor_pid=os.getpid()))
    return collection, supervisor, config_hash


def test_only_successful_complete_collection_can_start(tmp_path: Path) -> None:
    collection, supervisor, config_hash = source(tmp_path)
    assert not collection_ready(collection, supervisor, config_hash)
    write_json(supervisor, dict(status="failed", returncode=1))
    with pytest.raises(RuntimeError, match="failed"):
        collection_ready(collection, supervisor, config_hash)
    write_json(supervisor, dict(status="complete", returncode=0))
    assert collection_ready(collection, supervisor, config_hash)
    (collection / "games/0001.json").unlink()
    with pytest.raises(RuntimeError, match="all frozen games"):
        collection_ready(collection, supervisor, config_hash)


def test_changed_source_or_game_is_rejected(tmp_path: Path) -> None:
    collection, supervisor, config_hash = source(tmp_path)
    write_json(supervisor, dict(status="complete", returncode=0))
    write_json(collection / "games/0000.json", dict(slot=dict(index=1), config_sha256=config_hash))
    with pytest.raises(ValueError, match="frozen collection"):
        collection_ready(collection, supervisor, config_hash)
    (collection / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration changed"):
        collection_ready(collection, supervisor, config_hash)


def test_completed_stage_is_reused_but_tampered_output_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "data/result.json"
    command = [sys.executable, "-c",
               "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
               "p.parent.mkdir(); p.write_text('{}')", str(artifact)]
    stage(command, tmp_path / "stage", 5, [artifact.parent])
    before = (tmp_path / "stage/stage.json").read_bytes()
    stage(command, tmp_path / "stage", 5, [artifact.parent])
    assert (tmp_path / "stage/stage.json").read_bytes() == before
    artifact.write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="Frozen input"):
        stage(command, tmp_path / "stage", 5, [artifact.parent])


def test_failed_stage_never_becomes_complete(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    with pytest.raises(subprocess.CalledProcessError):
        stage(command, tmp_path / "stage", 5, [tmp_path / "missing.json"])
    receipt = json.loads((tmp_path / "stage/stage.json").read_text())
    assert receipt["status"] == "failed" and receipt["returncode"] == 7


def test_timeout_terminates_stage_and_preserves_interruption(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import time; time.sleep(10)"]
    with pytest.raises(subprocess.TimeoutExpired):
        stage(command, tmp_path / "stage", 1, [tmp_path / "unused.json"])
    receipt = json.loads((tmp_path / "stage/stage.json").read_text())
    assert receipt["status"] == "interrupted" and receipt["returncode"] != 0


def test_source_lock_allows_child_reader_and_excludes_collector(tmp_path: Path) -> None:
    script = ("import fcntl,sys; f=open(sys.argv[1], 'a'); "
              "fcntl.flock(f, int(sys.argv[2]) | fcntl.LOCK_NB)")
    import fcntl
    with collection_lock(tmp_path):
        reader = subprocess.run([sys.executable, "-c", script, str(tmp_path / "run.lock"),
                                 str(fcntl.LOCK_SH)], capture_output=True)
        writer = subprocess.run([sys.executable, "-c", script, str(tmp_path / "run.lock"),
                                 str(fcntl.LOCK_EX)], capture_output=True)
    assert reader.returncode == 0, reader.stderr
    assert writer.returncode != 0 and b"BlockingIOError" in writer.stderr
