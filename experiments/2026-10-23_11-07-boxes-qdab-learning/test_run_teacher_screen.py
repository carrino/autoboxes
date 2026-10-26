"""Input integrity and interrupted-run checks for the short teaching screen."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "teacher_screen_runner", Path(__file__).with_name("run_teacher_screen.py")
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def bundled(tmp_path: Path, names: list[str], *, link: bool = False) -> Path:
    """Write a tiny correctly hashed archive containing nominated member paths."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in names:
            member = tarfile.TarInfo(name)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = "/tmp/escape"
                archive.addfile(member)
            else:
                member.size = 2
                archive.addfile(member, io.BytesIO(b"{}"))
    blob = stream.getvalue()
    (data / "teacher-screen-payload.tgz.b64").write_text(base64.b64encode(blob).decode())
    (data / "teacher-screen-payload.sha256").write_text(hashlib.sha256(blob).hexdigest())
    return data


def test_bundled_payload_checks_hash_paths_duplicates_and_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "SCRIPT", tmp_path / "run_teacher_screen.py")
    data = bundled(tmp_path, ["manifest.json"])
    runner.unpack_payload(None, tmp_path / "good")
    assert (tmp_path / "good/manifest.json").read_text() == "{}"
    (data / "teacher-screen-payload.sha256").write_text("0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        runner.unpack_payload(None, tmp_path / "bad-hash")
    for index, names in enumerate((["../escape"], ["/escape"], ["manifest.json"] * 2)):
        bundled(tmp_path, names)
        with pytest.raises(ValueError, match="Unsafe"):
            runner.unpack_payload(None, tmp_path / f"unsafe-{index}")
    bundled(tmp_path, ["manifest.json"], link=True)
    with pytest.raises(ValueError, match="Unsafe"):
        runner.unpack_payload(None, tmp_path / "linked")


def test_run_preserves_partial_fit_and_reuses_verified_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner.write_json(tmp_path / "config.json", {})
    calls = []

    def fit_job(command: list[str], log: Path, cap: int) -> None:
        calls.append(log.parent)
        fit = tmp_path / "fit"
        fit.mkdir(exist_ok=True)
        endpoint = fit / "saved-first-arm.pt"
        if len(calls) == 1:
            endpoint.write_bytes(b"completed first arm")
            log.write_text("interruption")
            raise RuntimeError("interrupted second arm")
        assert endpoint.read_bytes() == b"completed first arm"
        runner.write_json(fit / "summary.json", dict(status="complete", steps=2))

    monkeypatch.setattr(runner, "fit_job", fit_job)
    config = dict(replay="replay", settings=dict(cpu=True, steps=2), fit_limit_seconds=20)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run(tmp_path, config)
    assert not (tmp_path / "fit-completed.json").exists()
    assert (calls[0] / "run.log").read_text() == "interruption"
    assert runner.run(tmp_path, config)["status"] == "complete"
    assert runner.run(tmp_path, config)["status"] == "complete"
    assert len(calls) == 2 and calls[0] != calls[1]
    (tmp_path / "fit/saved-first-arm.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Completed fit changed"):
        runner.run(tmp_path, config)


def test_create_freezes_inputs_and_verify_rejects_replay_or_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alpha_go.boxes import policy_target_ab

    for name in (
        "run_teacher_screen.py",
        "run_teacher_screen.sh",
        "fit_teacher_screen.py",
        "prepare_teacher_screen.py",
        "prepare_teacher.py",
    ):
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(runner, "SCRIPT", tmp_path / "run_teacher_screen.py")
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner, "runtime", lambda: {})
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "commit\n")
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "source.npz").write_bytes(b"frozen replay")
    monkeypatch.setattr(policy_target_ab, "bundle_files", runner.tree_hashes)
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"frozen reference")
    monkeypatch.setattr(runner, "CHECKPOINT_SHA", runner.digest(reference))
    payload = tmp_path / "payload"
    payload.mkdir()
    runner.write_json(payload / "manifest.json", dict(checkpoint_sha256=runner.CHECKPOINT_SHA))
    args = argparse.Namespace(
        checkpoint=reference,
        replay=replay,
        payload=payload,
        output_root=tmp_path / "runs",
        steps=2,
        batch_size=16,
        teacher_per_batch=4,
        lr=1e-4,
        seed=7,
        cpu=True,
        min_undrawn=24,
    )
    output, config = runner.create(args)
    runner.verify(output, config)
    assert (output / "resume.sh").is_file()
    assert (output / "inputs/reference.pt").read_bytes() == reference.read_bytes()
    (replay / "source.npz").write_bytes(b"mutated replay")
    with pytest.raises(ValueError, match="Source replay changed"):
        runner.verify(output, config)
    (replay / "source.npz").write_bytes(b"frozen replay")
    (tmp_path / "fit_teacher_screen.py").write_text("changed code")
    with pytest.raises(ValueError, match="Screen source changed"):
        runner.verify(output, config)


def test_fit_command_records_recipe_and_has_no_arena_arguments(tmp_path: Path) -> None:
    settings = dict(
        steps=2, batch_size=16, teacher_per_batch=4, lr=1e-4, seed=9, cpu=True, min_undrawn=24
    )
    command = runner.fit_command(tmp_path, dict(replay="existing-replay", settings=settings))
    assert "--cpu" in command and command[command.index("--teacher-per-batch") + 1] == "4"
    assert command[command.index("--steps") + 1] == "2"
    assert "--games" not in command and "--release" not in command


def test_package_keeps_readouts_excludes_weights_inputs_and_sample_arrays(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/source.json").write_text("large input")
    (tmp_path / "fit").mkdir()
    for name in ("model.pt", "indices.npz", "indices.npy"):
        (tmp_path / "fit" / name).write_bytes(b"large")
    (tmp_path / "fit/teacher-reads.jsonl").write_text("{}\n")
    runner.write_json(tmp_path / "summary.json", dict(status="complete"))
    runner.package(tmp_path)
    with tarfile.open(tmp_path / "results.tgz") as archive:
        assert archive.getnames() == ["fit/teacher-reads.jsonl", "summary.json"]


def test_main_does_not_acquire_qdab_port_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alpha_go.boxes import qdab

    runner.write_json(
        tmp_path / "config.json", dict(settings=dict(steps=2, teacher_per_batch=4, batch_size=16))
    )
    monkeypatch.setattr(runner.sys, "argv", ["screen", "--resume", str(tmp_path)])
    monkeypatch.setattr(runner, "REPO", Path.cwd())
    monkeypatch.setattr(runner, "verify", lambda *args: None)
    seen: list[Any] = []
    monkeypatch.setattr(runner, "run", lambda *args: seen.append("fit"))
    monkeypatch.setattr(runner, "package", lambda *args: None)
    monkeypatch.setattr(qdab.tempfile, "gettempdir", lambda: str(tmp_path))
    original_flock = runner.fcntl.flock

    def only_run_lock(file: Any, operation: int) -> None:
        assert Path(file.name).name == "run.lock", "screen must not acquire a Qdab port lock"
        original_flock(file, operation)

    with qdab.port_lock():
        monkeypatch.setattr(runner.fcntl, "flock", only_run_lock)
        runner.main()
    assert seen == ["fit"]
