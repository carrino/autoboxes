"""Protocol, interruption and input-integrity checks for the overnight runner."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "teacher_runner", Path(__file__).with_name("run_teacher_ab.py")
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def bundled(tmp_path: Path, name: str, content: bytes = b"{}", *, link: bool = False) -> Path:
    """Build a tiny correctly hashed archive to exercise member validation."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/escape"
            archive.addfile(member)
        else:
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    blob = stream.getvalue()
    (data / "teacher-payload.tgz.b64").write_text(base64.b64encode(blob).decode())
    (data / "teacher-payload.sha256").write_text(hashlib.sha256(blob).hexdigest())
    return data


def test_archive_hash_and_path_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "SCRIPT", tmp_path / "run_teacher_ab.py")
    data = bundled(tmp_path, "manifest.json")
    runner.unpack_payload(None, tmp_path / "good")
    assert (tmp_path / "good/manifest.json").read_text() == "{}"
    (data / "teacher-payload.sha256").write_text("0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        runner.unpack_payload(None, tmp_path / "wrong")
    bundled(tmp_path, "../escape")
    with pytest.raises(ValueError, match="Unsafe"):
        runner.unpack_payload(None, tmp_path / "escape")
    bundled(tmp_path, "manifest.json", link=True)
    with pytest.raises(ValueError, match="Unsafe"):
        runner.unpack_payload(None, tmp_path / "linked")


def test_stage_preserves_failure_reuses_complete_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = []

    def fake_job(command: list[str], log: Path, cap: int) -> None:
        attempts.append(log.parent)
        target = log.parent / "train"
        target.mkdir()
        (target / "checkpoint.pt").write_bytes(b"checkpoint")
        if len(attempts) == 1:
            raise RuntimeError("interrupted")
        (target / "summary.json").write_text('{"status":"complete"}')

    monkeypatch.setattr(runner, "job", fake_job)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.stage(tmp_path, "fit", lambda p: [str(p)], 10, "train")
    assert not (tmp_path / "fit/completed.json").exists()
    assert (attempts[0] / "train/checkpoint.pt").exists()
    completed = runner.stage(tmp_path, "fit", lambda p: [str(p)], 10, "train")
    assert len(attempts) == 2 and completed.parent != attempts[0]
    assert runner.stage(tmp_path, "fit", lambda p: [str(p)], 10, "train") == completed
    assert len(attempts) == 2
    (completed / "checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Completed stage changed"):
        runner.stage(tmp_path, "fit", lambda p: [str(p)], 10, "train")


def test_fresh_schedule_excludes_whole_symmetry_family() -> None:
    from alpha_go.boxes.qdab_training_collect import opening_family

    first = runner.fresh_schedule(200, 20261121, set())
    excluded = {opening_family(slot["opening"]) for slot in first[:10]}
    second = runner.fresh_schedule(200, 20261121, excluded)
    assert second == runner.fresh_schedule(200, 20261121, excluded)
    families = [opening_family(slot["opening"]) for slot in second[::2]]
    assert len(set(families)) == 100 and not excluded.intersection(families)
    for index in range(0, 200, 2):
        a, b = second[index : index + 2]
        assert a["opening"] == b["opening"]
        assert a["a_first"] and not b["a_first"]
        assert a["pair"] == b["pair"] == index // 2
    with pytest.raises(ValueError, match="positive even"):
        runner.fresh_schedule(3, 0, set())


def test_matched_training_requires_traces_and_counts_teacher_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alpha_go.boxes import policy_target_ab

    fake_data = SimpleNamespace(
        games=[
            dict(
                boards=np.zeros((3, 3, 3)), approximate_teacher_mask=np.array([False, True, False])
            )
        ],
        position_indices=[np.arange(3)],
    )
    monkeypatch.setattr(
        policy_target_ab,
        "load_arm",
        lambda *args: SimpleNamespace(split_stable=lambda *a: (fake_data, None)),
    )
    paths = {}
    summary: dict[str, Any] = dict(
        status="complete",
        reference_sha256="same",
        settings={},
        result=dict(
            status="complete",
            index_trace_sha256="same",
            steps=2048,
            augmentation_schedule_sha256="same",
        ),
    )
    for role in runner.ROLES:
        path = tmp_path / role
        path.mkdir()
        paths[role] = path
        runner.write_json(path / "summary.json", summary)
        np.savez(path / "control-indices.npz", indices=np.array([[0, 1, 2, 1]]))
    report = runner.matched_traces(paths, tmp_path)
    assert report["teacher_exposures"] == 2 and report["unique_teacher_rows_exposed"] == 1
    summary["result"]["augmentation_schedule_sha256"] = "changed"
    runner.write_json(paths[runner.ROLES[1]] / "summary.json", summary)
    with pytest.raises(ValueError, match="augmentation_schedule"):
        runner.matched_traces(paths, tmp_path)


def test_teacher_read_executes_real_checkpoint_and_legal_mask(tmp_path: Path) -> None:
    from alpha_go.boxes.model import BoxesNet
    from alpha_go.boxes.nn_agent import save_boxes_net

    checkpoint = tmp_path / "tiny.pt"
    model = BoxesNet(5, channels=4, n_blocks=1, features="chains")
    save_boxes_net(model, checkpoint)
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    for index, split in enumerate(("training", "validation")):
        runner.write_json(
            teacher / f"{index}.json",
            dict(
                root=dict(prefix=[0, 1, 2, 3], move_count=4),
                teacher_edge=4,
                source_family=f"family-{index}",
                split=split,
            ),
        )
    result = runner.teacher_reads(tmp_path, {"tiny": checkpoint})
    assert result["tiny"]["metrics"]["validation"]["positions"] == 1
    # Fresh readouts are uniform over 56 legal edges, not all 60 lattice edges.
    assert result["tiny"]["metrics"]["training"]["mean_teacher_probability"] == pytest.approx(
        1 / 56
    )


def test_archive_excludes_inputs_weights_and_replay(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/payload.json").write_text("large")
    (tmp_path / "weights.pt").write_text("weights")
    (tmp_path / "indices.npz").write_text("large")
    (tmp_path / "summary.json").write_text("{}")
    runner.package(tmp_path)
    with tarfile.open(tmp_path / "results.tgz") as archive:
        assert archive.getnames() == ["summary.json"]


def test_match_comparison_clusters_both_seats_and_requires_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alpha_go.boxes import qdab_benchmark as match

    slots = runner.fresh_schedule(4, 9, set())
    margins = [[-1, -1, 1, 1], [1, 1, 1, 1]]
    records = {}
    paths = {}
    for role, values in zip(runner.ROLES, margins, strict=True):
        path = tmp_path / role
        path.mkdir()
        runner.write_json(path / "config.json", {})
        runner.write_json(
            path / "summary.json", {"panels": {"paired": {"wins": sum(v > 0 for v in values)}}}
        )
        paths[role] = path
        records[path] = [
            dict(slot=slot, a_margin=value) for slot, value in zip(slots, values, strict=True)
        ]
    monkeypatch.setattr(match, "completed", lambda path, config: records[path])
    report = runner.compare_matches(paths, dict(games_per_arm=4, final_seed=9))
    assert report["wins"] == [2, 4] and report["qdab_target_minus_control"] == 0.5
    assert report["paired_difference_ci95"] == [0, 1]
    records[paths[runner.ROLES[1]]].pop()
    with pytest.raises(ValueError, match="Complete both"):
        runner.compare_matches(paths, dict(games_per_arm=4, final_seed=9))
