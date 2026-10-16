"""Operational contracts for an overnight run, independent of external engine/GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from alpha_go.boxes import curriculum


class TestJobResume:
    def test_completed_job_reuses_and_rejects_changed_artifact(self, tmp_path: Path):
        output = tmp_path / "artifact.txt"
        command = [
            sys.executable,
            "-c",
            'from pathlib import Path; Path(__import__("sys").argv[1]).write_text("ok")',
            str(output),
        ]
        curriculum.run_job(tmp_path / "job", command, [output], timeout=10)
        receipt = (tmp_path / "job/job.json").read_bytes()
        curriculum.run_job(tmp_path / "job", command, [output], timeout=10)
        assert (tmp_path / "job/job.json").read_bytes() == receipt
        output.write_text("changed")
        with pytest.raises(ValueError, match="changed"):
            curriculum.run_job(tmp_path / "job", command, [output], timeout=10)

    def test_changed_recipe_rejected(self, tmp_path: Path):
        job = tmp_path / "job"
        job.mkdir()
        (job / "job.json").write_text(json.dumps(dict(command=["old"], artifacts={})))
        with pytest.raises(ValueError, match="recipe changed"):
            curriculum.run_job(job, ["new"], [], timeout=1)

    def test_failure_never_commits_completion(self, tmp_path: Path):
        import subprocess

        with pytest.raises(subprocess.CalledProcessError):
            curriculum.run_job(
                tmp_path / "job",
                [sys.executable, "-c", "raise SystemExit(3)"],
                [tmp_path / "missing"],
                timeout=10,
            )
        assert not (tmp_path / "job/job.json").exists()

    def test_timeout_preserves_partial_file_and_leaves_no_receipt(self, tmp_path: Path):
        partial = tmp_path / "partial"
        command = [
            sys.executable,
            "-c",
            "import time,sys; from pathlib import Path; "
            'Path(sys.argv[1]).write_text("partial"); time.sleep(30)',
            str(partial),
        ]
        with pytest.raises(TimeoutError):
            curriculum.run_job(tmp_path / "job", command, [partial], timeout=0.5)
        assert partial.read_text() == "partial"
        assert not (tmp_path / "job/job.json").exists()


class TestBudget:
    def test_budget_finishes_current_cycle_then_runs_final(self, tmp_path: Path, monkeypatch):
        source = tmp_path / "source.npz"
        source.write_bytes(b"source")
        ref = tmp_path / "reference.pt"
        ref.write_bytes(b"reference")
        curriculum.write_json(
            tmp_path / "source-games.json", {str(source): curriculum.digest(source)}
        )
        config = dict(
            code_sha256={},
            reference=str(ref),
            reference_sha256=curriculum.digest(ref),
            sources_file_sha256=curriculum.digest(tmp_path / "source-games.json"),
            hours=0.001,
            final_games=200,
            monitor_games=1,
            seed=123,
            cycles=128,
            teacher_games=1,
            retained_games=1,
            rehearsal_dirs=[],
            epochs=1,
            cpu=True,
        )
        ticks = iter([0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        monkeypatch.setattr(curriculum.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(curriculum, "collect", lambda output, *args: output / "prefixes.jsonl")
        def label(output, *args):
            prepared = output / "prepared"
            prepared.mkdir(parents=True)
            curriculum.write_json(output / "job.json", dict(command=["label"], artifacts={}))
            return prepared

        monkeypatch.setattr(curriculum, "label", label)
        calls = []

        def train(_job, command, _artifacts, timeout):
            output = Path(command[command.index("--output") + 1])
            curriculum.write_json(output / "summary.json", {"accepted": False})
            checkpoint = output / "candidate.pt"
            checkpoint.write_bytes(b"candidate")
            curriculum.write_json(_job / "job.json", dict(
                command=command, artifacts={str(checkpoint): curriculum.digest(checkpoint)}))
            calls.append("train")

        monkeypatch.setattr(curriculum, "run_job", train)
        def final_match(output, *args):
            curriculum.write_json(output / "final-vs-iter0/job.json",
                                  dict(command=[], artifacts={}))
            calls.append("final")

        monkeypatch.setattr(curriculum, "final_match", final_match)
        monkeypatch.setattr(curriculum, "package_results", lambda *args: None)
        curriculum.run(tmp_path, config)
        assert calls == ["train", "final"]
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["completed_cycles"] == 1 and state["status"] == "complete"
        assert state["accepted_checkpoint"] == str(ref)
        assert state["last_candidate"].endswith("cycle-001/train/candidate.pt")

    def test_archive_excludes_weights_data_and_itself(self, tmp_path: Path):
        import tarfile

        for name in ["reference.pt", "data.npz", "results.tgz", "config.json", "run.log"]:
            (tmp_path / name).write_text("{}")
        curriculum.package_results(tmp_path)
        with tarfile.open(tmp_path / "results.tgz") as archive:
            assert set(archive.getnames()) == {"config.json", "run.log"}


class TestArtifactIntegrity:
    def test_completed_checkpoint_change_is_rejected(self, tmp_path: Path):
        training = tmp_path / "training"
        training.mkdir()
        checkpoint = training / "candidate.pt"
        checkpoint.write_bytes(b"original weights")
        receipt = tmp_path / "job.json"
        curriculum.write_json(receipt, dict(
            command=["train"], directories=[str(training)],
            artifacts={str(checkpoint): curriculum.digest(checkpoint)}))
        assert curriculum.check_receipt(receipt)["command"] == ["train"]
        checkpoint.write_bytes(b"different weights")
        with pytest.raises(ValueError, match="changed"):
            curriculum.check_receipt(receipt)

    def test_added_prepared_npz_is_rejected(self, tmp_path: Path):
        prepared = tmp_path / "prepared"
        prepared.mkdir()
        original = prepared / "original.npz"
        original.write_bytes(b"original labels")
        receipt = tmp_path / "job.json"
        curriculum.write_json(receipt, dict(
            command=["label"], directories=[str(prepared)],
            artifacts={str(original): curriculum.digest(original)}))
        curriculum.check_receipt(receipt)
        (prepared / "additional.npz").write_bytes(b"unrecorded labels")
        with pytest.raises(ValueError, match="added/missing"):
            curriculum.check_receipt(receipt)

    @pytest.mark.parametrize("changed", ["checkpoint", "receipt"])
    def test_resume_rejects_mutated_completed_stage_before_new_work(
        self, tmp_path: Path, monkeypatch, changed: str,
    ):
        import hashlib

        source = tmp_path / "source.npz"
        source.write_bytes(b"source")
        reference = tmp_path / "reference.pt"
        reference.write_bytes(b"reference")
        curriculum.write_json(tmp_path / "source-games.json",
                              {str(source): curriculum.digest(source)})
        config = dict(
            code_sha256={}, reference=str(reference),
            reference_sha256=curriculum.digest(reference),
            sources_file_sha256=curriculum.digest(tmp_path / "source-games.json"),
        )
        checkpoint = tmp_path / "candidate.pt"
        checkpoint.write_bytes(b"candidate")
        receipt = tmp_path / "training-job.json"
        curriculum.write_json(receipt, dict(
            command=["train"], directories=[],
            artifacts={str(checkpoint): curriculum.digest(checkpoint)}))
        curriculum.write_json(tmp_path / "state.json", dict(
            completed_cycles=1, active_seconds=3600, accepted_checkpoint=str(checkpoint),
            last_candidate=str(checkpoint), maximum_undrawn=34, prepared=[],
            receipts={str(receipt): curriculum.digest(receipt)},
            config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        ))

        def forbidden(*args, **kwargs):
            raise AssertionError("must reject changed prior artifacts before collecting")

        monkeypatch.setattr(curriculum, "collect", forbidden)
        if changed == "checkpoint":
            checkpoint.write_bytes(b"modified candidate")
        else:
            document = json.loads(receipt.read_text())
            document["command"] = ["different recipe"]
            curriculum.write_json(receipt, document)
        with pytest.raises(ValueError, match="changed"):
            curriculum.run(tmp_path, config)


class TestFinalEvidence:
    def test_complete_games_replay_and_corruption_rejected(self, tmp_path: Path):
        from alpha_go.boxes.arena import play_match

        result = play_match("boxes-random", "boxes-random", 5, 5, 2, 123, 1,
                            opening_moves=4)
        rows = [dict(index=index, a_first=index % 2 == 0, a_margin=margin,
                     opening=4, moves=moves)
                for index, (margin, moves) in enumerate(zip(result.margins, result.moves))]
        path = tmp_path / "games.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary = result.summary()
        curriculum.verify_match(path, summary, 2)
        assert summary["independent_replay_verified"]
        low, high = summary["paired_game_score_ci95"]
        assert 0 <= low <= high <= 1
        rows[1]["a_margin"] += 2
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(AssertionError):
            curriculum.verify_match(path, result.summary(), 2)

    def test_rehearsal_population_rejects_additions(self, tmp_path: Path):
        existing = tmp_path / "searched.npz"
        existing.write_bytes(b"searched opening")
        config = dict(rehearsal_dirs=[str(tmp_path)],
                      extra_inputs_sha256={str(existing): curriculum.digest(existing)})
        curriculum.check_rehearsal(config)
        (tmp_path / "new.npz").write_bytes(b"new unrecorded opening")
        with pytest.raises(ValueError, match="frozen population"):
            curriculum.check_rehearsal(config)
