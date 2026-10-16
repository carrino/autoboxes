"""Qdab protocol boundaries, bounded cleanup, and legal shared-game histories."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import tarfile
from unittest.mock import MagicMock

import pytest

from alpha_go.boxes import qdab, qdab_collect
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.boxes.symmetry import edge_permutation
from alpha_go.gameplay import play_game
from alpha_go.games import get_game


def masks(edges):
    result = {"H": 0, "V": 0}
    for edge in edges:
        axis, bit = qdab.encode_edge(edge)
        result[axis] |= 1 << bit
    return result


class TestProtocol:
    def test_all_edges_roundtrip(self):
        board = BoxesBoard(5)
        for edge in range(60):
            assert qdab.decode_result(masks([edge]), board) == [edge]
        assert masks(list(range(60))) == {"H": (1 << 30) - 1, "V": (1 << 30) - 1}

    def test_replay_random_histories_and_retained_turns(self):
        rng = random.Random(20261016)
        captures = doubles = 0
        for _ in range(50):
            edges = list(range(60))
            rng.shuffle(edges)
            ref = BoxesBoard(5)
            turns = 0
            for step, edge in enumerate(edges):
                mover = ref.player()
                score = ref.boxes[mover]
                assert ref.play_edge(edge)
                gain = ref.boxes[mover] - score
                captures += gain > 0
                doubles += gain == 2
                turns += ref.player() != mover
                board, request = qdab.replay_prefix(edges[:step + 1])
                assert request["Turn"] == turns
                assert request["Now"] == ref.player()
                assert request["S"] == ref.boxes == board.boxes
                assert request["H"] == masks(edges[:step + 1])["H"]
                assert request["V"] == masks(edges[:step + 1])["V"]
        assert captures > 0 and doubles > 0

    def test_bundle_capture_first_then_turn_pass(self):
        board, _ = qdab.replay_prefix([0, 5, 30])
        # Edge31 captures top-left box; edge59 then passes to the opponent.
        assert qdab.decode_result(masks([59, 31]), board) == [31, 59]
        assert board.boxes == [0, 0]  # Decoder never mutates its input.
        board, _ = qdab.replay_prefix([0, 5, 1, 6, 30, 32])
        assert qdab.decode_result(masks([31]), board) == [31]  # Double capture.

    @pytest.mark.parametrize("result", [
        {"H": -1, "V": 0}, {"H": 1 << 30, "V": 0}, {"H": True, "V": 0},
        {"H": 0, "V": 0}, {"H": 3, "V": 0},
    ])
    def test_reject_bad_masks_and_ambiguous_bundles(self, result):
        with pytest.raises(ValueError):
            qdab.decode_result(result, BoxesBoard(5))

    def test_reject_drawn_and_invalid_prefix(self):
        board, _ = qdab.replay_prefix([0])
        with pytest.raises(ValueError, match="drawn"):
            qdab.decode_result(masks([0]), board)
        for prefix in ([0, 0], [60], [-1], [True]):
            with pytest.raises(ValueError):
                qdab.replay_prefix(prefix)

    def test_canonical_family_matches_symmetries_and_later_branches(self):
        prefix = [0, 43, 16, 22, 51, 31, 12, 7, 14]
        board = BoxesBoard(5)
        group = qdab_collect.family_fields(prefix)["source_replay_group"]
        for transform in range(8):
            permutation = edge_permutation(board.geo, transform)
            transformed = [int(permutation[edge]) for edge in prefix]
            assert qdab_collect.family_fields(transformed)["source_replay_group"] == group
        assert qdab_collect.family_fields(prefix[:8] + [23])["source_replay_group"] == group


class TestInstallation:
    def test_verified_archive_only_extracts_pinned_regular_files(self, tmp_path, monkeypatch):
        packed = tmp_path / "fixture.tar.bz2"
        payloads = {"server": b"test executable", "AnnModels/24.ann": b"test model"}
        with tarfile.open(packed, "w:bz2") as tar:
            for name, payload in {**payloads, "../escape": b"ignored"}.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        monkeypatch.setattr(qdab, "ARTIFACTS", {
            name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()})
        monkeypatch.setattr(qdab, "ARCHIVE_SHA256", qdab.sha256(packed))
        release = qdab.install_release(tmp_path / "installed", packed)
        assert (release / "server").read_bytes() == payloads["server"]
        assert not (tmp_path / "escape").exists()
        (release / "server").write_bytes(b"changed")
        with pytest.raises(ValueError, match="checksum"):
            qdab.install_release(release, packed)

    def test_archive_checksum_is_checked_before_extraction(self, tmp_path):
        packed = tmp_path / "bad.tar.bz2"
        packed.write_bytes(b"not a tar")
        with pytest.raises(ValueError, match="archive checksum"):
            qdab.install_release(tmp_path / "release", packed)


class TestBoundedExecution:
    def test_timeout_terminates_child(self, tmp_path, monkeypatch):
        process = MagicMock()
        process.poll.return_value = None
        sock = MagicMock()
        sock.__enter__.return_value = sock
        sock.makefile.return_value.__enter__.return_value.readline.side_effect = TimeoutError
        monkeypatch.setattr(qdab.socket, "socket", MagicMock())
        monkeypatch.setattr(qdab.socket, "create_connection", lambda *args, **kwargs: sock)
        monkeypatch.setattr(qdab.subprocess, "Popen", lambda *args, **kwargs: process)
        monkeypatch.setattr(qdab.time, "sleep", lambda _: None)
        monkeypatch.setattr(qdab, "owns_port", lambda pid: True)
        monkeypatch.setattr(qdab.tempfile, "gettempdir", lambda: str(tmp_path))
        with pytest.raises(TimeoutError):
            qdab.query(tmp_path, [], 5, 20, tmp_path / "engine.log")
        sock.settimeout.assert_called_once_with(20)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=2)

    def test_listener_ownership_requires_our_process_and_loopback(self, tmp_path):
        fd = tmp_path / "123/fd"
        fd.mkdir(parents=True)
        (fd / "3").symlink_to("socket:[456]")
        net = tmp_path / "net"
        net.mkdir()
        (net / "tcp").write_text("header\n0: 0100007F:3039 00000000:0000 0A 0 0 0 0 0 456\n")
        assert qdab.owns_port(123, tmp_path)
        (net / "tcp").write_text("header\n0: 0100007F:3039 00000000:0000 0A 0 0 0 0 0 789\n")
        assert not qdab.owns_port(123, tmp_path)
        (net / "tcp").write_text("header\n0: 00000000:3039 00000000:0000 0A 0 0 0 0 0 456\n")
        assert not qdab.owns_port(123, tmp_path)

    def test_force_kills_child_that_ignores_termination(self):
        process = MagicMock()
        process.wait.side_effect = [qdab.subprocess.TimeoutExpired("server", 2), 0]
        qdab._stop(process)
        process.kill.assert_called_once()

    def test_port_lock_rejects_second_collector(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qdab.tempfile, "gettempdir", lambda: str(tmp_path))
        with qdab.port_lock(), pytest.raises(BlockingIOError), qdab.port_lock():
            pass

    @pytest.mark.parametrize("seconds,timeout", [(0, 1), (-1, 2), (5, 4), (float("nan"), 2)])
    def test_invalid_budgets_do_not_spawn(self, tmp_path, seconds, timeout):
        with pytest.raises(ValueError):
            qdab.query(tmp_path, [], seconds, timeout, tmp_path / "engine.log")


class TestCollection:
    def test_shared_gameplay_full_history_and_resume_without_targets(self, tmp_path, monkeypatch):
        rng = random.Random(12)
        history = list(range(60))
        rng.shuffle(history)
        calls = []

        def fake_query(release, prefix, seconds, timeout, log):
            calls.append(len(prefix))
            return {"edges": [history[len(prefix)]], "decision_seconds": 0.0}

        monkeypatch.setattr(qdab, "query", fake_query)
        monkeypatch.setattr(qdab_collect, "install_release", lambda release, archive: release)
        agents = [qdab.QdabAgent(tmp_path, 5, 20, tmp_path / "engine.log") for _ in range(2)]
        record = play_game(*agents, board_size=5, board_cols=5, max_moves=60,
                           game=get_game("boxes"), collect_boards=False, komi=0)
        assert len(record.moves) == 60
        assert agents[0].prefix == agents[1].prefix == history
        assert sum(agents[0].reference.boxes) == 25
        args = argparse.Namespace(release=tmp_path, archive=None, output=tmp_path / "pool.jsonl",
                                  num_games=1, max_moves=30, seconds=5, hard_timeout=20,
                                  seed=3, verbose=False, attempts=3)
        qdab_collect.collect(args)
        row = json.loads(args.output.read_text())
        assert row["outcome_target"] is None and row["undrawn"] == 30
        assert len(row["prefix"]) == 30
        before = len(calls)
        qdab_collect.collect(args)
        assert len(calls) == before
        args.seconds = 6
        with pytest.raises(ValueError, match="changed recipe"):
            qdab_collect.collect(args)

    @pytest.mark.parametrize("failures", [1, 3])
    def test_retries_timeouts_and_never_commits_failed_game(self, tmp_path, monkeypatch, failures):
        calls = 0

        def flaky_query(release, prefix, seconds, timeout, log):
            nonlocal calls
            calls += 1
            if calls <= failures:
                raise TimeoutError("bounded test timeout")
            return {"edges": [len(prefix)], "decision_seconds": 0.0}

        monkeypatch.setattr(qdab, "query", flaky_query)
        monkeypatch.setattr(qdab_collect, "install_release", lambda release, archive: release)
        args = argparse.Namespace(release=tmp_path, archive=None, output=tmp_path / "pool.jsonl",
                                  num_games=1, max_moves=2, seconds=5, hard_timeout=20,
                                  seed=3, verbose=False, attempts=3)
        if failures == 3:
            with pytest.raises(TimeoutError):
                qdab_collect.collect(args)
            assert not args.output.exists()
            assert not list(args.output.with_suffix(".games").glob("game*.json"))
        else:
            qdab_collect.collect(args)
            row = json.loads(args.output.read_text())
            assert row["attempts"] == 2 and row["prefix"] == [0, 1]
        failed = list(args.output.with_suffix(".games").joinpath("failures").glob("*.json"))
        assert len(failed) == failures
        for path in failed:
            row = json.loads(path.read_text())
            assert row["outcome_target"] is None
            assert row["error_type"] == "TimeoutError"
            assert row["requested_seconds"] == 5 and row["hard_timeout_seconds"] == 20

    def test_programming_errors_do_not_retry(self, tmp_path, monkeypatch):
        calls = 0

        def bad_query(*args):
            nonlocal calls
            calls += 1
            raise ValueError("invalid board")

        monkeypatch.setattr(qdab, "query", bad_query)
        args = argparse.Namespace(seconds=5, hard_timeout=20, verbose=False,
                                  max_moves=2, seed=3, attempts=3)
        with pytest.raises(ValueError, match="invalid board"):
            qdab_collect.searched_game(tmp_path, args, 0, tmp_path)
        assert calls == 1

    def test_history_rejects_truncated_outcome(self):
        prefix = list(range(30))
        board, _ = qdab.replay_prefix(prefix)
        row = dict(prefix=prefix, scores=board.boxes, to_play=board.to_play(),
                   undrawn=30, outcome_target=1, **qdab_collect.family_fields(prefix))
        with pytest.raises(ValueError, match="truncated outcomes"):
            qdab_collect.validate_history(row)
