"""Full-game replay, opening synchronization, interrupted matches and separate ratings."""
from __future__ import annotations

import copy
import json
import signal
import sys

import pytest

from alpha_go.agents.base import Agent
from alpha_go.boxes import qdab
from alpha_go.boxes import qdab_benchmark as bench
from alpha_go.boxes.rules import BoxesBoard


class LastLegal(Agent):
    def select_move(self, board, seed):
        return tuple(board.row_col(board.get_legal_moves_flat()[-1]))


@pytest.fixture
def fake_qdab(monkeypatch):
    calls = []

    def query(release, prefix, seconds, hard_timeout, log):
        calls.append(list(prefix))
        board, _ = qdab.replay_prefix(prefix)
        return dict(edges=[board.get_legal_moves_flat()[0]], decision_seconds=0.0)

    monkeypatch.setattr(qdab, "query", query)
    return calls


def configuration(tmp_path, games=2, empty=2):
    config = dict(schedule=bench.schedule(games, empty, 20261112), seed=20261112,
                  checkpoint_sha256="fixture", release=str(tmp_path), seconds=5,
                  hard_timeout=30, max_attempts_per_game=3)
    bench.write_json(tmp_path / "config.json", config)
    return config


@pytest.mark.parametrize("a_first", [True, False])
@pytest.mark.parametrize("prefix", [[], [0, 5, 30, 31]])
def test_full_game_syncs_capture_opening_and_side(tmp_path, fake_qdab, prefix, a_first):
    board = BoxesBoard(5)
    slot = dict(index=0, panel="test", pair=0, seed=7, a_first=a_first,
                opening=[list(board.row_col(e)) for e in prefix])
    opponent = bench.OpeningQdabAgent(tmp_path, 5, 30, tmp_path / "qdab.log",
                                     opening=slot["opening"])
    result = bench.play_slot(LastLegal(), opponent, slot)
    bench.verify_game(result, slot)
    assert len(opponent.prefix) == len(result["moves"]) == 60
    assert fake_qdab[0][:len(prefix)] == prefix
    assert opponent.reference.is_game_over()
    assert result["a_moves"] + result["b_moves"] == 60 - len(prefix)


def test_frozen_schedule_pairs_and_empty_separate():
    slots = bench.schedule(100, 20, 20261112)
    assert slots == bench.schedule(100, 20, 20261112)
    for a, b in zip(slots[::2], slots[1::2]):
        assert a["opening"] == b["opening"]
        assert a["pair"] == b["pair"] and a["a_first"] and not b["a_first"]
        assert len(a["opening"]) == (4 if a["panel"] == "paired" else 0)
    assert slots != bench.schedule(100, 20, 20261113)


def test_resume_keeps_unpaired_game_and_does_not_skip_failure(tmp_path, fake_qdab, monkeypatch):
    config = configuration(tmp_path)
    original = bench.play_slot
    attempted = []

    def fail_second(agent, opponent, slot):
        attempted.append(slot["index"])
        if slot["index"] == 1:
            raise TimeoutError("engine deadline")
        return original(agent, opponent, slot)

    monkeypatch.setattr(bench, "play_slot", fail_second)
    with pytest.raises(TimeoutError):
        bench.run_games(tmp_path, config, LastLegal())
    first = (tmp_path / "games/0000.json").read_bytes()
    report = bench.report(tmp_path, config)
    assert report["status"] == "incomplete"
    assert report["panels"]["paired"]["completed"] == 1
    assert "relative_elo" not in report["panels"]["paired"]
    assert report["failed_or_interrupted_attempts"] == 1
    assert attempted == [0, 1]

    monkeypatch.setattr(bench, "play_slot", original)
    bench.run_games(tmp_path, config, LastLegal())
    assert (tmp_path / "games/0000.json").read_bytes() == first
    assert len(bench.completed(tmp_path, config)) == 4
    report = bench.report(tmp_path, config)
    assert report["status"] == "complete"
    assert report["panels"]["paired"]["games"] == 2
    assert "relative_elo" in report["panels"]["paired"]
    assert "relative_elo" not in report["panels"]["empty"]
    before = len(fake_qdab)
    bench.run_games(tmp_path, config, LastLegal())
    assert len(fake_qdab) == before


def test_bad_game_cannot_be_accepted_or_counted_as_forfeit(tmp_path, fake_qdab):
    config = configuration(tmp_path, empty=0)
    bench.run_games(tmp_path, config, LastLegal())
    path = tmp_path / "games/0000.json"
    good = json.loads(path.read_text())
    variants = [dict(a_margin=-good["a_margin"]), dict(moves=good["moves"][:-1]),
                dict(a_moves=good["a_moves"] + 1), dict(termination="solved")]
    for change in variants:
        bad = copy.deepcopy(good)
        bad.update(change)
        bench.write_json(path, bad)
        with pytest.raises(ValueError):
            bench.completed(tmp_path, config)
    bench.write_json(path, good)
    config["schedule"][0]["seed"] += 1
    with pytest.raises(ValueError, match="slot"):
        bench.completed(tmp_path, config)


def test_attempt_cap_keeps_failed_slots_pending(tmp_path, monkeypatch):
    config = configuration(tmp_path, empty=0)

    def failure(*args):
        raise RuntimeError("illegal move")

    monkeypatch.setattr(bench, "play_slot", failure)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="illegal move"):
            bench.run_games(tmp_path, config, LastLegal())
    with pytest.raises(RuntimeError, match="Repeated"):
        bench.run_games(tmp_path, config, LastLegal())
    assert bench.completed(tmp_path, config) == []
    assert bench.report(tmp_path, config)["failed_or_interrupted_attempts"] == 3


def test_resume_rejects_changed_configuration(tmp_path, fake_qdab):
    config = configuration(tmp_path, empty=0)
    bench.run_games(tmp_path, config, LastLegal())
    config["seconds"] = 6
    bench.write_json(tmp_path / "config.json", config)
    with pytest.raises(ValueError, match="protocol"):
        bench.completed(tmp_path, config)


def test_sigterm_packages_incomplete_result(tmp_path, monkeypatch):
    configuration(tmp_path, empty=0)
    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.update({sig: handler}))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--resume", str(tmp_path)])

    def terminate(output, config):
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(bench, "run", terminate)
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        bench.main()
    assert (tmp_path / "results.tgz").is_file()
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "incomplete"
