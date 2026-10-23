"""Fresh family exclusions, shared-game targets and resumable teacher corrections."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from alpha_go.agents.base import Agent
from alpha_go.boxes import qdab
from alpha_go.boxes import qdab_training_collect as collect
from alpha_go.boxes.qdab_benchmark import OpeningQdabAgent, schedule
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.boxes.symmetry import edge_permutation


class SearchedLastLegal(Agent):
    def select_move(self, board, seed):
        edge = board.get_legal_moves_flat()[-1]
        tree = SimpleNamespace(get_child_visit_counts=lambda: {edge: 400},
                               get_child_q_values=lambda: {edge: 0.5},
                               get_root_policy_priors=lambda: {edge: 1.0},
                               get_root_q_value=lambda: 0.5)
        self.last_search_result = SimpleNamespace(tree=tree,
                                                 training_visit_counts=lambda: {edge: 400})
        return tuple(board.row_col(edge))


@pytest.fixture
def fake_qdab(monkeypatch):
    calls = []

    def query(release, prefix, seconds, hard_timeout, log):
        calls.append(list(prefix))
        board, state = qdab.replay_prefix(prefix)
        edge = board.get_legal_moves_flat()[0]
        axis, bit = qdab.encode_edge(edge)
        result = dict(H=0, V=0)
        result[axis] = 1 << bit
        return dict(edges=[edge], decision_seconds=0.0,
                    rpc_request=dict(params=[dict(Board=state)]),
                    rpc_response=dict(result=result))

    monkeypatch.setattr(qdab, "query", query)
    monkeypatch.setattr(collect, "query", query)
    return calls


def game(tmp_path, first=True):
    slot = collect.fresh_schedule(2, 71)[0 if first else 1]
    opponent = OpeningQdabAgent(tmp_path, 5, 30, tmp_path / "qdab.log",
                               opening=slot["opening"])
    return collect.play_training_slot(SearchedLastLegal(), opponent, slot)


def test_families_exclude_benchmarks_and_keep_symmetries_together():
    slots = collect.fresh_schedule(60, 20261123)
    assert slots == collect.fresh_schedule(60, 20261123)
    families = {slot["source_family"] for slot in slots}
    assert len(families) == 60
    excluded = {collect.opening_family(slot["opening"])
                for count, seed in ((100, 20261112), (8, 20261116))
                for slot in schedule(count, 0, seed)}
    assert not families & excluded
    board = BoxesBoard(5)
    opening = slots[0]["opening"]
    transformed = [list(board.row_col(int(edge_permutation(board.geo, 1)[
        board.edge_index(*move)]))) for move in opening[::-1]]
    assert collect.opening_family(opening) == collect.opening_family(transformed)
    assert {slot["split"] for slot in slots} == {"training", "validation"}


@pytest.mark.parametrize("first", [True, False])
def test_shared_game_stores_only_learner_searches_and_replays(tmp_path, fake_qdab, first):
    row = game(tmp_path, first)
    collect.verify_training_game(row, row["slot"])
    assert row["termination"] == "all_edges" and len(row["moves"]) == 60
    assert row["learner_decisions"]
    for root in row["learner_decisions"]:
        board, _ = qdab.replay_prefix(root["prefix"])
        assert board.player() == (0 if first else 1)
        assert root["policy_target"][root["learner_edge"]] == 1
    selected = collect.teacher_roots(row)
    assert len(selected) <= 4
    assert all(38 <= root["undrawn"] <= 50 for root in selected)
    assert "exact_margin_p1" not in row


def test_teacher_queries_resume_without_requery_or_replacing_control(tmp_path, fake_qdab):
    row = game(tmp_path)
    config = dict(release=str(tmp_path), seconds=5, hard_timeout=30,
                  checkpoint_sha256="test")
    collect.write_json(tmp_path / "config.json", config)
    before = copy.deepcopy(row)
    collect.collect_teachers(tmp_path, config, row)
    count = len(fake_qdab)
    collect.collect_teachers(tmp_path, config, row)
    assert len(fake_qdab) == count and row == before
    teachers = [json.loads(p.read_text()) for p in (tmp_path / "teacher").glob("*.json")]
    assert len(teachers) == len(collect.teacher_roots(row)) > 0
    assert all(t["teacher_kind"] == "approximate_qdab_action" and not t["exact_policy"]
               for t in teachers)
    assert all(t["source_family"] == row["source_family"] and t["split"] == row["split"]
               for t in teachers)


def test_corrupted_policy_or_root_is_rejected(tmp_path, fake_qdab):
    row = game(tmp_path)
    bad = copy.deepcopy(row)
    bad["learner_decisions"][0]["scores"] = [99, 0]
    with pytest.raises(ValueError, match="independent replay"):
        collect.verify_training_game(bad, bad["slot"])
    bad = copy.deepcopy(row)
    bad["learner_decisions"][0]["policy_target"] = [1.0] * 60
    with pytest.raises(ValueError, match="normalized"):
        collect.verify_training_game(bad, bad["slot"])
    bad = copy.deepcopy(row)
    root = bad["learner_decisions"][0]
    root["visits"][root["learner_edge"]] = -1
    with pytest.raises(ValueError, match="integer visit"):
        collect.verify_training_game(bad, bad["slot"])
    bad = copy.deepcopy(row)
    bad["learner_decisions"][0]["raw_visits"] = [0] * 60
    with pytest.raises(ValueError, match="Raw exploration"):
        collect.verify_training_game(bad, bad["slot"])


def test_corrupted_teacher_and_query_failure_preserve_game(tmp_path, fake_qdab, monkeypatch):
    row = game(tmp_path)
    config = dict(release=str(tmp_path), seconds=5, hard_timeout=30,
                  checkpoint_sha256="test")
    collect.write_json(tmp_path / "config.json", config)
    collect.write_json(tmp_path / "games/0000.json", row)
    original = collect.query

    def failure(*args):
        raise TimeoutError("teacher deadline")

    monkeypatch.setattr(collect, "query", failure)
    with pytest.raises(TimeoutError, match="deadline"):
        collect.collect_teachers(tmp_path, config, row)
    assert json.loads((tmp_path / "games/0000.json").read_text()) == row
    assert len(list((tmp_path / "teacher/failures").glob("*.json"))) == 1
    monkeypatch.setattr(collect, "query", original)
    collect.collect_teachers(tmp_path, config, row)
    path = next((tmp_path / "teacher").glob("*.json"))
    bad = json.loads(path.read_text())
    bad["teacher_edge"] = -1
    collect.write_json(path, bad)
    with pytest.raises(ValueError, match="legal capture"):
        collect.collect_teachers(tmp_path, config, row)


def test_missing_search_is_not_a_uniform_teacher_target():
    assert collect.teacher_roots(dict(learner_decisions=[])) == []
