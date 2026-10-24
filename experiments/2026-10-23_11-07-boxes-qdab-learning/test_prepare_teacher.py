"""Approximate-teacher provenance, masking, target semantics and portable receipts."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.qdab import encode_edge, replay_prefix
from alpha_go.boxes.qdab_training_collect import fresh_schedule, teacher_roots
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.boxes.symmetry import edge_permutation
from alpha_go.boxes.validation import replay_group

SPEC = importlib.util.spec_from_file_location(
    "prepare_teacher", Path(__file__).with_name("prepare_teacher.py")
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def saved_game(slot):
    board = BoxesBoard(5)
    history = []
    for move in slot["opening"]:
        edge = board.edge_index(*move)
        assert board.play_edge(edge)
        history.append(edge)
    decisions = []
    counts = [0, 0]
    while not board.is_game_over():
        edge = board.get_legal_moves_flat()[-1]
        counts[board.player()] += 1
        if board.player() == (0 if slot["a_first"] else 1):
            visits = np.zeros(60, int)
            visits[board.get_legal_moves_flat()] = 1
            decisions.append(
                dict(
                    move_count=len(history),
                    prefix=history.copy(),
                    rows=5,
                    cols=5,
                    scores=list(board.boxes),
                    to_play=board.to_play(),
                    undrawn=60 - len(history),
                    learner_edge=edge,
                    visits=visits.tolist(),
                    policy_target=(visits / visits.sum()).tolist(),
                    raw_visits=None,
                )
            )
        assert board.play_edge(edge)
        history.append(edge)
    margin = int(board.score())
    counts = counts if slot["a_first"] else counts[::-1]
    return dict(
        slot=slot,
        moves=[list(map(int, board.row_col(e))) for e in history],
        a_margin=margin * (1 if slot["a_first"] else -1),
        result=f"{'B' if margin > 0 else 'W'}+{abs(margin):.1f}",
        termination="all_edges",
        a_moves=counts[0],
        b_moves=counts[1],
        a_seconds=1.0,
        b_seconds=2.0,
        qdab_decisions=[],
        learner_decisions=decisions,
        source_family=slot["source_family"],
        split=slot["split"],
    )


def collection(tmp_path):
    folder = tmp_path / "collection"
    (folder / "games").mkdir(parents=True)
    (folder / "teacher").mkdir()
    slots = fresh_schedule(2, 99)
    slots[0]["split"], slots[1]["split"] = "training", "validation"
    config = dict(schedule=slots, checkpoint_sha256="a" * 64, code_sha256={})
    MODULE.write_json(folder / "config.json", config)
    teachers = {}
    games = []
    for slot in slots:
        game = saved_game(slot)
        game["config_sha256"] = MODULE.sha256(folder / "config.json")
        games.append(game)
        MODULE.write_json(folder / "games" / f"{slot['index']:04d}.json", game)
        for root in teacher_roots(game):
            board, rpc = replay_prefix(root["prefix"])
            edge = board.get_legal_moves_flat()[0]
            axis, bit = encode_edge(edge)
            result = dict(H=0, V=0)
            result[axis] = 1 << bit
            saved = dict(
                schema=1,
                root=root,
                source_family=slot["source_family"],
                split=slot["split"],
                source_game=slot["index"],
                exact_policy=False,
                teacher_kind="approximate_qdab_action",
                teacher_edge=edge,
                checkpoint_sha256=config["checkpoint_sha256"],
                config_sha256=MODULE.sha256(folder / "config.json"),
                teacher_reply=dict(
                    edges=[edge],
                    rpc_request=dict(params=[dict(Board=rpc)]),
                    rpc_response=dict(result=result),
                ),
            )
            name = f"{slot['index']:04d}-{root['move_count']:02d}.json"
            MODULE.write_json(folder / "teacher" / name, saved)
            teachers[name] = saved
    MODULE.write_json(
        folder / "summary.json",
        dict(
            games_completed=2,
            games_planned=2,
            training_performed=False,
            weights_changed=False,
            teacher_queries=len(teachers),
            split_games=dict(training=1, validation=1),
        ),
    )
    return folder, games, teachers


def benchmark(tmp_path):
    folder = tmp_path / "benchmark"
    (folder / "games").mkdir(parents=True)
    slot = fresh_schedule(1, 55)[0]
    game = saved_game(slot)
    MODULE.write_json(folder / "config.json", dict(schedule=[slot]))
    game["config_sha256"] = MODULE.sha256(folder / "config.json")
    MODULE.write_json(folder / "games/0000.json", game)
    return folder


def test_real_loader_preserves_value_targets_and_only_learner_rows(tmp_path):
    folder, games, teachers = collection(tmp_path)
    MODULE.verified_collection(folder)
    old = MODULE.game_arrays(games[0], teachers, "b" * 64)
    new = copy.deepcopy(old)
    for row in np.flatnonzero(old["approximate_teacher_mask"]):
        new["mcts_visits"][row] = 0
        new["mcts_visits"][row, old["approximate_teacher_edge"][row]] = 1
    datasets = [BoxesDataset([], games=[g], min_undrawn=24, features="chains") for g in (old, new)]
    report = MODULE.matched(*datasets)
    assert report["eligible_teacher_rows"] > 0 and report["changed_policy_rows"] > 0
    assert "exact_margin_p1" not in new and "policy_ab_root_mask" not in new
    assert not any(k.startswith("exact_policy") for k in new)
    for i, source_row in enumerate(datasets[0].position_indices[0]):
        assert int(old["to_play"][source_row]) == (0 if games[0]["slot"]["a_first"] else 1)
        a, b = datasets[0][i], datasets[1][i]
        assert a["margin"] == b["margin"]
        assert np.array_equal(a["planes"], b["planes"])
        assert bool(a["has_mcts"]) and bool(b["has_mcts"])
        assert not a["exact"] and not b["exact"]
        if old["approximate_teacher_mask"][source_row]:
            edge = int(old["approximate_teacher_edge"][source_row])
            assert float(b["policy"][edge]) == 1 and float(b["policy"].sum()) == 1
        else:
            assert np.array_equal(a["policy"], b["policy"])
    outside = next(
        i for i in datasets[1].position_indices[0] if not old["approximate_teacher_mask"][i]
    )
    new["mcts_visits"][outside] = 0
    with pytest.raises(AssertionError, match="outside teacher"):
        MODULE.matched(*datasets)


def test_export_roundtrip_verifies_receipts_and_excludes_heldout(tmp_path):
    source, games, _ = collection(tmp_path)
    blocked = benchmark(tmp_path)
    payload = tmp_path / "payload"
    report = MODULE.export(source, [blocked], payload)
    assert report["training_families"] == [games[0]["source_family"]]
    assert report["validation_families"] == [games[1]["source_family"]]
    keys = set(json.loads((payload / "exclusion-keys.json").read_text()))
    assert MODULE.exclusion_states(games[1]) <= keys
    assert MODULE.verify_payload(payload) == report
    MODULE.verified_collection(payload)
    (payload / "surprise.json").write_text("{}")
    with pytest.raises(AssertionError, match="file set changed"):
        MODULE.verify_payload(payload)


def test_changed_teacher_or_game_is_rejected_before_export(tmp_path):
    source, _, _ = collection(tmp_path)
    path = next((source / "teacher").glob("*.json"))
    saved = json.loads(path.read_text())
    saved["teacher_edge"] = saved["root"]["prefix"][0]
    MODULE.write_json(path, saved)
    with pytest.raises(ValueError, match="disagrees"):
        MODULE.verified_collection(source)


def test_d4_and_inherited_family_exclusions_use_four_edges(tmp_path):
    _, games, teachers = collection(tmp_path)
    game = MODULE.game_arrays(games[0], teachers, "b" * 64)
    assert MODULE.game_family(game) == games[0]["source_family"]
    basic = {k: v for k, v in game.items() if k not in ("source_family", "source_replay_group")}
    group = replay_group(basic, MODULE.GEO)
    inherited = dict(
        termination=np.array("solver"),
        source_replay_group=np.array(group.hex()),
        boards=basic["boards"][23:24],
        moves=np.zeros((1, 2), int),
    )
    assert MODULE.game_family(inherited) == MODULE.game_family(basic)
    prefix = games[0]["learner_decisions"][5]["prefix"]
    board, _ = replay_prefix(prefix)
    transformed = BoxesBoard(5)
    perm = edge_permutation(board.geo, 1)
    for edge in prefix:
        assert transformed.play_edge(int(perm[edge]))
    assert MODULE.board_key(board) == MODULE.board_key(transformed)
    assert MODULE.grid_key(board.to_numpy(), board.player()) == MODULE.board_key(board)


def test_value_target_or_family_changes_are_not_allowed(tmp_path):
    _, games, teachers = collection(tmp_path)
    old = MODULE.game_arrays(games[0], teachers, "b" * 64)
    new = copy.deepcopy(old)
    new["result"] = np.array("B+25.0" if str(old["result"]) != "B+25.0" else "W+25.0")
    a = BoxesDataset([], games=[old], min_undrawn=24)
    b = BoxesDataset([], games=[new], min_undrawn=24)
    with pytest.raises(AssertionError, match="result"):
        MODULE.matched(a, b)


@pytest.mark.parametrize("corruption", ["wrong_action", "exact_marker"])
def test_teacher_is_the_declared_legal_action_and_never_exact(tmp_path, corruption):
    _, games, teachers = collection(tmp_path)
    old = MODULE.game_arrays(games[0], teachers, "b" * 64)
    new = copy.deepcopy(old)
    for row in np.flatnonzero(old["approximate_teacher_mask"]):
        new["mcts_visits"][row] = 0
        new["mcts_visits"][row, old["approximate_teacher_edge"][row]] = 1
    row = int(np.flatnonzero(old["approximate_teacher_mask"])[0])
    if corruption == "wrong_action":
        correct = int(old["approximate_teacher_edge"][row])
        new["mcts_visits"][row, correct] = 0
        new["mcts_visits"][row, (correct + 1) % 60] = 1
    else:
        for game in (old, new):
            game["policy_ab_root_mask"] = np.zeros(60, bool)
            game["policy_ab_root_mask"][row] = True
    a = BoxesDataset([], games=[old], min_undrawn=24)
    b = BoxesDataset([], games=[new], min_undrawn=24)
    with pytest.raises(AssertionError, match="one-hot action|claim an exact policy"):
        MODULE.matched(a, b)
