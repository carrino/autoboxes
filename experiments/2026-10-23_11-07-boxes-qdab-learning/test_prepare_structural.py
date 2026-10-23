"""Small semantic tests; no timed opponent and no large-board solver calls."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

PATH = Path(__file__).with_name("prepare_structural.py")
SPEC = importlib.util.spec_from_file_location("prepare_structural_under_test", PATH)
assert SPEC and SPEC.loader
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def record(prefix, family="a", split="training", role="root", game="0000", root="r"):
    board, native = prepare.replay(prefix, 1, 2)
    return prepare.state_record(board, prefix, family, split, game, root, role), board, native


def test_double_capture_keeps_turn_and_fixed_player_margin():
    before, _ = prepare.replay([0, 1, 2, 3, 4, 6], 1, 2)
    mover = before.to_play()
    assert before.play_edge(5)
    assert before.to_play() == mover == 1 and before.boxes == [2, 0]
    assert prepare.final_p1(before, 0) == 2
    # A single capture can retain physical player two too.
    player_two, _ = prepare.replay([0, 2, 6, 7], 2, 2)
    assert player_two.to_play() == 2 and player_two.boxes == [0, 1]
    assert prepare.final_p1(player_two, 1) == -2
    with pytest.raises(AssertionError, match="invalid exact"):
        prepare.final_p1(player_two, 0)


def test_semantic_key_blocks_symmetry_and_physical_color_swaps():
    board, _ = prepare.replay([0, 2, 6, 7], 2, 2)
    permutation = prepare.edge_permutation(board.geo, 1)
    symmetric, _ = prepare.replay([int(permutation[e]) for e in [0, 2, 6, 7]], 2, 2)
    assert prepare.semantic_key(board) == prepare.semantic_key(symmetric)
    swapped = board.copy()
    swapped._to_play = 3 - swapped.to_play()
    swapped.boxes.reverse()
    swapped.owner = [3 - owner if owner else 0 for owner in swapped.owner]
    assert prepare.semantic_key(board) == prepare.semantic_key(swapped)
    train = prepare.state_record(board, [], "train-family", "training", "0", "t", "root")
    heldout = prepare.state_record(swapped, [], "val-family", "validation", "1", "v", "root")
    states = {s["state_id"]: s for s in (train, heldout)}
    roots = [
        dict(root_id=r, state_id=s["state_id"], children=[])
        for r, s in [("t", train), ("v", heldout)]
    ]
    result = prepare.apply_exclusions(states, roots, set())
    assert result["eligible_train_validation_intersection"] == 0
    assert not train["training_eligible"] and heldout["training_eligible"]
    assert "validation_semantic_transposition" in train["exclusion_reasons"]


def tiny_group(prefix):
    board, native = prepare.replay(prefix, 1, 2)
    states = {}
    root_id = prepare.add_state(
        states, board, native, prefix, "family", "training", "0000", "r", "root"
    )
    children = []
    for edge in board.get_legal_moves_flat():
        child, child_native = prepare.replay(prefix + [edge], 1, 2)
        sid = prepare.add_state(
            states,
            child,
            child_native,
            prefix + [edge],
            "family",
            "training",
            "0000",
            "r",
            "raw_child",
        )
        children.append(
            dict(
                edge=edge,
                raw_state_id=sid,
                collapsed_state_id=sid,
                exact_margin_p1=None,
                status="not_attempted",
            )
        )
    root = dict(
        root_id="r",
        state_id=root_id,
        to_play=board.to_play(),
        undrawn=len(children),
        children=children,
        root_derived=False,
    )
    return states, root


def test_unknown_direct_root_recovered_only_from_complete_children():
    states, root = tiny_group([0, 1, 2, 3, 4, 6])
    root_state = states[root["state_id"]]
    root_state.update(status="node_budget", direct_query_status="node_budget")
    child = states[root["children"][0]["raw_state_id"]]
    prepare.set_exact(child, 2, "exact", "fake-bounded-query")
    prepare.reconcile(states, [root])
    assert root["complete_children"] and root["root_derived"]
    assert root["exact_margin_p1"] == 2 and root["optimal_edges"] == [5]
    assert root["direct_query_status"] == "node_budget"


def test_partial_children_do_not_become_policy_or_unknown_root_label():
    states, root = tiny_group([0, 1, 2, 3, 4])
    child = states[root["children"][0]["raw_state_id"]]
    prepare.set_exact(child, 0, "exact", "fake-query")
    prepare.reconcile(states, [root])
    assert not root["complete_children"] and root["known_children"] == 1
    assert root["exact_margin_p1"] is None and "optimal_edges" not in root
    assert root["children"][1]["exact_margin_p1"] is None


def full_source():
    from alpha_go.boxes.qdab_training_collect import opening_family

    board = prepare.BoxesBoard(5)
    moves = [list(board.row_col(e)) for e in range(60)]
    opening = moves[:4]
    slot = dict(
        index=0,
        a_first=True,
        panel="training",
        seed=1,
        opening=opening,
        source_family=opening_family(opening),
        split="training",
    )
    counts = [0, 0]
    for i in range(60):
        if i >= 4:
            counts[board.player()] += 1
        assert board.play_edge(i)
    margin = int(board.score())
    return dict(
        slot=slot,
        moves=moves,
        a_margin=margin,
        result=f"{'B' if margin > 0 else 'W'}+{abs(margin):.1f}",
        termination="all_edges",
        a_moves=counts[0],
        b_moves=counts[1],
        a_seconds=0.0,
        b_seconds=0.0,
        source_family=slot["source_family"],
        split="training",
        learner_decisions=[],
    )


def test_full_replay_freezes_both_actors_all_children_and_capture_prefixes():
    source = full_source()
    states, roots = {}, []
    prepare.freeze_game(source, states, roots)
    assert {r["undrawn"] for r in roots} == {35, 36, 37, 38}
    assert len(roots) == 4 and {r["actor"] for r in roots} == {"learner", "Qdab"}
    for root in roots:
        assert len(root["children"]) == root["undrawn"]
        for child in root["children"]:
            raw, _ = prepare.replay(states[child["raw_state_id"]]["prefix"])
            collapsed, _ = prepare.replay(states[child["collapsed_state_id"]]["prefix"])
            expected, _ = prepare.replay(root["prefix"] + [child["edge"]])
            assert raw.to_numpy().tolist() == expected.to_numpy().tolist()
            for edge in child["forced_prefix"]:
                assert expected.play_edge(edge)
            assert expected.to_numpy().tolist() == collapsed.to_numpy().tolist()
    bad = copy.deepcopy(source)
    bad["a_margin"] += 2
    with pytest.raises(ValueError, match="outcome"):
        prepare.freeze_game(bad, {}, [])


def test_phase_one_only_roots_and_played_children_before_alternatives():
    states, roots = {}, []
    prepare.freeze_game(full_source(), states, roots)
    first = prepare.task_order(roots, states, 1)
    ids = {s["state_id"] for rows in first.values() for s in rows}
    expected = {r["state_id"] for r in roots} | {
        next(c["raw_state_id"] for c in r["children"] if c["edge"] == r["played_edge"])
        for r in roots
    }
    assert ids <= expected
    all_queries = {
        s["query_id"] for rows in prepare.task_order(roots, states, 2).values() for s in rows
    }
    assert len(all_queries) > len(ids)


def test_shared_family_two_games_keep_private_state_and_query_owners():
    board, native = prepare.replay([0, 1], 1, 2)
    states = {}
    first = prepare.add_state(
        states, board, native, [0, 1], "family", "training", "0000", "root0", "root"
    )
    second = prepare.add_state(
        states, board, native, [0, 1], "family", "training", "0001", "root1", "root"
    )
    assert first != second
    assert states[first]["query_id"] != states[second]["query_id"]
    assert states[first]["source_games"] == ["0000"]
    assert states[second]["source_games"] == ["0001"]
    assert states[first]["canonical_d4_key"] == states[second]["canonical_d4_key"]


def test_query_unknown_is_saved_and_reused_without_resolving(tmp_path, monkeypatch):
    task, _, _ = record([0, 1])
    calls = []

    class FakeSolver:
        def __init__(self, *args):
            pass

        def value_within(self, edges, budget):
            calls.append((edges, budget))
            return None

        def nodes(self):
            return 100

    monkeypatch.setattr(prepare.cpp, "BoxesSolver", FakeSolver)
    (tmp_path / "receipts").mkdir()
    settings = dict(node_budget=100, table_entries=64)
    assert prepare.solve_group(tmp_path, "0000", [task], settings, time.monotonic() + 1) == 1
    assert prepare.solve_group(tmp_path, "0000", [task], settings, time.monotonic() + 1) == 0
    assert len(calls) == 1
    path = next((tmp_path / "receipts/0000").glob("*.json"))
    saved = json.loads(path.read_text())
    assert saved["status"] == "node_budget" and saved["remaining"] is None
    saved["remaining"] = 1
    path.write_text(json.dumps(saved))
    with pytest.raises(AssertionError, match="receipt changed"):
        prepare.solve_group(tmp_path, "0000", [task], settings, time.monotonic() + 1)


def test_active_collection_lock_blocks_preparation(tmp_path):
    import fcntl

    with (tmp_path / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="still active"):
            prepare.active_collection(tmp_path)


def test_supervisor_interrupt_reaps_worker_and_exports_partial(tmp_path):
    # A tiny sleeping subprocess stands in for a native worker; no solver is called.
    harness = tmp_path / "harness.py"
    harness.write_text(
        """import importlib.util, json, signal, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("prep", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
out = Path(sys.argv[2])
def fake_export(output, session):
    (out/"manifest.json").write_text(json.dumps(dict(status="partial", session=session)))
    return dict(status="partial")
m.export = fake_export
def stop(*args):
    raise KeyboardInterrupt("test SIGTERM")
signal.signal(signal.SIGTERM, stop)
child = [sys.executable, "-c", "import os,time,pathlib;pathlib.Path(" +
         repr(str(out/"worker.pid")) + ").write_text(str(os.getpid()));time.sleep(60)"]
m.supervise(out, 60, 1, command=child)
"""
    )
    proc = subprocess.Popen(
        [sys.executable, str(harness), str(PATH), str(tmp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "worker.pid").exists() and time.monotonic() < deadline:
            assert proc.poll() is None
            time.sleep(0.05)
        assert (tmp_path / "worker.pid").exists()
        worker = int((tmp_path / "worker.pid").read_text())
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) != 0
        with pytest.raises(ProcessLookupError):
            os.kill(worker, 0)
        report = json.loads((tmp_path / "manifest.json").read_text())
        assert report["status"] == "partial" and report["session"]["interrupted"]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_actual_export_to_tiny_structural_fit(tmp_path):
    """Exercise the real preparer exporter and fitter together, using only a 2x2 oracle."""
    import argparse

    import torch
    from fit_structural import run as fit_run

    from alpha_go.boxes.model import BoxesNet
    from alpha_go.boxes.nn_agent import save_boxes_net
    from alpha_go.boxes.oracle import Oracle

    folder = tmp_path / "prepared"
    (folder / "frozen").mkdir(parents=True)
    (folder / "receipts").mkdir()
    states, roots = {}, []
    oracle = Oracle(2)
    for i, (split, prefix) in enumerate([("training", [0]), ("validation", [0, 2, 4, 10])]):
        game = f"{i:04d}"
        family = f"tiny-{split}"
        rootid = f"root-{split}"
        board, native = prepare.replay(prefix, 2, 2)
        root_state = prepare.add_state(
            states, board, native, prefix, family, split, game, rootid, "root"
        )
        children = []
        for edge in board.get_legal_moves_flat():
            raw, raw_native = prepare.replay(prefix + [edge], 2, 2)
            raw_id = prepare.add_state(
                states, raw, raw_native, prefix + [edge], family, split, game, rootid, "raw_child"
            )
            collapsed_native = prepare.cpp.BoxesSearchState(raw_native)
            forced = list(collapsed_native.prefix())
            col, col_native = prepare.replay(prefix + [edge] + forced, 2, 2)
            col_id = prepare.add_state(
                states,
                col,
                col_native,
                prefix + [edge] + forced,
                family,
                split,
                game,
                rootid,
                "collapsed_child",
            )
            children.append(
                dict(
                    edge=edge,
                    raw_state_id=raw_id,
                    collapsed_state_id=col_id,
                    forced_prefix=forced,
                    status="not_attempted",
                    exact_margin_p1=None,
                )
            )
        roots.append(
            dict(
                root_id=rootid,
                source_game=game,
                source_family=family,
                split=split,
                prefix=prefix,
                state_id=root_state,
                to_play=board.to_play(),
                played_edge=children[0]["edge"],
                actor="learner",
                undrawn=len(children),
                training_eligible=True,
                exclusion_reasons=[],
                status="not_attempted",
                root_derived=False,
                complete_children=False,
                children=children,
            )
        )
    report = prepare.apply_exclusions(states, roots, set())
    report.update(
        known_exclusions_verified=True, inputs=[], scope="Synthetic 2x2 contract fixture only"
    )
    prepare.write_json(folder / "exclusion_receipt.json", report)
    prepare.save_jsonl(folder / "frozen/states.jsonl", list(states.values()))
    prepare.save_jsonl(folder / "frozen/roots.jsonl", roots)
    for state in states.values():
        board, _ = prepare.replay(state["prefix"], 2, 2)
        remaining = int(oracle.final_margin(board)) - board.margin()
        receipt = dict(
            query_id=state["query_id"],
            edges=state["edges"],
            node_budget=100,
            remaining=remaining,
            status="exact",
            nodes=0,
            seconds=0.0,
            fixture_evidence="2x2 exhaustive oracle",
        )
        receipt["payload_sha256"] = prepare.digest_object(receipt)
        path = folder / "receipts" / state["source_games"][0] / f"{state['query_id']}.json"
        path.parent.mkdir(exist_ok=True)
        prepare.write_json(path, receipt)
    torch.set_num_threads(1)
    torch.manual_seed(4)
    checkpoint = tmp_path / "base.pt"
    save_boxes_net(
        BoxesNet(2, channels=8, n_blocks=1, value_hidden=8, features="chains"), checkpoint
    )
    recipe = dict(
        checkpoint_sha256=prepare.sha256(checkpoint),
        source_games=2,
        exclusion_receipt_sha256=prepare.sha256(folder / "exclusion_receipt.json"),
        settings=dict(node_budget=100),
    )
    prepare.write_json(folder / "recipe.json", recipe)
    exported = prepare.export(folder, dict(timed_out=False))
    assert exported["status"] == "complete" and exported["complete_root_groups"] == 2
    args = argparse.Namespace(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=prepare.sha256(checkpoint),
        prepared=folder,
        output=tmp_path / "fit",
        steps=2,
        batch_size=4,
        lr=0.001,
        seed=3,
        threads=1,
        device="cpu",
        development_with_pending_exclusions=True,
    )
    result = fit_run(args)
    assert result["status"] == "complete" and not result["arena_eligible"]
    assert result["ranking_coverage"]["validation"]["complete_families"] == 1
