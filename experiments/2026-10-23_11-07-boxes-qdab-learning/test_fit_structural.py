"""Small-board semantics/isolation checks, not evidence of learning strength."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import alpha_go_cpp
import numpy as np
import pytest
import torch
from fit_structural import (
    ARMS,
    cache_features,
    child_reads,
    choice_metrics,
    fit_heads,
    load_bundle,
    run,
    sample_indices,
    semantic_key,
    sha256,
    summarize_children,
    training_groups,
    write_json,
)

from alpha_go.boxes.encode import encode
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import save_boxes_net
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard


def make_bundle(folder: Path, prefixes: tuple | None = None
                ) -> tuple[list[dict], list[dict], list[BoxesBoard]]:
    """Two distinct early root families with exact tiny-board complete children."""
    oracle = Oracle(2)
    states, roots, boards = [], [], []

    def add(prefix: list[int], family: str, split: str, role: str) -> str:
        board = BoxesBoard(2)
        for edge in prefix:
            assert board.play_edge(edge)
        state_id = f"s{len(states)}"
        value = oracle.final_margin(board) * (1 if board.to_play() == 1 else -1)
        states.append(dict(state_id=state_id, rows=2, cols=2, prefix=prefix,
            edges=str(board.edges), to_play=board.to_play(), scores=board.boxes,
            score_p1=int(board.score()), exact_margin_p1=value, status="exact",
            source_family=family, split=split, root_groups=[f"root-{split}"], roles=[role],
            canonical_d4_key=semantic_key(board).hex(), training_eligible=True,
            exclusion_reasons=[]))
        boards.append(board)
        return state_id

    for split, prefix in (prefixes or (("training", [0]), ("validation", [0, 2, 4, 10]))):
        family = f"family-{split}"
        root_id = f"root-{split}"
        state_id = add(prefix, family, split, "root")
        board = boards[-1]
        children = []
        for edge in board.get_legal_moves_flat():
            raw_id = add(prefix + [edge], family, split, "raw_child")
            value = states[-1]["exact_margin_p1"]
            native = alpha_go_cpp.BoxesBoard(2, 2)
            for move in prefix + [edge]:
                native.play_edge(move)
            collapsed_prefix = list(alpha_go_cpp.BoxesSearchState(native).prefix())
            collapsed_id = add(prefix + [edge] + collapsed_prefix,
                               family, split, "collapsed_child")
            assert states[-1]["exact_margin_p1"] == value
            children.append(dict(edge=edge, raw_state_id=raw_id,
                                 collapsed_state_id=collapsed_id, status="exact",
                                 exact_margin_p1=value))
        roots.append(dict(root_id=root_id, source_game=family, source_family=family,
            split=split, prefix=prefix, state_id=state_id, to_play=board.to_play(),
            played_edge=children[0]["edge"], actor="learner", undrawn=12-len(prefix),
            training_eligible=True, exclusion_reasons=[], status="exact",
            complete_children=True, root_derived=False, children=children))
    folder.mkdir()
    for name, rows in (("states.jsonl", states), ("roots.jsonl", roots)):
        (folder / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(folder / "exclusion_receipt.json", dict(known_exclusions_verified=True,
        eligible_train_validation_intersection=0, inputs=[], scope="Synthetic small-board fixture"))
    write_json(folder / "manifest.json", dict(schema=1, status="complete",
        known_exclusions_verified=True, exclusions_verified=False,
        overlap_pending_actual_gpu=True, development_only=True,
        exclusion_receipt_sha256=sha256(folder / "exclusion_receipt.json"),
        artifact_sha256={name: sha256(folder / name) for name in
                        ("states.jsonl", "roots.jsonl", "exclusion_receipt.json")}))
    return states, roots, boards


def refresh_digest(folder: Path, name: str) -> None:
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["artifact_sha256"][name] = sha256(folder / name)
    write_json(folder / "manifest.json", manifest)


def test_bundle_replay_exclusions_and_tampering(tmp_path: Path) -> None:
    folder = tmp_path / "prepared"
    make_bundle(folder)
    with pytest.raises(ValueError, match="explicit development"):
        load_bundle(folder, False)
    _, states, roots, _ = load_bundle(folder, True)
    assert len(states) > 20 and len(roots) == 2
    states[0]["scores"] = [1, 0]
    (folder / "states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in states))
    with pytest.raises(ValueError, match="digest"):
        load_bundle(folder, True)
    refresh_digest(folder, "states.jsonl")
    with pytest.raises(ValueError, match="Replayed"):
        load_bundle(folder, True)


def test_semantic_overlap_ignores_owner_placement_and_physical_color() -> None:
    board = BoxesBoard(2)
    for edge in (0, 1, 2):
        board.play_edge(edge)
    flipped = board.copy()
    flipped._to_play = 3 - flipped.to_play()
    flipped.boxes.reverse()
    assert semantic_key(board) == semantic_key(flipped)
    different_score = board.copy()
    different_score.boxes = [1, 0]
    assert semantic_key(board) != semantic_key(different_score)


def test_cross_split_overlap_and_missing_complete_children_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "prepared"
    states, roots, _ = make_bundle(folder)
    duplicate = dict(states[0], state_id="duplicate", split="validation",
                     source_family="another-family")
    (folder / "states.jsonl").write_text("".join(
        json.dumps(row) + "\n" for row in states + [duplicate]))
    refresh_digest(folder, "states.jsonl")
    with pytest.raises(ValueError, match="leakage"):
        load_bundle(folder, True)
    (folder / "states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in states))
    refresh_digest(folder, "states.jsonl")
    roots[0]["children"].pop()
    (folder / "roots.jsonl").write_text("".join(json.dumps(row) + "\n" for row in roots))
    refresh_digest(folder, "roots.jsonl")
    with pytest.raises(ValueError, match="omits legal"):
        load_bundle(folder, True)


@pytest.mark.parametrize("field,value,match", [
    ("status", "node_budget", "does not establish"),
    ("exact_margin_p1", False, "Impossible exact"),
])
def test_nonexact_status_and_boolean_labels_rejected(
        tmp_path: Path, field: str, value: object, match: str) -> None:
    folder = tmp_path / "prepared"
    states, _, _ = make_bundle(folder)
    states[0][field] = value
    (folder / "states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in states))
    refresh_digest(folder, "states.jsonl")
    with pytest.raises(ValueError, match=match):
        load_bundle(folder, True)


def test_wrong_legal_collapsed_child_is_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "prepared"
    _, roots, _ = make_bundle(folder)
    # An unrelated sibling is legal and has the same exact label on this root.
    roots[0]["children"][0]["collapsed_state_id"] = roots[0]["children"][1]["collapsed_state_id"]
    (folder / "roots.jsonl").write_text("".join(json.dumps(row) + "\n" for row in roots))
    refresh_digest(folder, "roots.jsonl")
    with pytest.raises(ValueError, match="forced-capture"):
        load_bundle(folder, True)


def test_cache_captures_exact_inputs_and_keeps_base_frozen() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = BoxesNet(2, channels=8, n_blocks=1, value_hidden=8, features="chains")
    with torch.no_grad():
        model.value_fc2.weight.normal_(std=.05)
    boards = [BoxesBoard(2), BoxesBoard(2)]
    boards[1].play_edge(0)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    arrays = cache_features(model, boards, 2, torch.device("cpu"))
    planes = torch.from_numpy(np.stack([encode(board, "chains") for board in boards]))
    with torch.no_grad():
        _, value = model(planes)
        from_pooled = model.value_fc2(torch.relu(model.value_fc1(
            torch.from_numpy(arrays["pooled_score"]))))
    np.testing.assert_array_equal(arrays["base_logits"], value.numpy())
    torch.testing.assert_close(from_pooled, value, rtol=0, atol=0)
    assert np.all(arrays["pooled_score"][:, -1] == 0)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    targets = np.array([4, 6])
    indices = np.array([[0, 1], [1, 0]], np.int32)
    predictions, _ = fit_heads(arrays, targets, indices, lr=.001, seed=6,
                              device=torch.device("cpu"))
    assert set(predictions) == {"base", *ARMS}
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


def test_sampler_balances_families_roots_and_unknowns(tmp_path: Path) -> None:
    states, roots, boards = make_bundle(tmp_path / "prepared")
    states[0]["exact_margin_p1"] = None
    states[0]["status"] = "node_budget"
    groups = training_groups(states, roots, boards)
    assert 0 not in [index for roots in groups.values() for group in roots for index in group]
    toy = {"small": [[0]], "large": [[1] * 37, [2]]}
    samples = sample_indices(toy, 200, 64, 12)
    counts = np.bincount(samples.ravel(), minlength=3) / samples.size
    assert .48 < counts[0] < .52
    assert .23 < counts[1] < .27 and .23 < counts[2] < .27
    np.testing.assert_array_equal(samples, sample_indices(toy, 200, 64, 12))


def test_child_perspective_capture_terminal_and_exact_ties(tmp_path: Path) -> None:
    states, roots, boards = make_bundle(tmp_path / "prepared", (
        ("training", [0]), ("validation", [9, 11, 1, 3, 2, 4, 6, 7])))
    logits = np.full((len(states), 9), -100., dtype=np.float32)
    for index, row in enumerate(states):
        value = row["exact_margin_p1"] * (1 if row["to_play"] == 1 else -1)
        logits[index, value + 4] = 100
    reads = child_reads({"perfect": logits}, states, roots, boards)
    assert reads and all(row["regret"] == 0 for row in reads if row["predictor"] != "win")
    assert all(row["win_preserved"] in (1, None) for row in reads)
    assert all(row["pairwise_accuracy"] in (1., None) for row in reads
               if row["predictor"] != "win")
    assert any(row["terminal_exact"] for row in reads)
    ids = {row["state_id"]: row for row in states}
    movers_changed = [ids[c["raw_state_id"]]["to_play"] != roots[1]["to_play"]
                      for c in roots[1]["children"]]
    assert any(movers_changed) and not all(movers_changed)
    root = roots[0]
    root["complete_children"] = False
    assert all(row["root_id"] != root["root_id"] for row in
               child_reads({"perfect": logits}, states, roots, boards))
    metric = choice_metrics(np.array([1., 1., 0.]), np.array([2., 2., -2.]), [0, 1, 2])
    assert metric["distinct_pairs"] == 2 and metric["pairwise_accuracy"] == 1
    tied = choice_metrics(np.zeros(3), np.array([2., 2., -2.]), [0, 1, 2])
    assert tied["pairwise_accuracy"] == .5 and tied["spearman"] is None


def test_family_delta_bootstrap_and_strict_json() -> None:
    rows = []
    for family, count in (("many-roots", 10), ("one-root", 1)):
        for index in range(count):
            for arm, regret in (("base", 2.), ("structural_residual", 1.)):
                rows.append(dict(arm=arm, split="validation", family=family,
                    root_id=f"{family}-{index}", mode="collapsed", predictor="utility", band=36,
                    regret=regret, optimal=0, winning_root=True, win_preserved=0,
                    pairwise_accuracy=.5, spearman=None))
    summary = summarize_children(rows, 1)
    delta = next(row for row in summary["validation_deltas"] if row["metric"] == "regret")
    assert delta["delta"] == -1 and delta["family_ci95"] == [-1, -1]
    assert delta["families"] == 2
    json.dumps(summary, allow_nan=False)


def test_complete_tiny_fit_outputs_and_cache_validation(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    make_bundle(prepared)
    # Clearing only the manifest must not clear the receipt's pending exclusions.
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest.update(exclusions_verified=True, overlap_pending_actual_gpu=False)
    write_json(prepared / "manifest.json", manifest)
    checkpoint = tmp_path / "base.pt"
    torch.manual_seed(4)
    model = BoxesNet(2, channels=8, n_blocks=1, value_hidden=8, features="chains")
    save_boxes_net(model, checkpoint)
    args = argparse.Namespace(checkpoint=checkpoint, expected_checkpoint_sha256=sha256(checkpoint),
        prepared=prepared, output=tmp_path / "fit", steps=2, batch_size=4, lr=.001,
        seed=3, threads=1, device="cpu", development_with_pending_exclusions=True)
    summary = run(args)
    assert summary["status"] == "complete" and not summary["arena_eligible"]
    assert "PROVISIONAL" in summary["scope"]
    assert len(summary["states"]) == 8 and summary["children"]["group_means"]
    assert (args.output / "inputs" / "states.jsonl").read_bytes() == (
        prepared / "states.jsonl").read_bytes()
    first_hash = sha256(args.output / "cache.npz")
    run(args)
    assert sha256(args.output / "cache.npz") == first_hash
    (args.output / "cache.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cache mismatch"):
        run(args)
