"""Portable teacher targets, D4 action votes, immutable data and heldout boundaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alpha_go.boxes.dataset import BoxesDataset

SPEC = importlib.util.spec_from_file_location(
    "teacher_screen_data", Path(__file__).with_name("prepare_teacher_screen.py")
)
assert SPEC and SPEC.loader
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def row(seed: int, split: str = "training") -> dict[str, Any]:
    """Build one legal early root with truthful approximate-only markers."""
    order = np.random.default_rng(seed).permutation(60).tolist()
    return data.record(order[:10], order[10], f"family-{seed}", split, {"seed": seed})


def make_payload(folder: Path) -> None:
    folder.mkdir()
    for i, split in enumerate(("train", "validation", "diagnostic")):
        rows, _ = data.deduplicate([row(i + 1, "training" if split == "train" else split)])
        (folder / f"{split}.jsonl").write_text(json.dumps(rows[0]) + "\n")
    data.write_json(
        folder / "exclusions.json",
        dict(heldout_families=["family-2", "family-3"], benchmark_families=[], semantic_keys=[]),
    )
    data.write_json(
        folder / "manifest.json",
        dict(
            schema=1,
            status="complete",
            checkpoint_sha256=data.CHECKPOINT_SHA,
            artifact_sha256={p.name: data.sha256(p) for p in folder.iterdir()},
        ),
    )


def game(record: dict[str, Any], group: str, family: str | None) -> dict[str, Any]:
    board = data.board_from_record(record)
    result = dict(
        boards=np.array([board.to_numpy()], dtype=np.int8),
        to_play=np.array([board.player()], dtype=np.int8),
        moves=np.array([board.row_col(record["teacher_edge"])], dtype=np.int16),
        result=np.array("B+1.0"),
        source_replay_group=np.array(group.encode().hex()),
    )
    if family is not None:
        result["source_family"] = np.array(family)
    return result


def test_payload_replays_legal_targets_and_rejects_changes(tmp_path: Path) -> None:
    folder = tmp_path / "payload"
    make_payload(folder)
    payload = data.load_payload(folder)
    assert len(payload["train_records"]) == len(payload["validation_records"]) == 1
    for name in ("train_records", "validation_records", "diagnostic_records"):
        r = payload[name][0]
        assert not r["exact_policy"] and not r["exact_value"]
        assert data.board_from_record(r).is_legal_edge(r["teacher_edge"])
    (folder / "train.jsonl").write_text("changed")
    with pytest.raises(AssertionError, match="train.jsonl"):
        data.load_payload(folder)


def test_split_leakage_rejected_even_with_updated_hashes(tmp_path: Path) -> None:
    folder = tmp_path / "payload"
    make_payload(folder)
    (folder / "validation.jsonl").write_bytes((folder / "train.jsonl").read_bytes())
    manifest = data.read_json(folder / "manifest.json")
    manifest["artifact_sha256"]["validation.jsonl"] = data.sha256(folder / "validation.jsonl")
    data.write_json(folder / "manifest.json", manifest)
    with pytest.raises(AssertionError):
        data.load_payload(folder)


def test_d4_duplicate_actions_and_disagreements_preserve_votes() -> None:
    original = row(11)
    perm = data.source.PERMS[1]
    rotated = data.record(
        [int(perm[e]) for e in original["prefix"]],
        int(perm[original["teacher_edge"]]),
        "rotated-family",
        "training",
        {},
    )
    rows, report = data.deduplicate([original, rotated])
    assert report["unique_states"] == 1 and report["duplicate_occurrences"] == 1
    assert report["conflicting_states"] == 0
    assert rows[0]["teacher_policy"][original["teacher_edge"]] == 1
    board = data.board_from_record(original)
    action, _ = data.canonical_actions(board, original["teacher_edge"])
    other = next(
        e for e in board.get_legal_moves_flat() if data.canonical_actions(board, e)[0] != action
    )
    disagreement = data.record(original["prefix"], other, original["source_family"], "training", {})
    rows, report = data.deduplicate([original, rotated, disagreement])
    assert report["conflicting_states"] == 1
    assert rows[0]["teacher_policy"][original["teacher_edge"]] == pytest.approx(2 / 3)
    assert rows[0]["teacher_policy"][other] == pytest.approx(1 / 3)


def test_family_exclusion_propagates_to_descendants_and_keeps_source_immutable() -> None:
    records = [row(i) for i in (20, 21, 22, 23)]
    games = [
        game(records[0], "lineage-A", "blocked"),
        game(records[1], "lineage-A", None),
        game(records[2], "lineage-B", "safe"),
        game(records[3], "lineage-C", "safe-C"),
    ]
    dataset = BoxesDataset([], games=games, features="chains")
    selected, report = data.filter_replay(dataset, {"blocked"}, {records[2]["semantic_key"]})
    assert report == dict(family_rows=2, state_rows=1)
    assert len(selected) == 1 and len(dataset) == 4
    assert all("train_mask" not in g for g in games)
    assert str(selected.games[0]["source_family"]) == "safe-C"


def test_semantic_exclusion_respects_mover_relative_scores() -> None:
    record = row(24)
    board = data.board_from_record(record)
    original = board.to_numpy()
    original[1, 1], original[1, 3] = 1, 2
    swapped = original.copy()
    owners = swapped[1::2, 1::2]
    owners[:] = np.where(owners == 1, 2, np.where(owners == 2, 1, owners))
    assert data.source.grid_key(original, 0) == data.source.grid_key(swapped, 1)
    swapped[1, 1] = 0
    assert data.source.grid_key(original, 0) != data.source.grid_key(swapped, 1)


def test_illegal_teacher_edge_is_rejected() -> None:
    r = row(25)
    r["teacher_edge"] = r["prefix"][0]
    with pytest.raises(AssertionError, match="illegal teacher edge"):
        data.board_from_record(r)
