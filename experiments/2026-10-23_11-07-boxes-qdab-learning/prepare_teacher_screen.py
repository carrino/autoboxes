"""Portable approximate Qdab targets and leakage-safe replay for a short fit screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from alpha_go.boxes.balanced_policy import (
    clean_validation,
    coverage,
    read_labels,
    verify_root_targets,
)
from alpha_go.boxes.curriculum_train import subset
from alpha_go.boxes.policy_target_ab import bundle_files, load_arm, sha256
from alpha_go.boxes.qdab import decode_result, replay_prefix
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.boxes.validation import replay_group

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("screen_source_helpers", HERE / "prepare_teacher.py")
assert SPEC and SPEC.loader
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)
CHECKPOINT_SHA = "c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f"


def read_json(path: Path) -> Any:
    """Read a finite saved artifact."""
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    """Write a portable artifact with strict JSON numbers."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read ordered nonempty JSON records."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def board_from_record(record: dict[str, Any]) -> BoxesBoard:
    """Replay a legal original-orientation teacher root and validate its action."""
    board, _ = replay_prefix(record["prefix"])
    assert board.is_legal_edge(int(record["teacher_edge"])), "illegal teacher edge"
    return board


def record(
    prefix: list[int], edge: int, family: str, split: str, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Keep an approximate action without inventing a value or search visits."""
    result = dict(
        prefix=prefix,
        teacher_edge=edge,
        source_family=family,
        split=split,
        provenance=provenance,
        target_kind="approximate_qdab_action",
        exact_policy=False,
        exact_value=False,
    )
    board = board_from_record(result)
    assert 38 <= 60 - len(prefix) <= 50, "teacher root outside declared band"
    result.update(
        semantic_key=source.board_key(board),
        to_play=board.to_play(),
        scores=list(board.boxes),
        undrawn=60 - len(prefix),
    )
    return result


def canonical_actions(board: BoxesBoard, edge: int) -> tuple[int, list[int]]:
    """Map an action to its orbit under all board-minimizing D4 transforms."""
    drawn = [e for e in range(60) if board.edges >> e & 1]
    masks = [sum(1 << int(p[e]) for e in drawn) for p in source.PERMS]
    transforms = [i for i, mask in enumerate(masks) if mask == min(masks)]
    return min(int(source.PERMS[i][edge]) for i in transforms), transforms


def deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Combine repeated semantic roots; preserve disagreements as approximate votes."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(row["semantic_key"], []).append(row)
    output = []
    conflicts = symmetric = 0
    for key, rows in sorted(grouped.items()):
        representative = dict(rows[0])
        votes: Counter[int] = Counter()
        for row in rows:
            board = board_from_record(row)
            action, _ = canonical_actions(board, int(row["teacher_edge"]))
            votes[action] += 1
        board = board_from_record(representative)
        _, transforms = canonical_actions(board, int(representative["teacher_edge"]))
        target: Any = np.zeros(60, dtype=np.float64)
        for action, count in votes.items():
            orbit = [
                e
                for e in board.get_legal_moves_flat()
                if min(int(source.PERMS[i][e]) for i in transforms) == action
            ]
            assert orbit, "canonical teacher action has no original-orientation equivalent"
            target[orbit] += count / len(orbit)
        target /= target.sum()
        conflicts += len(votes) > 1
        symmetric += int(np.count_nonzero(target)) > len(votes)
        representative.update(
            teacher_policy=target.tolist(),
            source_families=sorted({r["source_family"] for r in rows}),
            occurrences=len(rows),
            action_votes=dict(votes),
            provenance=[r["provenance"] for r in rows],
        )
        output.append(representative)
    return output, dict(
        occurrences=len(records),
        unique_states=len(output),
        duplicate_occurrences=len(records) - len(output),
        conflicting_states=conflicts,
        symmetry_orbit_targets=symmetric,
    )


def export(
    bank: Path,
    collections: list[Path],
    primary_collection: Path,
    benchmarks: list[Path],
    output: Path,
) -> dict[str, Any]:
    """Freeze recycled match teaching plus fresh advice, with sealed family holdouts."""
    assert not output.exists(), "payload output must be fresh"
    assert len(collections) == 2 and primary_collection.resolve() in {
        p.resolve() for p in collections
    }
    provenance = read_json(bank / "provenance.json")
    for name, expected in provenance["outputs_sha256"].items():
        assert sha256(bank / name) == expected, f"retired bank changed: {name}"
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []
    heldout_families: set[str] = set()
    exclusion_keys: set[str] = set()
    benchmark_families: set[str] = set()
    sources = {
        "retired_bank": dict(path=str(bank), provenance_sha256=sha256(bank / "provenance.json"))
    }
    for benchmark in benchmarks:
        config = read_json(benchmark / "config.json")
        for slot in config["schedule"]:
            game_path = benchmark / "games" / f"{slot['index']:04d}.json"
            game = read_json(game_path)
            source.verify_game(game, slot)
            assert game["config_sha256"] == sha256(benchmark / "config.json")
            exclusion_keys.update(source.exclusion_states(game))
            if len(slot["opening"]) == 4:
                benchmark_families.add(source.opening_family(slot["opening"]))
        sources[str(benchmark)] = dict(
            config_sha256=sha256(benchmark / "config.json"),
            summary_sha256=sha256(benchmark / "summary.json"),
        )
    for collection in collections:
        config, games, teachers = source.verified_collection(collection)
        assert config["checkpoint_sha256"] == CHECKPOINT_SHA
        is_primary = collection.resolve() == primary_collection.resolve()
        for game in games:
            if game["split"] == "validation":
                heldout_families.add(game["source_family"])
                exclusion_keys.update(source.exclusion_states(game))
        for name, saved in sorted(teachers.items()):
            split = (
                "training"
                if saved["split"] == "training"
                else ("validation" if is_primary else "diagnostic")
            )
            row = record(
                saved["root"]["prefix"],
                int(saved["teacher_edge"]),
                saved["source_family"],
                split,
                dict(
                    kind="fresh_learner_query",
                    source=name,
                    teacher_sha256=sha256(collection / "teacher" / name),
                    collection_config_sha256=sha256(collection / "config.json"),
                ),
            )
            {"training": train, "validation": validation, "diagnostic": diagnostic}[split].append(
                row
            )
        sources[str(collection)] = dict(
            config_sha256=sha256(collection / "config.json"),
            summary_sha256=sha256(collection / "summary.json"),
        )
    retired_families: set[str] = set()
    for saved in read_jsonl(bank / "occurrences.jsonl"):
        if saved["quarantine_reasons"]:
            continue
        prefix = saved["root"]["prefix"]
        board, request = replay_prefix(prefix)
        reply = saved["teacher_response"]
        assert reply["rpc_request"]["params"][0]["Board"] == request
        assert decode_result(reply["rpc_response"]["result"], board) == reply["edges"]
        assert saved["teacher_edge"] == reply["edges"][0]
        retired_families.add(saved["source_family"])
        train.append(
            record(
                prefix,
                int(saved["teacher_edge"]),
                saved["source_family"],
                "training",
                dict(
                    kind="retired_completed_match",
                    occurrence_id=saved["occurrence_id"],
                    source_game_sha256=saved["source_game_sha256"],
                    match_config_sha256=saved["match_config_sha256"],
                ),
            )
        )
    blocked = heldout_families | benchmark_families
    before = len(train)
    train = [
        r
        for r in train
        if r["source_family"] not in blocked and r["semantic_key"] not in exclusion_keys
    ]
    assert train and validation and diagnostic, "missing a declared source split"
    output.mkdir(parents=True)
    summaries = {}
    for split, rows in (("train", train), ("validation", validation), ("diagnostic", diagnostic)):
        rows, counts = deduplicate(rows)
        summaries[split] = dict(
            **counts, families=len({f for r in rows for f in r["source_families"]})
        )
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows)
        )
    write_json(
        output / "exclusions.json",
        dict(
            heldout_families=sorted(heldout_families),
            benchmark_families=sorted(benchmark_families),
            semantic_keys=sorted(exclusion_keys),
            retired_training_families=sorted(retired_families),
            scope=(
                "Heldout/retained benchmark families and listed trajectories/critical children "
                "excluded. Retired match families must be excluded from future evaluation."
            ),
        ),
    )
    manifest = dict(
        schema=1,
        status="complete",
        checkpoint_sha256=CHECKPOINT_SHA,
        counts=summaries,
        removed_training_occurrences=before - len(train),
        primary_validation="Fresh second collection: 20 frozen families",
        diagnostic_validation="Previously read first collection: 15 families; diagnostic only",
        source_sha256=sources,
        target_kind="approximate_qdab_action_not_exact",
        value_targets="none; teacher rows must never contribute value loss",
        artifact_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    write_json(output / "manifest.json", manifest)
    load_payload(output)
    return manifest


def load_payload(path: Path) -> dict[str, Any]:
    """Verify the exact payload file set, hashes, legal targets and split separation."""
    manifest = read_json(path / "manifest.json")
    assert manifest["schema"] == 1 and manifest["status"] == "complete"
    assert manifest["checkpoint_sha256"] == CHECKPOINT_SHA
    actual = {
        str(p.relative_to(path))
        for p in path.rglob("*")
        if p.is_file() and p != path / "manifest.json"
    }
    assert actual == set(manifest["artifact_sha256"]), "payload file set changed"
    for name, digest in manifest["artifact_sha256"].items():
        artifact = (path / name).resolve()
        assert artifact.is_relative_to(path.resolve()) and sha256(artifact) == digest, name
    result: dict[str, Any] = dict(
        manifest=manifest,
        manifest_sha256=sha256(path / "manifest.json"),
        artifact_sha256=manifest["artifact_sha256"],
        exclusions=read_json(path / "exclusions.json"),
    )
    families: dict[str, set[str]] = {}
    keys: dict[str, set[str]] = {}
    for split in ("train", "validation", "diagnostic"):
        rows = read_jsonl(path / f"{split}.jsonl")
        assert rows, f"empty {split} teacher pool"
        for row in rows:
            board = board_from_record(row)
            assert row["semantic_key"] == source.board_key(board)
            target = np.asarray(row["teacher_policy"], dtype=np.float64)
            assert target.shape == (60,) and np.isfinite(target).all() and (target >= 0).all()
            assert np.isclose(target.sum(), 1) and all(
                board.is_legal_edge(int(e)) for e in np.flatnonzero(target)
            )
            assert row["exact_policy"] is False and row["exact_value"] is False
        families[split] = {f for r in rows for f in r["source_families"]}
        keys[split] = {r["semantic_key"] for r in rows}
        assert len(keys[split]) == len(rows), "duplicate teacher inputs"
        result[
            {
                "train": "train_records",
                "validation": "validation_records",
                "diagnostic": "diagnostic_records",
            }[split]
        ] = rows
    assert not families["train"] & (families["validation"] | families["diagnostic"])
    assert not keys["train"] & (keys["validation"] | keys["diagnostic"])
    assert not families["validation"] & families["diagnostic"]
    assert not keys["validation"] & keys["diagnostic"]
    return result


def filter_replay(
    dataset: Any, blocked_families: set[str], blocked_keys: set[str]
) -> tuple[Any, dict[str, int]]:
    """Filter original eligible rows, propagating excluded source families to descendants."""
    groups = {
        replay_group(g, dataset.geo)
        for g in dataset.games
        if source.game_family(g) in blocked_families
    }
    masks = []
    removed: Counter[str] = Counter()
    for game, indices in zip(dataset.games, dataset.position_indices, strict=True):
        mask: Any = np.zeros(len(game["boards"]), dtype=bool)
        if replay_group(game, dataset.geo) in groups:
            removed["family_rows"] += len(indices)
        else:
            for row in indices:
                if source.grid_key(game["boards"][row], int(game["to_play"][row])) in blocked_keys:
                    removed["state_rows"] += 1
                else:
                    mask[row] = True
        masks.append(mask)
    return subset(dataset, masks), dict(removed)


def load_replay(
    prepared: Path, payload: dict[str, Any], features: str = "chains", min_undrawn: int = 24
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Keep source splits stable while excluding teacher holdouts from every replay row."""
    frozen = bundle_files(prepared)
    dataset = load_arm(prepared, "exact", features, min_undrawn)
    original_train, original_validation = dataset.split_stable(0.1, 0)
    exclusions = payload["exclusions"]
    blocked = set(exclusions["heldout_families"] + exclusions["benchmark_families"])
    keys = set(exclusions["semantic_keys"])
    train, train_removed = filter_replay(original_train, blocked, keys)
    teacher_families = {f for r in payload["train_records"] for f in r["source_families"]}
    teacher_keys = {r["semantic_key"] for r in payload["train_records"]}
    validation, validation_removed = filter_replay(
        original_validation, blocked | teacher_families, keys | teacher_keys
    )
    validation, cleaned = clean_validation(train, validation)
    labels, _ = read_labels(prepared, dataset.geo)
    verify_root_targets(train, labels)
    verify_root_targets(validation, labels)
    assert coverage(train)["exact_roots"] and coverage(validation)["exact_roots"], (
        "exact anchors lost"
    )
    assert bundle_files(prepared) == frozen, "source replay changed"
    return (
        train,
        validation,
        labels,
        dict(
            training=coverage(train),
            validation=coverage(validation),
            train_removed=train_removed,
            validation_removed=validation_removed,
            validation_cleaning=cleaned,
            replay_sha256=frozen,
            teacher_validation_families=len(exclusions["heldout_families"]),
            scope=(
                "Original stable split, teacher-heldout and retained-benchmark exclusions; "
                "teacher-training family/state leakage removed from replay validation."
            ),
        ),
    )


def main() -> None:
    """Build a fresh portable screen payload without training or solver work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--collection", type=Path, action="append", required=True)
    parser.add_argument("--primary-collection", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(
        args.bank, args.collection, args.primary_collection, args.benchmark, args.output
    )
    print(json.dumps(result["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
