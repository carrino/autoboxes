"""Export verified approximate Qdab advice and add matched targets to replay.

The portable payload contains source receipts and held-out state exclusions. It
never calls an engine or solver. Both fitting bundles retain identical outcomes,
exact anchors, masks and ordering; only explicit approximate teacher rows differ.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from alpha_go.boxes.balanced_policy import clean_validation, coverage, root_mask, root_subset
from alpha_go.boxes.forced import collapse
from alpha_go.boxes.policy_target_ab import bundle_files, load_arm, sha256
from alpha_go.boxes.qdab_benchmark import verify_game
from alpha_go.boxes.qdab_training_collect import (
    opening_family,
    teacher_roots,
    verify_teacher,
    verify_training_game,
)
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.symmetry import edge_permutation, transforms
from alpha_go.boxes.validation import replay_group

ROLES = ("learner-targets", "qdab-targets")
GEO = geometry(5, 5)
PERMS = [edge_permutation(GEO, k) for k in transforms(5, 5)]


def write_json(path: Path, value: Any) -> None:
    """Write finite portable metadata."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@lru_cache(maxsize=250000)
def canonical_mask(mask: int) -> int:
    edges = [edge for edge in range(60) if mask >> edge & 1]
    return min(sum(1 << int(perm[e]) for e in edges) for perm in PERMS)


def board_key(board: BoxesBoard) -> str:
    """D4 geometry and mover-relative scores, including color transpositions."""
    me = board.player()
    return f"{canonical_mask(board.edges)}:{board.boxes[me]}:{board.boxes[1 - me]}"


def grid_key(grid: Any, mover: int) -> str:
    mask = sum(1 << e for e, (r, c) in enumerate(GEO.edge_rc) if grid[r, c] == 1)
    owners = grid[1::2, 1::2]
    mine = int((owners == mover + 1).sum())
    theirs = int((owners == 2 - mover).sum())
    return f"{canonical_mask(mask)}:{mine}:{theirs}"


def game_family(game: dict[str, Any]) -> str | None:
    """Recover four-edge families from trajectories or inherited eight-edge lineage."""
    if "source_family" in game:
        return str(np.asarray(game["source_family"]).item())
    if "source_replay_group" in game:
        group = bytes.fromhex(str(np.asarray(game["source_replay_group"]).item()))
        if group.startswith(b"opening:"):
            edges = np.frombuffer(group[len(b"opening:") :], dtype="<i4")
            if len(edges) >= 4:
                return str(opening_family([list(map(int, GEO.edge_rc[e])) for e in edges[:4]]))
    if str(np.asarray(game.get("termination", "")).item()) == "solver":
        return None
    prefix = np.asarray(game.get("start_moves", []), dtype=np.int64).reshape(-1, 2)
    moves = np.asarray(game["moves"], dtype=np.int64).reshape(-1, 2)
    history = np.concatenate([prefix, moves])
    return opening_family(history[:4].tolist()) if len(history) >= 4 else None


def exclusion_states(row: dict[str, Any], children: bool = True) -> set[str]:
    """Freeze evaluation trajectories after their opening and all critical children."""
    board = BoxesBoard(5)
    keys: set[str] = set()
    for index, move in enumerate(row["moves"]):
        if index >= 4:
            keys.add(board_key(board))
        if children and 35 <= 60 - index <= 38:
            for edge in board.get_legal_moves_flat():
                child = board.copy()
                assert child.play_edge(edge)
                keys.add(board_key(child))
                keys.add(board_key(collapse(child).board))
        assert board.play_edge(board.edge_index(*move))
    keys.add(board_key(board))
    return keys


def verified_collection(
    collection: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Replay complete games and each recommendation against its frozen source root."""
    config = json.loads((collection / "config.json").read_text())
    summary = json.loads((collection / "summary.json").read_text())
    assert summary["games_completed"] == summary["games_planned"] == len(config["schedule"])
    assert summary["training_performed"] is False and summary["weights_changed"] is False
    paths = sorted((collection / "games").glob("*.json"))
    assert [p.name for p in paths] == [f"{s['index']:04d}.json" for s in config["schedule"]]
    games, teachers = [], {}
    for path, slot in zip(paths, config["schedule"], strict=True):
        game = json.loads(path.read_text())
        verify_training_game(game, slot)
        assert game["config_sha256"] == sha256(collection / "config.json")
        games.append(game)
        for root in teacher_roots(game):
            name = f"{slot['index']:04d}-{root['move_count']:02d}.json"
            saved = json.loads((collection / "teacher" / name).read_text())
            verify_teacher(saved, root, game, collection, config)
            teachers[name] = saved
    assert len(teachers) == summary["teacher_queries"]
    assert {p.name for p in (collection / "teacher").glob("*.json")} == set(teachers)
    families = [g["source_family"] for g in games]
    assert len(families) == len(set(families)), "duplicate source family"
    assert dict(Counter(g["split"] for g in games)) == summary["split_games"]
    return config, games, teachers


def export(collection: Path, benchmarks: list[Path], output: Path) -> dict[str, Any]:
    """Produce a self-contained, hash-checked payload for the WSL training machine."""
    assert not output.exists(), "payload output must be fresh"
    config, games, teachers = verified_collection(collection)
    assert benchmarks, "actual benchmark archives are required"
    source_checks = {}
    for name, digest in config["code_sha256"].items():
        assert sha256(Path(name)) == digest, f"collector source changed: {name}"
        source_checks[name] = digest
    keys: set[str] = set()
    benchmark_families: set[str] = set()
    benchmark_hashes = {}
    for benchmark in benchmarks:
        recipe = json.loads((benchmark / "config.json").read_text())
        benchmark_hashes[str(benchmark / "config.json")] = sha256(benchmark / "config.json")
        paths = sorted((benchmark / "games").glob("*.json"))
        assert len(paths) == len(recipe["schedule"]), "benchmark is incomplete"
        for path, slot in zip(paths, recipe["schedule"], strict=True):
            game = json.loads(path.read_text())
            verify_game(game, slot)
            assert game["config_sha256"] == sha256(benchmark / "config.json")
            keys.update(exclusion_states(game))
            if len(slot["opening"]) == 4:
                benchmark_families.add(opening_family(slot["opening"]))
            benchmark_hashes[str(path)] = sha256(path)
    assert not benchmark_families & {g["source_family"] for g in games}
    for game in games:
        if game["split"] == "validation":
            keys.update(exclusion_states(game))
    output.mkdir(parents=True)
    names = ["config.json", "summary.json"]
    names += [f"games/{g['slot']['index']:04d}.json" for g in games]
    names += [f"teacher/{name}" for name in sorted(teachers)]
    for name in names:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(collection / name, path)
    write_json(output / "exclusion-keys.json", sorted(keys))
    manifest = dict(
        schema=1,
        status="complete",
        checkpoint_sha256=config["checkpoint_sha256"],
        opening_families=sorted(g["source_family"] for g in games),
        training_families=sorted(g["source_family"] for g in games if g["split"] == "training"),
        validation_families=sorted(g["source_family"] for g in games if g["split"] == "validation"),
        benchmark_opening_families=sorted(benchmark_families),
        benchmark_sha256=benchmark_hashes,
        source_code_sha256=source_checks,
        exclusion_states=len(keys),
        teacher_queries=len(teachers),
        source_games=len(games),
        scope=(
            "Approximate teacher actions only; heldout families "
            "and actual benchmark states excluded"
        ),
        artifact_sha256={
            str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()
        },
    )
    write_json(output / "manifest.json", manifest)
    verify_payload(output)
    return manifest


def verify_payload(payload: Path) -> dict[str, Any]:
    """Reject missing, changed, unrecorded or escaping payload files."""
    manifest: dict[str, Any] = json.loads((payload / "manifest.json").read_text())
    assert manifest["schema"] == 1 and manifest["status"] == "complete"
    actual = {
        str(p.relative_to(payload))
        for p in payload.rglob("*")
        if p.is_file() and p != payload / "manifest.json"
    }
    assert actual == set(manifest["artifact_sha256"]), "payload file set changed"
    for name, digest in manifest["artifact_sha256"].items():
        path = (payload / name).resolve()
        assert path.is_relative_to(payload.resolve()) and sha256(path) == digest, name
    return manifest


def game_arrays(
    game: dict[str, Any], teachers: dict[str, dict[str, Any]], source_hash: str
) -> dict[str, Any]:
    """Preserve complete history, selecting only real learner searches for fitting."""
    board = BoxesBoard(5)
    grids, movers = [], []
    for move in game["moves"]:
        grids.append(board.to_numpy())
        movers.append(board.player())
        assert board.play_edge(board.edge_index(*move))
    visits: Any = np.zeros((60, 60), np.float32)
    selected: Any = np.zeros(60, bool)
    for decision in game["learner_decisions"]:
        row = decision["move_count"]
        visits[row] = decision["visits"]
        selected[row] = True
    marked: Any = np.zeros(60, bool)
    teacher_edges: Any = np.full(60, -1, np.int16)
    for decision in teacher_roots(game):
        row = decision["move_count"]
        teacher = teachers[f"{game['slot']['index']:04d}-{row:02d}.json"]
        marked[row], teacher_edges[row] = True, teacher["teacher_edge"]
    family = game["source_family"]
    return dict(
        boards=np.asarray(grids, np.int8),
        to_play=np.asarray(movers, np.int8),
        moves=np.asarray(game["moves"], np.int16),
        result=np.array(game["result"]),
        game=np.array("boxes"),
        termination=np.array("all_edges"),
        num_moves=np.array(60),
        source_family=np.array(family),
        source_sha256=np.array(source_hash),
        source_replay_group=np.array((b"qdab-source:" + family.encode()).hex()),
        source_split=np.array(game["split"]),
        train_mask=selected,
        mcts_visits=visits,
        learner_mcts_visits=visits.copy(),
        mcts_root_values=np.zeros(60, np.float32),
        root_value_semantics=np.array("unavailable_placeholder_q_mix_must_be_zero"),
        approximate_teacher_mask=marked,
        approximate_teacher_edge=teacher_edges,
        approximate_teacher_kind=np.array("approximate_qdab_action_not_exact"),
    )


def matched(a: Any, b: Any) -> dict[str, int]:
    """Permit policy-vector changes only at explicit approximate teacher positions."""
    assert len(a.games) == len(b.games) and len(a) == len(b)
    changed = teachers = 0
    for i, (old, new) in enumerate(zip(a.games, b.games, strict=True)):
        assert old.keys() == new.keys()
        assert replay_group(old, a.geo) == replay_group(new, b.geo)
        assert np.array_equal(a.position_indices[i], b.position_indices[i])
        for key in old:
            if key != "mcts_visits":
                assert np.array_equal(old[key], new[key]), f"arms differ in {key}"
        mask = np.asarray(old.get("approximate_teacher_mask", np.zeros(len(old["boards"]), bool)))
        assert mask.dtype == np.bool_ and mask.shape == (len(old["boards"]),)
        assert not np.any(mask & root_mask(old)), "teacher must not claim an exact policy"
        if "exact_policy_root_mask" in old:
            assert not np.any(mask & old["exact_policy_root_mask"])
        if "mcts_visits" in old:
            old_target, new_target = old["mcts_visits"], new["mcts_visits"]
            assert np.isfinite(old_target).all() and np.isfinite(new_target).all()
            assert np.all(old_target >= 0) and np.all(new_target >= 0)
            for row in np.flatnonzero(mask):
                edge = int(old["approximate_teacher_edge"][row])
                assert 0 <= edge < 60 and old["boards"][row][tuple(GEO.edge_rc[edge])] == 0
                expected: Any = np.zeros(60, np.float32)
                expected[edge] = 1
                assert np.array_equal(new_target[row], expected), (
                    "teacher target is not its one-hot action"
                )
                assert old_target[row].sum() > 0, "teacher control row lacks real search"
            old_target = old_target / np.maximum(old_target.sum(axis=1, keepdims=True), 1)
            new_target = new_target / np.maximum(new_target.sum(axis=1, keepdims=True), 1)
            differs = np.any(old_target != new_target, axis=1)
            assert not np.any(differs & ~mask), "target changed outside teacher rows"
            changed += int(differs[a.position_indices[i]].sum())
        teachers += int(mask[a.position_indices[i]].sum())
    return dict(
        positions=len(a),
        games=len(a.games),
        eligible_teacher_rows=teachers,
        changed_policy_rows=changed,
    )


def augment(
    payload: Path, replay: Path, output: Path, *, validation_band: int = 37
) -> dict[str, Any]:
    """Add fresh learner rows to verified replay without changing existing labels."""
    assert not output.exists(), "bundle output must be fresh"
    assert 0 < validation_band <= 60
    manifest = verify_payload(payload)
    config, games, teachers = verified_collection(payload)
    assert manifest["checkpoint_sha256"] == config["checkpoint_sha256"]
    original_hashes = bundle_files(replay)
    blocked_families = set(manifest["validation_families"] + manifest["benchmark_opening_families"])
    keys = set(json.loads((payload / "exclusion-keys.json").read_text()))
    output.mkdir(parents=True)
    removed: Counter[str] = Counter()
    source_games = {}
    for name in sorted(original_hashes):
        if name.endswith(".npz"):
            with np.load(replay / name, allow_pickle=False) as archive:
                source_games[name] = dict(archive)
    family_groups = {
        replay_group(g, GEO) for g in source_games.values() if game_family(g) in blocked_families
    }
    for name, game in source_games.items():
        old = np.asarray(game.get("train_mask", np.ones(len(game["boards"]), bool))).copy()
        selected = old.copy()
        family = replay_group(game, GEO)
        if family in family_groups:
            selected[:] = False
            removed["replay_family_rows"] += int(old.sum())
        else:
            for row in np.flatnonzero(selected):
                if grid_key(game["boards"][row], int(game["to_play"][row])) in keys:
                    selected[row] = False
                    removed["replay_state_rows"] += 1
        if not np.array_equal(old, selected):
            game["train_mask"] = selected
        removed["replay_exact_roots"] += int((root_mask(game) & old & ~selected).sum())
    new_games = []
    for game in games:
        if game["split"] != "training":
            continue
        arrays = game_arrays(
            game, teachers, sha256(payload / "games" / f"{game['slot']['index']:04d}.json")
        )
        for row in np.flatnonzero(arrays["train_mask"]):
            if grid_key(arrays["boards"][row], int(arrays["to_play"][row])) in keys:
                arrays["train_mask"][row] = False
                removed["teacher_game_state_rows"] += 1
        new_games.append((f"{game['slot']['index']:04d}.npz", arrays))
    assert new_games, "no training families"
    for role in ROLES:
        root = output / role
        root.mkdir()
        for name in sorted(original_hashes):
            if name == "preparation.json":
                continue
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name in source_games:
                np.savez_compressed(target, **source_games[name])
            else:
                shutil.copyfile(replay / name, target)
        (root / "qdab-teacher").mkdir()
        for name, arrays in new_games:
            values = dict(arrays)
            if role == "qdab-targets":
                values["mcts_visits"] = arrays["mcts_visits"].copy()
                for row in np.flatnonzero(arrays["approximate_teacher_mask"]):
                    values["mcts_visits"][row] = 0
                    values["mcts_visits"][row, arrays["approximate_teacher_edge"][row]] = 1
            np.savez_compressed(root / "qdab-teacher" / name, **values)
        text = (replay / "exact-manifest.txt").read_text().rstrip() + "\nqdab-teacher\n"
        for arm in ("control", "exact"):
            (root / f"{arm}-manifest.txt").write_text(text)
        metadata = dict(
            schema=1,
            status="complete",
            role=role,
            source_replay_sha256=original_hashes,
            payload_manifest_sha256=sha256(payload / "manifest.json"),
            checkpoint_sha256=config["checkpoint_sha256"],
            scope=(
                "Approximate Qdab policy targets; outcome/exact value targets identical; "
                "q_mix=0 required"
            ),
            artifact_sha256={
                str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()
            },
        )
        write_json(root / "preparation.json", metadata)
        bundle_files(root)
    datasets = [load_arm(output / role, "exact", "chains", 24) for role in ROLES]
    matching = matched(*datasets)
    train, validation = datasets[0].split_stable(0.1, 0)
    validation, validation_filter = clean_validation(train, validation)
    assert len(root_subset(validation, validation_band)), (
        f"no clean heldout {validation_band}-edge anchors"
    )
    counts = dict(training=coverage(train), validation=coverage(validation))
    assert counts["training"]["exact_roots"] and counts["validation"]["exact_roots"], (
        "exact anchors lost"
    )
    training_mask = sum(
        int(
            np.asarray(g.get("approximate_teacher_mask", np.zeros(len(g["boards"]), bool)))[
                ix
            ].sum()
        )
        for g, ix in zip(train.games, train.position_indices, strict=True)
    )
    assert training_mask, "no approximate teacher rows survive the trainer split"
    report = dict(
        schema=1,
        status="complete",
        checkpoint_sha256=config["checkpoint_sha256"],
        matching=matching,
        removed=dict(removed),
        coverage=counts,
        validation_filter=validation_filter,
        validation_band=validation_band,
        eligible_training_teacher_rows=training_mask,
        payload_manifest_sha256=sha256(payload / "manifest.json"),
        source_replay_manifest_sha256=original_hashes["preparation.json"],
        validation_families_excluded=len(manifest["validation_families"]),
        root_values=(
            "Unavailable for new sources; placeholder zeros never used by frontier_fit (q_mix=0)"
        ),
    )
    write_json(output / "report.json", report)
    assert bundle_files(replay) == original_hashes, "source replay changed"
    assert verify_payload(payload) == manifest, "payload changed"
    return report


def main() -> None:
    """Export cloud receipts or augment local replay using a portable payload."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--collection", type=Path, required=True)
    export_parser.add_argument("--benchmark", type=Path, action="append", required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    augment_parser = commands.add_parser("augment")
    for name in ("payload", "replay", "output"):
        augment_parser.add_argument("--" + name, type=Path, required=True)
    augment_parser.add_argument("--validation-band", type=int, default=37)
    args = parser.parse_args()
    result = (
        export(args.collection, args.benchmark, args.output)
        if args.command == "export"
        else augment(args.payload, args.replay, args.output, validation_band=args.validation_band)
    )
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("sha256")}, indent=2))


if __name__ == "__main__":
    main()
