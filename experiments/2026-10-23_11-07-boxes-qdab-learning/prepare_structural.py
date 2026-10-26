"""Freeze and exactly label fresh Qdab states for an offline structural-value pilot.

No NPZ/policy bundle or training is produced. A supervisor enforces a wall budget;
query receipts survive interruption. Source games, splits and frozen code stay unchanged.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import alpha_go_cpp as cpp  # type: ignore[import-not-found]
import numpy as np

from alpha_go.boxes import forced, policy_ab_data, qdab_training_collect, rules, symmetry
from alpha_go.boxes.forced import collapse
from alpha_go.boxes.policy_ab_data import sha256, write_json
from alpha_go.boxes.qdab_training_collect import verify_training_game
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.symmetry import edge_permutation, transforms

SCHEMA = 1
CUTS = (38, 37, 36, 35)
DEFAULT_POSITIONS = (
    "work/cloud-pipeline-20261023/delivery/positions.jsonl",
    "work/ordinary-four-upload-results/inputs/positions-35-36.jsonl",
    "work/ordinary-four-upload-results/inputs/positions-37.jsonl",
)
DEFAULT_GAMES = ("work/qdab-cloud-pilot-w895zcit/qdab-match-2g8p825p/games",)
STOP = False


def implementation_hashes() -> dict[str, str]:
    modules = (forced, policy_ab_data, qdab_training_collect, rules, symmetry)
    return dict(
        labeler=sha256(Path(__file__)),
        native_extension=sha256(Path(cpp.__file__)),
        **{module.__name__: sha256(Path(module.__file__)) for module in modules},
    )


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def digest_object(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def semantic_key(board: BoxesBoard) -> str:
    """D4 geometry plus mover-relative scores; physical color swaps also collide."""
    edges = [e for e in range(board.num_edges()) if board.edges >> e & 1]
    masks = [
        sum(1 << int(edge_permutation(board.geo, k)[e]) for e in edges)
        for k in transforms(board.rows(), board.cols())
    ]
    me = board.player()
    return f"{board.rows()}x{board.cols()}:{min(masks)}:{board.boxes[me]}:{board.boxes[1 - me]}"


def check_boards(python: BoxesBoard, native: Any) -> None:
    assert python.edges == int(native.edges()), "native edge mismatch"
    assert python.to_play() == int(native.to_play()), "native mover mismatch"
    assert int(python.margin()) == int(native.margin()), "native margin mismatch"
    assert np.array_equal(python.to_numpy(), native.to_numpy()), "native board mismatch"
    assert sum(python.boxes) == sum(v != 0 for v in python.owner)


def replay(prefix: list[int], rows: int = 5, cols: int = 5) -> tuple[BoxesBoard, Any]:
    python, native = BoxesBoard(rows, cols), cpp.BoxesBoard(rows, cols)
    for edge in prefix:
        assert type(edge) is int and python.is_legal_edge(edge), "illegal prefix"
        assert python.play_edge(edge) and native.play_edge(edge)
        check_boards(python, native)
    return python, native


def final_p1(board: BoxesBoard, remaining: int) -> int:
    value = int(board.score()) + int(remaining) * (1 if board.to_play() == 1 else -1)
    boxes = board.rows() * board.cols()
    assert abs(value) <= boxes and (value - boxes) % 2 == 0, "invalid exact final margin"
    assert abs(int(remaining)) <= boxes - sum(board.boxes), "invalid remaining margin"
    return value


def state_record(
    board: BoxesBoard,
    prefix: list[int],
    family: str,
    split: str,
    source_game: str,
    root_id: str,
    role: str,
) -> dict[str, Any]:
    # Keep query receipts source-private even if future paired games share a family.
    identity = [source_game, family, board.edges, board.to_play(), board.to_numpy().tobytes().hex()]
    return dict(
        state_id="s-" + digest_object(identity)[:24],
        rows=board.rows(),
        cols=board.cols(),
        prefix=prefix,
        edges=str(board.edges),
        to_play=board.to_play(),
        scores=list(board.boxes),
        score_p1=int(board.score()),
        exact_margin_p1=None,
        status="not_attempted",
        source_family=family,
        split=split,
        source_games=[source_game],
        root_groups=[root_id],
        roles=[role],
        canonical_d4_key=semantic_key(board),
        training_eligible=True,
        exclusion_reasons=[],
        query_id=f"q-{source_game}-{board.edges}",
        terminal=board.is_game_over(),
    )


def add_state(
    states: dict[str, dict[str, Any]],
    board: BoxesBoard,
    native: Any,
    prefix: list[int],
    family: str,
    split: str,
    game: str,
    root: str,
    role: str,
) -> str:
    check_boards(board, native)
    row = state_record(board, prefix, family, split, game, root, role)
    key = str(row["state_id"])
    if key in states:
        previous = states[key]
        assert (
            previous["split"] == split and previous["canonical_d4_key"] == row["canonical_d4_key"]
        )
        for name in ("root_groups", "source_games", "roles"):
            previous[name] = sorted(set(previous[name]) | set(row[name]))
    else:
        states[key] = row
    return key


def freeze_game(
    row: dict[str, Any],
    states: dict[str, dict[str, Any]],
    roots: list[dict[str, Any]],
    cuts: tuple[int, ...] = CUTS,
) -> None:
    """Independent full-game replay; expand every legal child before labels exist."""
    slot = row["slot"]
    verify_training_game(row, slot)
    geo = geometry(5, 5)
    moves = [int(geo.lattice_edge[r, c]) for r, c in row["moves"]]
    python, native = BoxesBoard(5, 5), cpp.BoxesBoard(5, 5)
    family, split = row["source_family"], row["split"]
    assert split in ("training", "validation")
    game = f"{slot['index']:04d}"
    for index, edge in enumerate(moves):
        check_boards(python, native)
        undrawn = 60 - index
        if undrawn in cuts:
            prefix = moves[:index]
            root_id = f"g{game}-u{undrawn}"
            state_id = add_state(
                states, python, native, prefix, family, split, game, root_id, "root"
            )
            children = []
            for action in python.get_legal_moves_flat():
                raw, raw_native = python.copy(), native.copy()
                assert raw.play_edge(action) and raw_native.play_edge(action)
                raw_prefix = prefix + [action]
                raw_id = add_state(
                    states, raw, raw_native, raw_prefix, family, split, game, root_id, "raw_child"
                )
                python_collapse = collapse(raw)
                native_collapse = cpp.BoxesSearchState(raw_native)
                forced = [int(e) for e in native_collapse.prefix()]
                collapsed = raw.copy()
                for capture in forced:
                    old_mover = collapsed.to_play()
                    assert collapsed.play_edge(capture)
                    assert collapsed.to_play() == old_mover, "forced prefix passed turn"
                check_boards(collapsed, native_collapse.board())
                assert np.array_equal(python_collapse.board.to_numpy(), collapsed.to_numpy())
                collapsed_id = add_state(
                    states,
                    collapsed,
                    native_collapse.board(),
                    raw_prefix + forced,
                    family,
                    split,
                    game,
                    root_id,
                    "collapsed_child",
                )
                children.append(
                    dict(
                        edge=action,
                        raw_state_id=raw_id,
                        collapsed_state_id=collapsed_id,
                        forced_prefix=forced,
                        status="not_attempted",
                        exact_margin_p1=None,
                    )
                )
            assert {c["edge"] for c in children} == set(python.get_legal_moves_flat())
            roots.append(
                dict(
                    root_id=root_id,
                    source_game=game,
                    source_family=family,
                    split=split,
                    prefix=prefix,
                    state_id=state_id,
                    rows=5,
                    cols=5,
                    to_play=python.to_play(),
                    played_edge=edge,
                    actor="learner"
                    if python.to_play() == (1 if slot["a_first"] else 2)
                    else "Qdab",
                    undrawn=undrawn,
                    children=children,
                    forced_prefix=[int(e) for e in cpp.BoxesSearchState(native).prefix()],
                    training_eligible=True,
                    exclusion_reasons=[],
                    status="not_attempted",
                    complete_children=False,
                    root_derived=False,
                    exact_margin_p1=None,
                )
            )
        assert python.play_edge(edge) and native.play_edge(edge)
    check_boards(python, native)
    assert python.is_game_over() and len(moves) == 60
    assert int(python.score()) * (1 if slot["a_first"] else -1) == row["a_margin"]


def exclusion_keys(prefix: list[int], expand_children: bool) -> set[str]:
    board, _ = replay(prefix)
    keys = {semantic_key(board)}
    if expand_children:
        for edge in board.get_legal_moves_flat():
            child = board.copy()
            assert child.play_edge(edge)
            keys.add(semantic_key(child))
            keys.add(semantic_key(collapse(child).board))
    return keys


def apply_exclusions(
    states: dict[str, dict[str, Any]], roots: list[dict[str, Any]], excluded: set[str]
) -> dict[str, Any]:
    validation = {s["canonical_d4_key"] for s in states.values() if s["split"] == "validation"}
    for state in states.values():
        reasons = []
        if state["canonical_d4_key"] in excluded:
            reasons.append("known_evaluation_or_development_state")
        if state["split"] == "training" and state["canonical_d4_key"] in validation:
            reasons.append("validation_semantic_transposition")
        state["training_eligible"] = not reasons
        state["exclusion_reasons"] = reasons
    for root in roots:
        referenced = [root["state_id"]] + [
            c[k] for c in root["children"] for k in ("raw_state_id", "collapsed_state_id")
        ]
        reasons = sorted(
            {reason for sid in referenced for reason in states[sid]["exclusion_reasons"]}
        )
        root["training_eligible"] = not reasons
        root["exclusion_reasons"] = reasons
    eligible_roots = {r["root_id"] for r in roots if r["training_eligible"]}
    for state in states.values():
        if not eligible_roots.intersection(state["root_groups"]):
            state["training_eligible"] = False
            state["exclusion_reasons"] = sorted(
                set(state["exclusion_reasons"]) | {"no_eligible_root_group"}
            )
    train = {
        s["canonical_d4_key"]
        for s in states.values()
        if s["split"] == "training" and s["training_eligible"]
    }
    val = {
        s["canonical_d4_key"]
        for s in states.values()
        if s["split"] == "validation" and s["training_eligible"]
    }
    assert not train & val
    return dict(
        semantic_key="D4 edge mask + mover score + opponent score; physical colors interchangeable",
        validation_wins=True,
        excluded_states=sum(not s["training_eligible"] for s in states.values()),
        excluded_roots=sum(not r["training_eligible"] for r in roots),
        eligible_train_validation_intersection=0,
    )


def active_collection(collection: Path) -> Any:
    """Hold a shared lock through freeze so timed collection cannot overlap labeling."""
    lock = (collection / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(
            "Qdab collection is still active; wait before preparing labels"
        ) from None
    return lock


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output, collection = args.output.resolve(), args.collection.resolve()
    code = implementation_hashes()
    # Immutable code/settings checks also apply to a resumed partial session.
    if (output / "recipe.json").exists():
        recipe: dict[str, Any] = json.loads((output / "recipe.json").read_text())
        assert recipe["code"] == code, "label implementation or native extension changed"
        assert recipe["settings"] == dict(
            node_budget=args.node_budget,
            workers=args.workers,
            table_entries=1 << 22,
            cuts=list(CUTS),
        ), "recipe changed"
        assert recipe["collection"] == str(collection), "collection changed"
        for name, digest in recipe["input_sha256"].items():
            assert sha256(output / name) == digest, f"frozen input changed: {name}"
        for name, digest in recipe["source_sha256"].items():
            assert sha256(collection / name) == digest, f"original source changed: {name}"
        frozen_config = json.loads((output / "inputs/config.json").read_text())
        assert sha256(Path(frozen_config["checkpoint"])) == frozen_config["checkpoint_sha256"]
        for name, digest in frozen_config["code_sha256"].items():
            assert sha256(Path(name)) == digest, f"collector implementation changed: {name}"
        assert sha256(output / "exclusion_receipt.json") == recipe["exclusion_receipt_sha256"]
        previous = output / "manifest.json"
        if previous.exists():
            for name, digest in json.loads(previous.read_text())["artifact_sha256"].items():
                assert sha256(output / name) == digest, f"completed artifact changed: {name}"
        return recipe
    assert not output.exists(), "output must be new or have an immutable recipe"
    config_path = collection / "config.json"
    config = json.loads(config_path.read_text())
    assert sha256(Path(config["checkpoint"])) == config["checkpoint_sha256"], "checkpoint changed"
    assert sha256(Path(cpp.__file__)) == config["runtime"]["extension_sha256"], (
        "collector native extension changed"
    )
    for name, digest in config["code_sha256"].items():
        assert sha256(Path(name)) == digest, f"collector implementation changed: {name}"
    games = sorted((collection / "games").glob("*.json"))
    assert games, "no completed fresh games"
    assert [p.stem for p in games] == [f"{i:04d}" for i in range(len(games))]
    output.mkdir(parents=True)
    for directory in ("inputs/games", "inputs/exclusions", "frozen", "receipts"):
        (output / directory).mkdir(parents=True)
    shutil.copyfile(config_path, output / "inputs/config.json")
    source_hashes = {"config.json": sha256(config_path)}
    states: dict[str, dict[str, Any]] = {}
    roots: list[dict[str, Any]] = []
    for path in games:
        row = json.loads(path.read_text())
        slot = config["schedule"][row["slot"]["index"]]
        assert row["slot"] == slot and row["config_sha256"] == source_hashes["config.json"]
        assert row["checkpoint_sha256"] == config["checkpoint_sha256"]
        source_hashes[f"games/{path.name}"] = sha256(path)
        shutil.copyfile(path, output / "inputs/games" / path.name)
        assert sha256(output / "inputs/games" / path.name) == source_hashes[f"games/{path.name}"]
        freeze_game(row, states, roots)
    positions = [
        Path(p) for p in ([] if args.no_default_exclusions else DEFAULT_POSITIONS)
    ] + args.exclude_positions
    game_dirs = [
        Path(p) for p in ([] if args.no_default_exclusions else DEFAULT_GAMES)
    ] + args.exclude_games
    excluded: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for path in positions:
        assert path.is_file(), f"missing required exclusion file: {path}"
        destination = output / "inputs/exclusions" / f"{len(receipts):03d}-positions.jsonl"
        shutil.copyfile(path, destination)
        count_before = len(excluded)
        for row in jsonl(path):
            assert (row.get("rows", 5), row.get("cols", 5)) == (5, 5)
            excluded.update(exclusion_keys(row["prefix"], True))
        receipts.append(
            dict(
                kind="development_roots_and_legal_children",
                source=str(path.resolve()),
                copied=str(destination.relative_to(output)),
                sha256=sha256(path),
                added_semantic_states=len(excluded) - count_before,
            )
        )
    geo = geometry(5, 5)
    for directory in game_dirs:
        files = sorted(directory.glob("*.json"))
        assert files, f"missing required exclusion games: {directory}"
        for path in files:
            row = json.loads(path.read_text())
            prefix = [int(geo.lattice_edge[r, c]) for r, c in row["moves"]]
            assert len(prefix) == 60
            replay(prefix)
            destination = output / "inputs/exclusions" / f"{len(receipts):03d}-game.json"
            shutil.copyfile(path, destination)
            count_before = len(excluded)
            for n in range(4, 61):
                excluded.update(exclusion_keys(prefix[:n], 60 - n in CUTS))
            receipts.append(
                dict(
                    kind="cloud_pilot_game_and_band_children",
                    source=str(path.resolve()),
                    copied=str(destination.relative_to(output)),
                    sha256=sha256(path),
                    added_semantic_states=len(excluded) - count_before,
                )
            )
    exclusion_report = apply_exclusions(states, roots, excluded)
    exclusion_report.update(
        inputs=receipts,
        known_exclusions_verified=bool(positions and game_dirs),
        overlap_pending_actual_gpu=True,
        exclusions_verified=False,
        scope="Only the listed inputs were checked. Actual GPU benchmark histories are pending.",
    )
    write_json(output / "exclusion_receipt.json", exclusion_report)
    save_jsonl(output / "frozen/states.jsonl", list(states.values()))
    save_jsonl(output / "frozen/roots.jsonl", roots)
    recipe = dict(
        schema=SCHEMA,
        collection=str(collection),
        code=code,
        settings=dict(
            node_budget=args.node_budget,
            workers=args.workers,
            table_entries=1 << 22,
            cuts=list(CUTS),
        ),
        checkpoint_sha256=config["checkpoint_sha256"],
        source_sha256=source_hashes,
        source_games=len(games),
        planned_source_games=len(config["schedule"]),
        split="Preserved collector canonical four-edge family split; no rehashing",
        exclusion_receipt_sha256=sha256(output / "exclusion_receipt.json"),
        input_sha256={
            str(p.relative_to(output)): sha256(p)
            for name in ("inputs", "frozen")
            for p in sorted((output / name).rglob("*"))
            if p.is_file()
        },
    )
    write_json(output / "recipe.json", recipe)
    return recipe


def task_order(
    roots: list[dict[str, Any]], states: dict[str, dict[str, Any]], phase: int
) -> dict[str, list[dict[str, Any]]]:
    """All root/played-child queries precede any other children, globally."""
    groups: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for root in sorted(roots, key=lambda r: (r["undrawn"], r["source_game"])):
        if not root["training_eligible"]:
            continue
        if phase == 1:
            ids = [
                root["state_id"],
                next(
                    c["raw_state_id"] for c in root["children"] if c["edge"] == root["played_edge"]
                ),
            ]
        else:
            ids = [c["raw_state_id"] for c in sorted(root["children"], key=lambda c: c["edge"])]
        for sid in ids:
            state = states[sid]
            if state["query_id"] not in seen:
                seen.add(state["query_id"])
                groups.setdefault(root["source_game"], []).append(state)
    return groups


def solve_group(
    output: Path, game: str, tasks: list[dict[str, Any]], settings: dict[str, Any], deadline: float
) -> int:
    """One private solver per game and phase; one atomic receipt per bounded query."""
    folder = output / "receipts" / game
    folder.mkdir(exist_ok=True)
    solver = None
    attempted = 0
    for task in tasks:
        path = folder / f"{task['query_id']}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            payload_hash = saved.pop("payload_sha256")
            assert digest_object(saved) == payload_hash, "query receipt changed"
            assert (
                saved["query_id"] == task["query_id"]
                and saved["node_budget"] == settings["node_budget"]
            )
            assert saved["edges"] == task["edges"] and saved["status"] in ("exact", "node_budget")
            continue
        if STOP or time.monotonic() >= deadline:
            break
        if solver is None:
            solver = cpp.BoxesSolver(task["rows"], task["cols"], settings["table_entries"])
        started = time.monotonic()
        remaining = solver.value_within(int(task["edges"]), settings["node_budget"])
        result = dict(
            query_id=task["query_id"],
            edges=task["edges"],
            node_budget=settings["node_budget"],
            remaining=None if remaining is None else int(remaining),
            status="node_budget" if remaining is None else "exact",
            nodes=int(solver.nodes()),
            seconds=time.monotonic() - started,
        )
        result["payload_sha256"] = digest_object(result)
        write_json(path, result)
        attempted += 1
    return attempted


def worker(output: Path, seconds: float, workers: int) -> None:
    global STOP
    signal.signal(signal.SIGTERM, lambda *_: setattr(sys.modules[__name__], "STOP", True))
    recipe = json.loads((output / "recipe.json").read_text())
    states = {s["state_id"]: s for s in jsonl(output / "frozen/states.jsonl")}
    roots = jsonl(output / "frozen/roots.jsonl")
    started = time.monotonic()
    locks = {r["source_game"]: threading.Lock() for r in roots}

    def locked_solve(game: str, tasks: list[dict[str, Any]], deadline: float) -> int:
        with locks[game]:
            return solve_group(output, game, tasks, recipe["settings"], deadline)

    for phase in (1, 2):
        # Root/played labels have global priority, but both phases share the
        # session deadline rather than abandoning unused time after phase one.
        deadline = started + seconds
        tasks = task_order(roots, states, phase)
        # Alternatives visit one root of every family before advancing the band.
        # Recreate private tables per chunk to keep at most four tables resident;
        # query receipts still reuse repeated masks. Locks prevent same-game races.
        chunks = (
            list(tasks.items())
            if phase == 1
            else [
                (root["source_game"], [states[c["raw_state_id"]] for c in root["children"]])
                for root in sorted(roots, key=lambda r: (r["undrawn"], r["source_game"]))
                if root["training_eligible"]
            ]
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(locked_solve, game, group, deadline) for game, group in chunks]
            for future in as_completed(futures):
                print(f"Phase {phase}: {future.result()} new query receipts", flush=True)
        if phase == 1:
            missing = [
                s
                for group in tasks.values()
                for s in group
                if not (
                    output / "receipts" / s["source_games"][0] / f"{s['query_id']}.json"
                ).exists()
            ]
            if missing:
                print(
                    f"Phase1 incomplete: {len(missing)} queries deferred; "
                    "alternatives wait for resume",
                    flush=True,
                )
                break
        if STOP:
            break


def set_exact(state: dict[str, Any], value: int, status: str, evidence: str) -> bool:
    boxes = state["rows"] * state["cols"]
    assert type(value) is int and abs(value) <= boxes and (value - boxes) % 2 == 0
    previous = state["exact_margin_p1"]
    assert previous is None or previous == value, "exact label disagreement"
    if previous is None:
        state.update(exact_margin_p1=value, status=status, label_evidence=evidence)
        return True
    return False


def reconcile(states: dict[str, dict[str, Any]], roots: list[dict[str, Any]]) -> None:
    """Complete-child backups and exact capture-collapse values, preserving direct status."""
    for _ in range(len(roots) + 2):
        changed = False
        for root in roots:
            values = []
            for child in root["children"]:
                raw, collapsed = states[child["raw_state_id"]], states[child["collapsed_state_id"]]
                value = raw["exact_margin_p1"]
                if value is not None:
                    changed |= set_exact(
                        collapsed, value, "exact_capture_collapse", raw["state_id"]
                    )
                child.update(exact_margin_p1=value, status=raw["status"])
                values.append(value)
            complete = len(root["children"]) == root["undrawn"] and all(
                v is not None for v in values
            )
            assert len({c["edge"] for c in root["children"]}) == root["undrawn"]
            root["complete_children"] = complete
            root_state = states[root["state_id"]]
            if complete:
                sign = 1 if root["to_play"] == 1 else -1
                derived = sign * max(sign * int(v) for v in values if v is not None)
                was_unknown = root_state["exact_margin_p1"] is None
                changed |= set_exact(root_state, derived, "exact_all_children", root["root_id"])
                root["root_derived"] = root["root_derived"] or was_unknown
                root["optimal_edges"] = [
                    c["edge"] for c in root["children"] if c["exact_margin_p1"] == derived
                ]
            root.update(
                exact_margin_p1=root_state["exact_margin_p1"],
                status=root_state["status"],
                direct_query_status=root_state.get("direct_query_status", "not_attempted"),
                known_children=sum(v is not None for v in values),
                legal_children=len(values),
            )
        if not changed:
            return
    raise AssertionError("label reconciliation did not converge")


def export(output: Path, session: dict[str, Any]) -> dict[str, Any]:
    recipe = json.loads((output / "recipe.json").read_text())
    states = {s["state_id"]: s for s in jsonl(output / "frozen/states.jsonl")}
    roots = jsonl(output / "frozen/roots.jsonl")
    receipt_paths = list((output / "receipts").glob("*/*.json"))
    assert len({p.stem for p in receipt_paths}) == len(receipt_paths), "duplicate query owners"
    receipts = {p.stem: json.loads(p.read_text()) for p in receipt_paths}
    for receipt in receipts.values():
        expected = receipt.pop("payload_sha256")
        assert digest_object(receipt) == expected, "query receipt changed"
    for state in states.values():
        query = receipts.get(state["query_id"])
        if query is not None:
            assert query["query_id"] == state["query_id"] and query["edges"] == state["edges"]
            assert query["node_budget"] == recipe["settings"]["node_budget"]
            state["direct_query_status"] = query["status"]
            state["status"] = query["status"]
            state["query_receipt"] = f"receipts/{state['source_games'][0]}/{state['query_id']}.json"
            if query["remaining"] is not None:
                board, _ = replay(state["prefix"], state["rows"], state["cols"])
                set_exact(
                    state, final_p1(board, query["remaining"]), "exact", state["query_receipt"]
                )
        elif state["training_eligible"]:
            state["status"] = "deadline" if session.get("timed_out") else "not_attempted"
        else:
            state["status"] = "excluded"
        if state["terminal"]:
            set_exact(state, state["score_p1"], "exact_terminal", "terminal_board")
    reconcile(states, roots)
    for root in roots:
        root["known_child_mask"] = [c["exact_margin_p1"] is not None for c in root["children"]]
    save_jsonl(output / "states.jsonl", list(states.values()))
    save_jsonl(output / "roots.jsonl", roots)
    required = {s["query_id"] for group in task_order(roots, states, 2).values() for s in group}
    required |= {s["query_id"] for group in task_order(roots, states, 1).values() for s in group}
    report = json.loads((output / "exclusion_receipt.json").read_text())
    assert sha256(output / "exclusion_receipt.json") == recipe["exclusion_receipt_sha256"]
    manifest = dict(
        schema=SCHEMA,
        status="complete" if required <= receipts.keys() else "partial",
        checkpoint_sha256=recipe["checkpoint_sha256"],
        source_games=recipe["source_games"],
        roots=len(roots),
        states=len(states),
        known_states=sum(s["exact_margin_p1"] is not None for s in states.values()),
        complete_root_groups=sum(r["complete_children"] for r in roots),
        eligible_root_groups=sum(r["training_eligible"] for r in roots),
        family_coverage={
            family: dict(
                split=next(r["split"] for r in roots if r["source_family"] == family),
                eligible_roots=sum(
                    r["training_eligible"] for r in roots if r["source_family"] == family
                ),
                complete_roots=sum(
                    r["training_eligible"] and r["complete_children"]
                    for r in roots
                    if r["source_family"] == family
                ),
                known_children=sum(
                    r["known_children"] for r in roots if r["source_family"] == family
                ),
            )
            for family in sorted({r["source_family"] for r in roots})
        },
        band_coverage={
            str(n): dict(
                roots=sum(r["undrawn"] == n for r in roots),
                complete_roots=sum(r["undrawn"] == n and r["complete_children"] for r in roots),
                known_roots=sum(
                    r["undrawn"] == n and r["exact_margin_p1"] is not None for r in roots
                ),
            )
            for n in CUTS
        },
        family_split={
            split: len({s["source_family"] for s in states.values() if s["split"] == split})
            for split in ("training", "validation")
        },
        development_only=True,
        known_exclusions_verified=report["known_exclusions_verified"],
        overlap_pending_actual_gpu=True,
        exclusions_verified=False,
        exclusion_receipt="exclusion_receipt.json",
        exclusion_receipt_sha256=recipe["exclusion_receipt_sha256"],
        session=session,
        recipe_sha256=sha256(output / "recipe.json"),
        artifact_sha256={
            str(path.relative_to(output)): sha256(path)
            for path in [
                output / name for name in ("states.jsonl", "roots.jsonl", "exclusion_receipt.json")
            ]
            + sorted((output / "receipts").glob("*/*.json"))
        },
        scope="Fresh training/development data only. Actual GPU benchmark overlap pending; "
        "no promotion, NPZ export, Qdab-exact labels, or policy substitution.",
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def stop_worker(process: subprocess.Popen[Any]) -> None:
    """A worker owns its process group; never leave its solver threads orphaned."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def supervise(
    output: Path,
    seconds: float,
    workers: int,
    freeze_only: bool = False,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Always stop the child and export receipts, including on SIGTERM/KeyboardInterrupt."""
    session: dict[str, Any] = dict(
        seconds_budget=seconds, timed_out=False, worker_returncode=None, interrupted=False
    )
    started = time.monotonic()
    process = None
    try:
        if not freeze_only:
            command = command or [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--output",
                str(output),
                "--worker",
                "--seconds",
                str(seconds),
                "--workers",
                str(workers),
            ]
            session["command"] = command
            session["command_sha256"] = digest_object(command)
            process = subprocess.Popen(command, start_new_session=True)
            try:
                session["worker_returncode"] = process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                session["timed_out"] = True
                stop_worker(process)
                session["worker_returncode"] = process.returncode
    except BaseException:
        session["interrupted"] = True
        raise
    finally:
        if process is not None:
            stop_worker(process)
            session["worker_returncode"] = process.returncode
        session["elapsed_seconds"] = time.monotonic() - started
        result = export(output, session)
        print(json.dumps(result, indent=2), flush=True)
    if session["worker_returncode"] not in (None, 0) and not session["timed_out"]:
        raise RuntimeError(
            f"Label worker failed with {session['worker_returncode']}; partial outputs retained"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2700)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--node-budget", type=int, default=20_000_000)
    parser.add_argument("--exclude-positions", type=Path, action="append", default=[])
    parser.add_argument("--exclude-games", type=Path, action="append", default=[])
    parser.add_argument(
        "--no-default-exclusions",
        action="store_true",
        help="Test/explicit-input use; empty exclusions stay unverified",
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert args.seconds > 0 and 1 <= args.workers <= 4 and args.node_budget > 0
    if args.worker:
        worker(args.output.resolve(), args.seconds, args.workers)
        return
    assert args.collection is not None, "--collection is required"

    def interrupted(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("SIGTERM: preserving partial exact-label receipts")

    signal.signal(signal.SIGTERM, interrupted)
    output = args.output.resolve()
    # The collection lock remains held until solving/export completes.
    lock = active_collection(args.collection.resolve())
    try:
        prepare(args)
        supervise(output, args.seconds, args.workers, args.freeze_only)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
