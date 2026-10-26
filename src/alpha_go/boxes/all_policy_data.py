"""Matched all-eligible versus mistake-only exact policies on fresh searched games.

Both arms contain the same full histories, exact values and collapsed child rows.
Control replaces complete policies only at played mistakes of at least --min-regret;
exact replaces every complete eligible policy. Unknown children never define negatives.
Frozen development families are excluded by canonical FIRST FOUR ordered edges, with
explicit replay lineage propagated across every supplied source. Inputs stay untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np

from alpha_go.boxes import policy_ab_data
from alpha_go.boxes.policy_ab_data import (
    ChildLabel,
    load_game,
    sha256,
    solve_children,
    write_arms,
    write_json,
)
from alpha_go.boxes.policy_labels import LabelSettings, Mistake, source_paths
from alpha_go.boxes.relabel import EXACT_UNKNOWN
from alpha_go.boxes.rules import BoxesBoard, BoxesGeometry, geometry
from alpha_go.boxes.symmetry import edge_permutation, transforms
from alpha_go.boxes.validation import replay_group


def canonical_opening4(edges: list[int], geo: BoxesGeometry) -> str:
    """Canonicalize exactly the first four available ordered edges under board symmetry."""
    selected = np.asarray(edges[:4], dtype=np.int64)
    assert len(selected) and np.all((selected >= 0) & (selected < geo.num_edges))
    return min(edge_permutation(geo, k)[selected].astype("<i4").tobytes()
               for k in transforms(geo.rows, geo.cols)).hex()


def replay_source(game: dict[str, Any], geo: BoxesGeometry
                  ) -> tuple[list[BoxesBoard], list[int]]:
    """Independently replay the complete saved history, rejecting malformed supervision."""
    assert str(game.get("game", "boxes")) == "boxes", "not a Boxes source"
    assert str(game.get("termination", "")) != "solver", "fresh histories required"
    moves = np.asarray(game["moves"])
    n = len(moves)
    assert n and moves.shape == (n, 2) and moves.dtype.kind in "iu", "invalid source moves"
    grids, movers = np.asarray(game["boards"]), np.asarray(game["to_play"])
    assert grids.shape == (n, *geo.lattice_shape), "source board dimensions mismatch"
    assert movers.shape == (n,) and movers.dtype.kind in "iu", "invalid source movers"
    starts = np.asarray(game.get("start_moves", np.empty((0, 2), np.int16)))
    assert starts.ndim == 2 and starts.shape[1] == 2 and starts.dtype.kind in "iu", (
        "invalid source branch prefix")
    board = BoxesBoard(geo.rows, geo.cols)
    history: list[int] = []
    for row, col in starts:
        edge = board.edge_index(int(row), int(col))
        assert board.play_edge(edge), "illegal source branch prefix"
        history.append(edge)
    states = []
    for index, (row, col) in enumerate(moves):
        assert board.player() == int(movers[index]), "source mover mismatch"
        assert np.array_equal(board.to_numpy(), grids[index]), "source board mismatch"
        states.append(board.copy())
        edge = board.edge_index(int(row), int(col))
        assert board.play_edge(edge), "illegal source move"
        history.append(edge)
    for name in ("train_mask", "policy_mask"):
        mask = np.asarray(game.get(name, np.ones(n, bool)))
        assert mask.shape == (n,) and mask.dtype == np.bool_, f"invalid {name}"
    visits = np.asarray(game.get("mcts_visits", np.zeros((n, geo.num_edges), np.int16)))
    arrays = ([visits, np.asarray(game["mcts_raw_visits"])]
              if "mcts_raw_visits" in game else [visits])
    for counts in arrays:
        assert counts.shape == (n, geo.num_edges), "invalid source visit shape"
        assert counts.dtype.kind in "iuf", "invalid source visit dtype"
        assert np.isfinite(counts).all() and (counts >= 0).all(), "invalid source visits"
        for state, row in zip(states, counts):
            legal = np.zeros(geo.num_edges, bool)
            legal[state.get_legal_moves_flat()] = True
            assert not row[~legal].any(), "source visits include illegal edges"
    if "exact_margin_p1" in game:
        labels = np.asarray(game["exact_margin_p1"])
        assert labels.shape == (n,) and labels.dtype.kind == "i", "invalid exact value array"
        assert np.all((labels == EXACT_UNKNOWN) | (np.abs(labels.astype(np.int64)) <=
                                                  geo.rows * geo.cols)), "invalid exact value"
    policy_mask = np.asarray(game.get("policy_mask", np.ones(n, bool)))
    if not policy_mask.all():
        assert "exact_margin_p1" in game, "value-only source row requires exact value"
        assert np.all(np.asarray(game["exact_margin_p1"])[~policy_mask] != EXACT_UNKNOWN), (
            "value-only source row requires exact value")
    return states, history


def filter_sources(paths: list[Path], exclusions: list[Path], geo: BoxesGeometry
                   ) -> tuple[list[Path], list[dict[str, Any]]]:
    """Exclude first-four opening families, exact roots and their declared lineage."""
    excluded_families: set[str] = set()
    excluded_openings: set[str] = set()
    excluded_states: set[tuple[int, int, int]] = set()
    for path in exclusions:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert (row["rows"], row["cols"]) == (geo.rows, geo.cols), "exclusion dimensions"
            prefix = row["prefix"]
            assert isinstance(prefix, list) and all(type(e) is int for e in prefix)
            board = BoxesBoard(geo.rows, geo.cols)
            for edge in prefix:
                assert board.play_edge(edge), "illegal exclusion prefix"
            assert len(prefix) >= 4, "frozen probe prefix must identify a four-edge family"
            excluded_openings.add(canonical_opening4(prefix, geo))
            excluded_families.add(str(row["family"]))
            excluded_states.add((board.edges, board.player(), round(board.score())))
    records = []
    for supplied in paths:
        assert not any("5x5-mid" in part.lower() for p in (supplied, supplied.resolve())
                       for part in p.parts), "evaluation input is not fresh training data"
    for path in sorted({p.resolve() for p in paths}):
        game = load_game(path)
        states, history = replay_source(game, geo)
        family = replay_group(game, geo).hex()
        opening = canonical_opening4(history, geo)
        same_state = any((b.edges, b.player(), round(b.score())) in excluded_states for b in states)
        reason = ("diagnostic_family" if family in excluded_families else
                  "diagnostic_opening4" if opening in excluded_openings else
                  "diagnostic_root_state" if same_state else "")
        records.append(dict(path=str(path), sha256=sha256(path), family=family,
                            canonical_opening4=opening, excluded_reason=reason))
    blocked = {r["family"] for r in records if r["excluded_reason"]}
    seen: set[str] = set()
    for row in records:
        if row["family"] in blocked and not row["excluded_reason"]:
            row["excluded_reason"] = "paired_family"
        if not row["excluded_reason"] and row["sha256"] in seen:
            row["excluded_reason"] = "duplicate_content"
        if not row["excluded_reason"]:
            seen.add(row["sha256"])
    return [Path(r["path"]) for r in records if not r["excluded_reason"]], records


class CachedSolver:
    """Cache each bounded mask/budget result, including unknown, within one source game."""

    def __init__(self, solver: Any) -> None:
        self.solver = solver
        self.values: dict[tuple[int, int], int | None] = {}

    def value_within(self, edges: int, budget: int) -> int | None:
        """The all-child validation pass reuses values without repeating bounded solves."""
        key = edges, budget
        if key not in self.values:
            value = self.solver.value_within(edges, budget)
            self.values[key] = None if value is None else int(value)
        return self.values[key]


def solve_source(path: Path, settings: LabelSettings, solver: Any = None
                 ) -> tuple[dict[str, Any], list[tuple[Mistake, list[ChildLabel]]],
                            Counter[str], list[dict[str, Any]]]:
    """Solve all searched band roots in chronological order, capped before inspecting labels."""
    settings.validate()
    geo = geometry(settings.rows, settings.cols)
    game = load_game(path)
    states, _ = replay_source(game, geo)
    n = len(states)
    game["train_mask"] = np.asarray(game.get("train_mask", np.ones(n, bool))).copy()
    visits = np.asarray(game.get("mcts_visits", np.zeros((n, geo.num_edges), np.int16))).copy()
    raw = np.asarray(game.get("mcts_raw_visits", visits))
    # Ordinary zero-visit forced moves retain the dataset's existing played-move fallback.
    # Search eligibility is separate: those rows must never acquire new exact policies.
    game["policy_mask"] = np.asarray(game.get("policy_mask", np.ones(n, bool))).copy()
    searched = (visits.sum(axis=1) > 0) & (raw.sum(axis=1) > 0)
    game["mcts_visits"] = visits
    margins = np.asarray(game.get("exact_margin_p1", np.full(n, EXACT_UNKNOWN, np.int16))).copy()
    saved = {(b.edges, b.player(), round(b.score())): int(v) for b, v in zip(states, margins)
             if v != EXACT_UNKNOWN}

    def check_saved(board: BoxesBoard, value: int | None) -> None:
        key = board.edges, board.player(), round(board.score())
        assert value is None or key not in saved or value == saved[key], (
            "saved exact value disagrees with solver")

    def final_p1(board: BoxesBoard, budget: int) -> int | None:
        rest = cache.value_within(board.edges, budget)
        return None if rest is None else round(board.score()) + (
            rest if board.player() == 0 else -rest)

    cache = CachedSolver(solver if solver is not None else alpha_go_cpp.BoxesSolver(
        settings.rows, settings.cols, settings.table_entries))
    counts: Counter[str] = Counter(source_train_rows=int(game["train_mask"].sum()),
                                  source_policy_rows=int((game["train_mask"] &
                                                          game["policy_mask"]).sum()))
    labels: list[tuple[Mistake, list[ChildLabel]]] = []
    diagnostics = []
    for index, board in enumerate(states):
        undrawn = geo.num_edges - board.move_count()
        if not settings.min_undrawn <= undrawn <= settings.max_undrawn:
            continue
        counts["band_roots"] += 1
        if not game["train_mask"][index]:
            counts["opponent_roots_excluded"] += 1
            continue
        if not game["policy_mask"][index] or not searched[index]:
            counts["unsearched_roots_excluded"] += 1
            continue
        counts["eligible_roots"] += 1
        if counts["policy_attempts"] >= settings.max_per_game:
            counts["capped_roots"] += 1
            continue
        counts["policy_attempts"] += 1
        root = final_p1(board, settings.node_budget)
        check_saved(board, root)
        sign = 1 if board.player() == 0 else -1
        child_values: dict[int, int | None] = {}
        for edge in board.get_legal_moves_flat():
            child = board.copy()
            assert child.play_edge(edge)
            child_values[edge] = final_p1(child, settings.child_node_budget)
            check_saved(child, child_values[edge])
        complete = all(v is not None for v in child_values.values())
        if complete:
            derived = sign * max(sign * int(v) for v in child_values.values() if v is not None)
            assert root is None or root == derived, "root/child solver mismatch"
            counts["recovered_roots"] += int(root is None)
            root = derived
            check_saved(board, root)
        edge = board.edge_index(*map(int, game["moves"][index]))
        played = child_values[edge]
        # The upper bound is only for checking partial child values. It is never saved as a label.
        before = sign * root if root is not None else geo.rows * geo.cols
        mistake = Mistake(index, board, edge, before, sign * played if played is not None else 0)
        children = solve_children(mistake, cache, settings.child_node_budget)
        for labelled in children:
            check_saved(labelled.collapsed, labelled.margin_p1)
        if root is not None:
            margins[index] = root
        regret = before - sign * played if root is not None and played is not None else None
        assert regret is None or regret >= 0, "negative exact regret"
        counts["known_roots"] += int(root is not None)
        counts["known_children"] += sum(v is not None for v in child_values.values())
        counts["children_attempted"] += len(child_values)
        counts["complete_policies" if complete else "incomplete_policies"] += 1
        diagnostic = dict(position=index, undrawn=undrawn, edges=str(board.edges),
                          root_margin_p1=root, played_edge=edge, played_regret=regret,
                          complete=complete, child_margin_p1=child_values,
                          original_visits=visits[index].tolist())
        diagnostics.append(diagnostic)
        if complete:
            assert root is not None and played is not None
            labels.append((mistake, children))
            optimal = [c.edge for c in children if c.margin_p1 == root]
            target = np.zeros(geo.num_edges, np.float64)
            target[optimal] = 1 / len(optimal)
            changed = not np.array_equal(visits[index] / visits[index].sum(), target)
            if mistake.regret >= settings.min_regret:
                visits[index] = 0
                visits[index, optimal] = 1
                counts["control_exact_policies"] += 1
                counts["shared_mistake_policy_changes"] += int(changed)
            else:
                counts["additional_exact_policies"] += 1
                counts["changed_policy_rows"] += int(changed)
    game["exact_margin_p1"] = margins
    return game, labels, counts, diagnostics


def artifact_hashes(folder: Path) -> dict[str, str]:
    """Hash the finalized contents of one immutable per-game shard."""
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != "receipt.json"}


def verify_shard(folder: Path, source_hash: str) -> dict[str, Any]:
    """Reject changed artifacts rather than silently mixing old and new preparation."""
    receipt: dict[str, Any] = json.loads((folder / "receipt.json").read_text())
    assert receipt["source_sha256"] == source_hash, "completed shard source changed"
    assert artifact_hashes(folder) == receipt["artifact_sha256"], "completed shard changed"
    return receipt


def prepare(paths: list[Path], output: Path, settings: LabelSettings, *,
            exclude_positions: list[Path] | None = None, workers: int = 4) -> dict[str, Any]:
    """Prepare resumable atomic source shards and portable, trainer-compatible manifests."""
    settings.validate()
    assert paths and workers > 0
    output = output.resolve()
    geo = geometry(settings.rows, settings.cols)
    exclusions = [p.resolve() for p in exclude_positions or []]
    fresh, records = filter_sources(paths, exclusions, geo)
    assert fresh, "no fresh games remain after exclusions"
    source_hashes = {r["path"]: r["sha256"] for r in records}
    recipe = dict(settings=asdict(settings), sources=records,
                  exclusions={str(p): sha256(p) for p in exclusions},
                  code={"all_policy_data": sha256(Path(__file__)),
                        "policy_ab_data": sha256(Path(policy_ab_data.__file__)),
                        "extension": sha256(Path(alpha_go_cpp.__file__))})
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "recipe.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == recipe, "preparation recipe changed"
    else:
        assert not any(output.iterdir()), "output must be empty or a matching preparation"
        write_json(frozen, recipe)
    metadata = output / "preparation.json"
    if metadata.exists():
        previous: dict[str, Any] = json.loads(metadata.read_text())
        if previous["status"] == "complete":
            actual = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                      if p.is_file() and p != metadata}
            assert actual == previous["artifact_sha256"], "completed preparation changed"
            print(f"Verified completed preparation: {output}", flush=True)
            return previous
    shards = output / "games"
    shards.mkdir(exist_ok=True)
    # A process killed during writing leaves only a disposable unpublished shard.
    for partial in shards.glob("*.partial"):
        assert partial.is_dir() and not partial.is_symlink()
        shutil.rmtree(partial)
    totals: Counter[str] = Counter(input_files=len(records), fresh_files=len(fresh),
                                  excluded_files=len(records) - len(fresh))
    summary: dict[str, Any] = dict(
        schema=1, status="running", settings=asdict(settings), workers=workers,
        sources=records, recipe_sha256=sha256(frozen), counts=dict(totals),
        manifest_paths={a: f"{a}-manifest.txt" for a in ("control", "exact")},
        exclusion_scope=("explicit family, canonical first four edges, exact roots; "
                         "not all transpositions"),
        selection="first eligible searched/train-mask roots in history, capped before solving",
        arms={"control": "complete exact policy when played regret >= min_regret",
              "exact": "complete exact policy at every eligible selected root"},
        child_values=("identical complete-group collapsed value rows; "
                      "incomplete groups diagnostic only"),
        regret_units="final score-margin points")
    write_json(output / "preparation.json", summary)
    pending_sources = []
    for index, path in enumerate(fresh):
        folder = shards / f"{index:07d}"
        if folder.exists():
            receipt = verify_shard(folder, source_hashes[str(path)])
            totals.update(receipt["counts"])
            totals["processed_fresh_files"] += 1
        else:
            pending_sources.append((index, path))
    print(f"{len(fresh)} fresh games; {totals['processed_fresh_files']} verified shards reused; "
          f"{workers} workers; {settings.table_entries * 16 / 1e6:.1f} MB/table; "
          f"budgets {settings.node_budget:,}/{settings.child_node_budget:,}", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for begin in range(0, len(pending_sources), workers):
            batch = pending_sources[begin:begin + workers]
            futures = {pool.submit(solve_source, p, settings): (i, p) for i, p in batch}
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    index, path = futures[future]
                    source, labels, counts, diagnostics = future.result()
                    assert sha256(path) == source_hashes[str(path)], "source changed during solve"
                    temporary = shards / f"{index:07d}.partial"
                    for name in ("control", "exact", "children"):
                        (temporary / name).mkdir(parents=True)
                    counts.update(write_arms(temporary, index, path, source, labels,
                                             source_hashes[str(path)], geo))
                    write_json(temporary / "diagnostics.json", diagnostics)
                    receipt = dict(source_sha256=source_hashes[str(path)], counts=dict(counts),
                                   artifact_sha256=artifact_hashes(temporary))
                    write_json(temporary / "receipt.json", receipt)
                    temporary.rename(shards / f"{index:07d}")
                    totals.update(counts)
                    totals["processed_fresh_files"] += 1
                summary.update(counts=dict(totals),
                               elapsed_seconds=round(time.monotonic() - started, 2))
                write_json(output / "preparation.json", summary)
                print(f"Prepared {totals['processed_fresh_files']}/{len(fresh)} games; "
                      f"{totals['complete_policies']} complete policies, "
                      f"{totals['additional_exact_policies']} beyond mistakes; "
                      f"{time.monotonic() - started:.0f}s", flush=True)
    for arm in ("control", "exact"):
        entries = []
        for index in range(len(fresh)):
            folder = shards / f"{index:07d}"
            entries.append(str((folder / arm).relative_to(output)))
            if list((folder / "children").glob("*.npz")):
                entries.append(str((folder / "children").relative_to(output)))
        (output / f"{arm}-manifest.txt").write_text("\n".join(entries) + "\n")
    summary.update(status="complete", counts=dict(totals))
    summary["artifact_sha256"] = {str(p.relative_to(output)): sha256(p)
                                  for p in sorted(output.rglob("*")) if p.is_file()
                                  and p.name != "preparation.json"}
    write_json(output / "preparation.json", summary)
    return summary


def main() -> None:
    """Prepare the frozen A/B bundle without training or changing source files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-positions", type=Path, action="append", default=[])
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--min-undrawn", type=int, default=35)
    parser.add_argument("--max-undrawn", type=int, default=36)
    parser.add_argument("--node-budget", type=int, default=20_000_000)
    parser.add_argument("--child-node-budget", type=int, default=20_000_000)
    parser.add_argument("--table-entries", type=int, default=1 << 22)
    parser.add_argument("--max-per-game", type=int, default=2)
    parser.add_argument("--min-regret", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    settings = LabelSettings(args.rows, args.cols or args.rows, args.min_undrawn,
                             args.max_undrawn, args.node_budget, args.child_node_budget,
                             args.table_entries, args.max_per_game, args.min_regret)
    paths = source_paths(args.games, [], Path(os.environ.get("GAME_DATA_DIR", ".")))
    result = prepare(paths, args.output, settings, exclude_positions=args.exclude_positions,
                     workers=args.workers)
    print(json.dumps(dict(status=result["status"], counts=result["counts"])), flush=True)


if __name__ == "__main__":
    main()
