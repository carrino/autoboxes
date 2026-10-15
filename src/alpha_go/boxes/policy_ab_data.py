"""Freeze a matched old-search / exact-policy experiment from fresh training games.

Both arms share boards, exact values, masks, replay and collapsed value-only children.
Only complete exact-optimal policies replace selected source roots in the exact arm.
Every solve has a finite node budget; incomplete groups remain diagnostic CSV rows.
Known diagnostic families and canonical first-eight-move prefixes are excluded from
fresh games AND replay. This is not a claim to exclude all possible transpositions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np

from alpha_go.boxes.forced import BoxesSearchState
from alpha_go.boxes.policy_labels import LabelSettings, Mistake, scan_game, source_paths
from alpha_go.boxes.relabel import EXACT_UNKNOWN
from alpha_go.boxes.rules import BoxesBoard, BoxesGeometry, geometry
from alpha_go.boxes.symmetry import edge_permutation, transforms
from alpha_go.boxes.validation import replay_group


def sha256(path: Path) -> str:
    """Hash a frozen source or produced artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_game(path: Path) -> dict[str, Any]:
    """Load pickle-free game arrays without modifying their source."""
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def write_json(path: Path, value: Any) -> None:
    """Persist an atomic preparation status update."""
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_game(path: Path, game: dict[str, Any]) -> None:
    """Replace an NPZ atomically, keeping partial preparations readable."""
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **game)
    temporary.replace(path)


def canonical_opening(edges: list[int], geo: BoxesGeometry) -> str:
    """Canonical first eight ordered edges, independent of declared split namespaces."""
    if not edges:
        return ""
    selected = np.asarray(edges[:8], dtype=np.int64)
    assert np.all((selected >= 0) & (selected < geo.num_edges)), "invalid opening edge"
    return min(edge_permutation(geo, k)[selected].astype("<i4").tobytes()
               for k in transforms(geo.rows, geo.cols)).hex()


def game_opening(game: dict[str, Any], geo: BoxesGeometry) -> str:
    """Sequential opening identity, including any branched start prefix."""
    if str(game.get("termination", "")) == "solver":
        return ""
    starts = np.asarray(game.get("start_moves", []), dtype=np.int64).reshape(-1, 2)
    moves = np.asarray(game["moves"], dtype=np.int64).reshape(-1, 2)
    opening = np.concatenate([starts, moves])[:8]
    assert np.all(opening >= 0), "negative edge in sequential source opening"
    edges = [int(geo.lattice_edge[r, c]) for r, c in opening]
    return canonical_opening(edges, geo)


def excluded_keys(paths: list[Path], geo: BoxesGeometry
                  ) -> tuple[set[str], set[str], set[tuple[int, int, int]]]:
    """Read frozen probe roots: families, canonical openings and exact root states."""
    families: set[str] = set()
    openings: set[str] = set()
    states: set[tuple[int, int, int]] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            position = json.loads(line)
            assert (position["rows"], position["cols"]) == (geo.rows, geo.cols), (
                f"exclusion board dimensions mismatch: {path}")
            families.add(str(position["family"]))
            prefix = [int(e) for e in position["prefix"]]
            opening = canonical_opening(prefix, geo)
            if opening:
                openings.add(opening)
            board = BoxesBoard(geo.rows, geo.cols)
            for edge in prefix:
                assert board.play_edge(edge), "invalid exclusion prefix"
            states.add((board.edges, board.player(), round(board.score())))
    return families, openings, states


def filter_sources(paths: list[Path], replay_paths: list[Path], excluded: list[Path],
                   geo: BoxesGeometry) -> tuple[list[Path], list[Path], list[dict[str, Any]]]:
    """Exclude diagnostic lineage, propagate paired groups, and deduplicate file content."""
    families, openings, states = excluded_keys(excluded, geo)
    records: list[dict[str, Any]] = []
    for kind, inputs in [("fresh", paths), ("replay", replay_paths)]:
        for supplied in inputs:
            for candidate in [supplied, supplied.resolve()]:
                assert not any("5x5-mid" in part.lower() for part in candidate.parts), (
                    f"evaluation data must not be used for training: {candidate}")
        for path in sorted({p.resolve() for p in inputs}):
            assert path.is_file(), f"missing source: {path}"
            game = load_game(path)
            grids = np.asarray(game["boards"])
            assert grids.ndim == 3 and grids.shape[1:] == geo.lattice_shape, (
                f"source board dimensions mismatch: {path}")
            assert str(game.get("game", "boxes")) == "boxes", f"not Boxes: {path}"
            family = replay_group(game, geo).hex()
            opening = game_opening(game, geo)
            matches_state = False
            for grid, mover in zip(grids, game["to_play"]):
                mask = sum(1 << e for e, (r, c) in enumerate(geo.edge_rc) if grid[r, c] == 1)
                owners = grid[1::2, 1::2]
                score = int((owners == 1).sum() - (owners == 2).sum())
                matches_state |= (mask, int(mover), score) in states
            reason = ("diagnostic_family" if family in families else
                      "diagnostic_opening" if opening and opening in openings else
                      "diagnostic_root_state" if matches_state else "")
            records.append(dict(path=str(path), kind=kind, sha256=sha256(path), family=family,
                                canonical_opening=opening, excluded_reason=reason))
    blocked = {r["family"] for r in records if r["excluded_reason"]}
    seen: set[str] = set()
    for record in records:
        if record["family"] in blocked and not record["excluded_reason"]:
            record["excluded_reason"] = "paired_family"
        if not record["excluded_reason"] and record["sha256"] in seen:
            record["excluded_reason"] = "duplicate_content"
        if not record["excluded_reason"]:
            seen.add(record["sha256"])
    return ([Path(r["path"]) for r in records if r["kind"] == "fresh"
             and not r["excluded_reason"]],
            [Path(r["path"]) for r in records if r["kind"] == "replay"
             and not r["excluded_reason"]], records)


@dataclass
class ChildLabel:
    """A legal edge, exact final player-one margin, and its forced-collapse state."""

    edge: int
    raw: BoxesBoard
    collapsed: BoxesBoard
    margin_p1: int | None


def solve_children(mistake: Mistake, solver: Any, node_budget: int) -> list[ChildLabel]:
    """Retain every child value; unknown children prevent a partial optimal policy."""
    assert node_budget > 0
    children = []
    for edge in mistake.board.get_legal_moves_flat():
        raw = mistake.board.copy()
        assert raw.play_edge(edge)
        rest = solver.value_within(raw.edges, node_budget)
        final = None if rest is None else round(raw.score()) + (
            int(rest) if raw.player() == 0 else -int(rest))
        collapsed = BoxesSearchState.from_board(raw).board
        children.append(ChildLabel(edge, raw, collapsed, final))
    sign = 1 if mistake.board.player() == 0 else -1
    known = [sign * c.margin_p1 for c in children if c.margin_p1 is not None]
    assert all(value <= mistake.before for value in known), (
        "root/child solver mismatch")
    if len(known) == len(children):
        assert max(known) == mistake.before, (
            "root/child solver mismatch")
    return children


def solve_game(path: Path, settings: LabelSettings
               ) -> tuple[dict[str, Any], list[tuple[Mistake, list[ChildLabel]]], Counter[str]]:
    """Scan a private bounded table, selecting only searched learner roots."""
    game = load_game(path)
    if str(game.get("termination", "")) == "solver":
        return game, [], Counter({"static_fresh_collections": 1})
    solver = alpha_go_cpp.BoxesSolver(settings.rows, settings.cols, settings.table_entries)
    mistakes, counts = scan_game(game, settings, solver)
    selected = []
    policy_mask = np.asarray(game.get("policy_mask", np.ones(len(game["moves"]), bool)))
    assert policy_mask.dtype == np.bool_ and policy_mask.shape == (len(game["moves"]),), (
        "source policy_mask shape/type mismatch")
    visits = np.asarray(game.get("mcts_visits", np.zeros((len(game["moves"]),
                                                       geometry(settings.rows,
                                                                settings.cols).num_edges))))
    assert visits.shape == (len(game["moves"]), geometry(settings.rows,
                                                       settings.cols).num_edges)
    for mistake in mistakes:
        target = visits[mistake.index]
        assert np.all(np.isfinite(target)) and np.all(target >= 0), "invalid MCTS target"
        if not policy_mask[mistake.index] or target.sum() <= 0:
            counts["unsearched_roots_excluded"] += 1
            continue
        illegal = [(mistake.board.edges >> edge) & 1 != 0 for edge in range(len(target))]
        assert np.all(target[illegal] == 0), "MCTS target on illegal edges"
        children = solve_children(mistake, solver, settings.child_node_budget)
        selected.append((mistake, children))
        counts["policy_attempts"] += 1
        complete = all(c.margin_p1 is not None for c in children)
        counts["complete_policies" if complete else "incomplete_policies"] += 1
        if len(selected) == settings.max_per_game:
            break
    return game, selected, counts


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append a source's diagnostic rows after its immutable source hash is checked."""
    if not rows:
        return
    existed = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not existed:
            writer.writeheader()
        writer.writerows(rows)


def write_arms(output: Path, index: int, path: Path, source: dict[str, Any],
               labels: list[tuple[Mistake, list[ChildLabel]]], source_hash: str,
               geo: BoxesGeometry, *, write_control: bool = True) -> Counter[str]:
    """Write matched source copies and one shared, deduplicated child collection."""
    family = replay_group(source, geo).hex()
    shared = dict(source, source_replay_group=np.array(family),
                  source_path=np.array(str(path)), source_sha256=np.array(source_hash))
    margins = np.asarray(source.get("exact_margin_p1", np.full(len(source["boards"]),
                                                               EXACT_UNKNOWN, np.int16))).copy()
    assert margins.shape == (len(source["boards"]),), "source exact-label shape mismatch"
    exact_visits = np.asarray(source.get("mcts_visits", np.zeros((len(source["boards"]),
                                                                geo.num_edges)))).copy()
    root_mask = np.zeros(len(source["boards"]), dtype=bool)
    roots: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    children_by_state: dict[tuple[int, int, int], tuple[BoxesBoard, int]] = {}
    counts: Counter[str] = Counter()
    for mistake, children in labels:
        complete = all(c.margin_p1 is not None for c in children)
        sign = 1 if mistake.board.player() == 0 else -1
        optimal = [c.edge for c in children if c.margin_p1 is not None
                   and sign * c.margin_p1 == mistake.before]
        original = np.asarray(source["mcts_visits"][mistake.index], dtype=np.float64)
        prefix_rc = np.concatenate([
            np.asarray(source.get("start_moves", []), dtype=np.int64).reshape(-1, 2),
            np.asarray(source["moves"][:mistake.index], dtype=np.int64).reshape(-1, 2)])
        prefix = [int(geo.lattice_edge[r, c]) for r, c in prefix_rc]
        roots.append(dict(source=str(path), source_sha256=source_hash, source_index=index,
                          family=family, position=mistake.index, prefix=json.dumps(prefix),
                          edges=str(mistake.board.edges), mover=mistake.board.player(),
                          score_p1=round(mistake.board.score()),
                          exact_margin_p1=sign * mistake.before, regret=mistake.regret,
                          complete=complete, optimal_edges=json.dumps(optimal if complete else []),
                          original_visits=json.dumps(original.tolist())))
        if complete:
            assert optimal, "complete policy has no optimal edge"
            previous = int(margins[mistake.index])
            assert previous in (EXACT_UNKNOWN, sign * mistake.before), (
                "saved exact root label disagrees with solver")
            margins[mistake.index] = sign * mistake.before
            exact_visits[mistake.index] = 0
            exact_visits[mistake.index, optimal] = 1
            root_mask[mistake.index] = True
            counts["replaced_roots"] += 1
        for child in children:
            board = child.collapsed
            child_rows.append(dict(source=str(path), source_sha256=source_hash,
                                   source_index=index, family=family, position=mistake.index,
                                   edge=child.edge, raw_edges=str(child.raw.edges),
                                   raw_mover=child.raw.player(),
                                   raw_score_p1=round(child.raw.score()),
                                   collapsed_edges=str(board.edges), collapsed_mover=board.player(),
                                   collapsed_score_p1=round(board.score()),
                                   exact_margin_p1=child.margin_p1,
                                   group_complete=complete,
                                   optimal=complete and child.edge in optimal))
            if complete:
                assert child.margin_p1 is not None
                key = board.edges, board.player(), round(board.score())
                if key in children_by_state:
                    assert children_by_state[key][1] == child.margin_p1, "child label mismatch"
                    counts["duplicate_children"] += 1
                else:
                    children_by_state[key] = board, child.margin_p1
    if counts["replaced_roots"]:
        shared["exact_margin_p1"] = margins
    shared["policy_ab_root_mask"] = root_mask
    exact = dict(shared)
    if not write_control:
        # Explicit semantics for live-loop reporting; A/B control roots also have the
        # shared selection mask but their policies are still ordinary search targets.
        exact["exact_policy_root_mask"] = root_mask
    if counts["replaced_roots"]:
        exact["mcts_visits"] = exact_visits
    name = f"source-{index:07d}.npz"
    if write_control:
        save_game(output / "control" / name, shared)
    save_game(output / "exact" / name, exact)
    ordered = sorted(children_by_state.values(), key=lambda pair: (
        pair[0].move_count(), pair[0].edges, pair[0].player(), pair[1]))
    if ordered:
        n = len(ordered)
        children_game = dict(
            boards=np.stack([board.to_numpy() for board, _ in ordered]),
            to_play=np.asarray([board.player() for board, _ in ordered], np.int8),
            exact_margin_p1=np.asarray([margin for _, margin in ordered], np.int16),
            moves=np.full((n, 2), -1, np.int16), result=np.array("Draw"), num_moves=n,
            game=np.array("boxes"), termination=np.array("solver"),
            policy_mask=np.zeros(n, bool), train_mask=np.ones(n, bool),
            mcts_visits=np.zeros((n, geo.num_edges), np.int16),
            mcts_root_values=np.zeros(n, np.float32), source_replay_group=np.array(family),
            source_path=np.array(str(path)), source_sha256=np.array(source_hash))
        save_game(output / "children" / name, children_game)
        counts["child_collections"] += 1
        counts["child_positions"] += n
    append_rows(output / "roots.csv", roots)
    append_rows(output / "children.csv", child_rows)
    return counts


def prepare(paths: list[Path], output: Path, settings: LabelSettings, *,
            replay_paths: list[Path] | None = None, exclude_positions: list[Path] | None = None,
            workers: int = 1) -> dict[str, Any]:
    """Freeze and label matched data, preserving diagnostic lineage and source immutability."""
    settings.validate()
    assert paths and workers > 0
    assert not output.exists(), f"output must be fresh: {output}"
    output = output.resolve()
    geo = geometry(settings.rows, settings.cols)
    exclusions = exclude_positions or []
    fresh, replay, records = filter_sources(paths, replay_paths or [], exclusions, geo)
    assert fresh, "no fresh training games remain after diagnostic exclusions"
    output.mkdir(parents=True)
    for name in ["control", "exact", "children", "replay"]:
        (output / name).mkdir()
    counts: Counter[str] = Counter()
    counts.update({"input_files": len(records), "fresh_files": len(fresh),
                   "replay_files": len(replay),
                   "excluded_files": sum(bool(r["excluded_reason"]) for r in records)})
    manifests = {arm: f"{arm}-manifest.txt" for arm in ["control", "exact"]}
    summary: dict[str, Any] = dict(
        schema=1, status="running", settings=asdict(settings), workers=workers,
        preparer_sha256=sha256(Path(__file__)), sources=records,
        exclusions={str(path): sha256(path) for path in exclusions}, counts=dict(counts),
        manifest_paths=manifests,
        exclusion_scope=("family, canonical first eight edges, exact root states; "
                         "not all transpositions"),
        regret_units="final score-margin points")
    write_json(output / "preparation.json", summary)
    hashes = {r["path"]: r["sha256"] for r in records}
    for index, path in enumerate(replay):
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == hashes[str(path)], "replay source changed"
        (output / "replay" / f"replay-{index:07d}.npz").write_bytes(content)
    print(f"{len(fresh)} fresh games, {len(replay)} replay files, {counts['excluded_files']} "
          f"excluded; {workers} workers; {settings.table_entries * 16 / 1e6:.1f} MB/table; "
          f"line/child budgets {settings.node_budget:,}/{settings.child_node_budget:,}", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for begin in range(0, len(fresh), workers):
            batch = fresh[begin:begin + workers]
            futures = {pool.submit(solve_game, path, settings): (begin + j, path)
                       for j, path in enumerate(batch)}
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                print(f"solved {begin + len(batch) - len(pending)}/{len(fresh)} games; "
                      f"{counts['replaced_roots']} roots saved; "
                      f"{time.monotonic() - started:.0f}s", flush=True)
            for future, (index, path) in futures.items():
                game, labels, coverage = future.result()
                assert sha256(path) == hashes[str(path)], f"source changed during solve: {path}"
                counts.update(coverage)
                counts.update(write_arms(output, index, path, game, labels, hashes[str(path)], geo))
                counts["processed_fresh_files"] += 1
                summary["counts"] = dict(counts)
                summary["elapsed_seconds"] = round(time.monotonic() - started, 2)
                write_json(output / "preparation.json", summary)
    for arm in ["control", "exact"]:
        names = [arm]
        if counts["child_collections"]:
            names.append("children")
        if replay:
            names.append("replay")
        (output / manifests[arm]).write_text("".join(name + "\n" for name in names))
    summary["status"] = "complete"
    summary["artifact_sha256"] = {str(path.relative_to(output)): sha256(path)
                                  for path in sorted(output.rglob("*"))
                                  if path.is_file() and path.name != "preparation.json"}
    write_json(output / "preparation.json", summary)
    print(json.dumps(dict(status="complete", counts=dict(counts), manifest_paths=manifests),
                     indent=2), flush=True)
    return summary


def main() -> None:
    """Prepare new frozen data; this command neither trains nor changes source games."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, nargs="+", required=True)
    parser.add_argument("--replay", type=Path, nargs="+", default=[])
    parser.add_argument("--replay-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-positions", type=Path, action="append", default=[])
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get(
        "GAME_DATA_DIR", str(Path.home() / "autoboxes-data/game_data_root"))))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--min-undrawn", type=int, default=35)
    parser.add_argument("--max-undrawn", type=int, default=38)
    parser.add_argument("--node-budget", type=int, default=20_000_000)
    parser.add_argument("--child-node-budget", type=int, default=20_000_000)
    parser.add_argument("--table-entries", type=int, default=1 << 22)
    parser.add_argument("--max-per-game", type=int, default=2)
    parser.add_argument("--min-regret", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    settings = LabelSettings(args.rows, args.cols or args.rows, args.min_undrawn,
                             args.max_undrawn, args.node_budget, args.child_node_budget,
                             args.table_entries, args.max_per_game, args.min_regret)
    fresh = source_paths(args.games, [], args.data_root)
    replay = (source_paths(args.replay, args.replay_manifest, args.data_root)
              if args.replay or args.replay_manifest else [])
    prepare(fresh, args.output, settings, replay_paths=replay,
            exclude_positions=args.exclude_positions, workers=args.workers)


if __name__ == "__main__":
    main()
