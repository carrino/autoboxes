"""Bounded exact move targets from searched game histories for a backward curriculum.

NPZ trajectories and JSONL ``prefix`` histories are replayed independently. Opening
families are assigned before any root selection or solve. Each immutable output shard
can be resumed; completed source files are checked and reused, never relabelled. Only
complete all-child groups become policy targets. Roots alone are the default training
population; optional collapsed children carry value loss only.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np

from alpha_go.boxes.forced import collapse
from alpha_go.boxes.policy_ab_data import (
    canonical_opening,
    excluded_keys,
    load_game,
    save_game,
    sha256,
    solve_children,
    write_json,
)
from alpha_go.boxes.policy_labels import Mistake
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.validation import replay_group, validation_game


@dataclass(frozen=True)
class CurriculumSettings:
    """Finite exact-solve budgets and a permanent opening-family validation split."""

    rows: int = 5
    cols: int = 5
    min_undrawn: int = 31
    max_undrawn: int = 34
    node_budget: int = 2_000_000
    child_node_budget: int = 2_000_000
    table_entries: int = 1 << 20
    validation_fraction: float = 0.2
    split_seed: int = 20261016
    include_children: bool = False

    def validate(self) -> None:
        """Reject unbounded budgets and impossible board/split settings."""
        assert self.rows > 0 and self.cols > 0
        edges = geometry(self.rows, self.cols).num_edges
        assert edges <= 64 and 1 <= self.min_undrawn <= self.max_undrawn <= edges
        assert min(self.node_budget, self.child_node_budget, self.table_entries) > 0
        assert 0 < self.validation_fraction < 1 and self.split_seed >= 0


@dataclass
class Source:
    """A verified history with source identity assigned before selecting any roots."""

    path: str
    source_id: str
    source_sha256: str
    family: str
    opening: str
    prefix: list[int]
    split: str = ""


def state_key(board: BoxesBoard) -> tuple[int, int, int]:
    """Include the player and accumulated score when deduplicating edge masks."""
    return board.edges, board.player(), round(board.score())


def replay_prefix(prefix: list[int], rows: int, cols: int) -> BoxesBoard:
    """Replay legal edges, retaining captures, scores and extra turns."""
    board = BoxesBoard(rows, cols)
    for edge in prefix:
        assert type(edge) is int and board.is_legal_edge(edge), "illegal source prefix edge"
        assert board.play_edge(edge)
    return board


def load_sources(paths: list[Path], settings: CurriculumSettings) -> list[Source]:
    """Read NPZ trajectories or JSONL prefixes; supplied endpoint metadata must agree."""
    geo = geometry(settings.rows, settings.cols)
    sources = []
    for path in sorted({p.resolve() for p in paths}):
        assert path.is_file(), f"missing source: {path}"
        assert not any("5x5-mid" in part.lower() for part in path.parts), (
            "evaluation data must not be used for training")
        digest = sha256(path)
        if path.suffix == ".npz":
            game = load_game(path)
            assert str(game.get("game", "boxes")) == "boxes", "not a Boxes source"
            assert str(game.get("termination", "")) != "solver", "source must be sequential"
            board = BoxesBoard(settings.rows, settings.cols)
            prefix = []
            starts = np.asarray(game.get("start_moves", []), dtype=np.int64).reshape(-1, 2)
            for row, col in starts:
                edge = board.edge_index(int(row), int(col))
                assert board.play_edge(edge), "illegal branch prefix"
                prefix.append(edge)
            moves = np.asarray(game["moves"])
            assert len(moves) == len(game["boards"]) == len(game["to_play"])
            for index, (row, col) in enumerate(moves):
                assert np.array_equal(board.to_numpy(), game["boards"][index]), (
                    f"source board mismatch: {path}, row {index}")
                assert board.player() == int(game["to_play"][index]), "source mover mismatch"
                edge = board.edge_index(int(row), int(col))
                assert board.play_edge(edge), "illegal source move"
                prefix.append(edge)
            assert prefix, "empty source history"
            sources.append(Source(str(path), str(path), digest, replay_group(game, geo).hex(),
                                  canonical_opening(prefix, geo), prefix))
        else:
            assert path.suffix == ".jsonl", "sources must be .npz or .jsonl"
            for index, line in enumerate(path.read_text().splitlines()):
                if not line.strip():
                    continue
                row = json.loads(line)
                assert (row.get("rows", settings.rows), row.get("cols", settings.cols)) == (
                    settings.rows, settings.cols), "source dimensions mismatch"
                prefix = row["prefix"]
                assert isinstance(prefix, list) and prefix, "empty source history"
                board = replay_prefix(prefix, settings.rows, settings.cols)
                if "scores" in row:
                    assert list(board.boxes) == row["scores"], "source endpoint scores mismatch"
                if "to_play" in row:
                    assert board.to_play() == row["to_play"], "source endpoint mover mismatch"
                # The split is derived from play, never an arbitrary declared name.
                game = dict(moves=np.asarray([geo.edge_rc[e] for e in prefix], np.int16))
                sources.append(Source(str(path), f"{path}:{index + 1}", digest,
                                      replay_group(game, geo).hex(),
                                      canonical_opening(prefix, geo), prefix.copy()))
    assert sources, "no source histories"
    return sources


def partition_sources(sources: list[Source], settings: CurriculumSettings,
                      exclusions: list[Path]) -> tuple[list[Source], list[dict[str, Any]]]:
    """Unify inherited families with identical canonical openings before splitting."""
    geo = geometry(settings.rows, settings.cols)
    excluded_families, excluded_openings, excluded_states = excluded_keys(exclusions, geo)
    parents = {source.family: source.family for source in sources}

    def representative(family: str) -> str:
        while parents[family] != family:
            family = parents[family]
        return family

    openings: dict[str, str] = {}
    for source in sources:
        previous = openings.setdefault(source.opening, source.family)
        a, b = representative(source.family), representative(previous)
        parents[max(a, b)] = min(a, b)
    blocked = set()
    for source in sources:
        board = BoxesBoard(settings.rows, settings.cols)
        match = False
        for edge in source.prefix:
            match |= state_key(board) in excluded_states
            assert board.play_edge(edge)
        match |= state_key(board) in excluded_states
        if source.family in excluded_families or source.opening in excluded_openings or match:
            blocked.add(representative(source.family))
    selected, records = [], []
    for source in sources:
        source.family = representative(source.family)
        source.split = "validation" if validation_game(
            dict(source_replay_group=np.array(source.family)), geo,
            settings.validation_fraction, settings.split_seed) else "train"
        record = asdict(source)
        record["excluded"] = source.family in blocked
        records.append(record)
        if not record["excluded"]:
            selected.append(source)
    assert selected, "no sources remain after diagnostic exclusions"
    return selected, records


def candidate_roots(source: Source, settings: CurriculumSettings
                    ) -> list[tuple[int, BoxesBoard]]:
    """Select every decision root in the band, including a stored history's endpoint."""
    board = BoxesBoard(settings.rows, settings.cols)
    roots = []
    for index in range(len(source.prefix) + 1):
        remaining = board.num_edges() - board.move_count()
        if (settings.min_undrawn <= remaining <= settings.max_undrawn
                and not collapse(board).prefix):
            roots.append((index, board.copy()))
        if index < len(source.prefix):
            assert board.play_edge(source.prefix[index])
    return roots


def solve_roots(source: Source, roots: list[tuple[int, BoxesBoard]],
                settings: CurriculumSettings
                ) -> tuple[list[dict[str, Any]], list[Any], Counter[str]]:
    """Use a private table per history; partial child groups never produce targets."""
    solver = alpha_go_cpp.BoxesSolver(settings.rows, settings.cols, settings.table_entries)
    counts: Counter[str] = Counter(attempted_roots=len(roots))
    records, labelled = [], []
    for index, board in roots:
        rest = solver.value_within(board.edges, settings.node_budget)
        if rest is None:
            counts["unknown_roots"] += 1
            continue
        final = board.margin() + int(rest)
        mistake = Mistake(index, board, board.get_legal_moves_flat()[0], final, final)
        children = solve_children(mistake, solver, settings.child_node_budget)
        if any(child.margin_p1 is None for child in children):
            counts["incomplete_children"] += 1
            continue
        sign = 1 if board.player() == 0 else -1
        child_values = {}
        for child in children:
            assert child.margin_p1 is not None
            child_values[str(child.edge)] = child.margin_p1
        optimal = [child.edge for child in children
                   if sign * child_values[str(child.edge)] == final]
        assert optimal
        margin_p1 = sign * final
        record = dict(source=source.path, source_id=source.source_id,
                      source_sha256=source.source_sha256, family=source.family,
                      split=source.split, rows=settings.rows, cols=settings.cols,
                      prefix=source.prefix[:index], source_position=index,
                      undrawn=board.num_edges() - board.move_count(),
                      edges=str(board.edges), mover=board.player(), score_p1=round(board.score()),
                      exact_margin_p1=margin_p1, root_margin=final, optimal_edges=optimal,
                      child_margins={edge: sign * margin for edge, margin in child_values.items()},
                      child_margin_p1=child_values)
        records.append(record)
        labelled.append((board, margin_p1, optimal, children))
    counts["complete_roots"] = len(records)
    return records, labelled, counts


def label_arrays(source: Source, labels: list[Any], settings: CurriculumSettings) -> dict[str, Any]:
    """Pack exact roots and optional unique children with explicit supervision masks."""
    rows: dict[tuple[int, int, int], tuple[BoxesBoard, int, list[int]]] = {}
    for board, margin, optimal, _ in labels:
        rows[state_key(board)] = board, margin, optimal
    if settings.include_children:
        for _, _, _, children in labels:
            for child in children:
                key = state_key(child.collapsed)
                if key not in rows:
                    rows[key] = child.collapsed, int(child.margin_p1), []
                assert rows[key][1] == child.margin_p1, "duplicate exact child values disagree"
    ordered = sorted(rows.values(), key=lambda item: (item[0].move_count(), item[0].edges))
    geo = geometry(settings.rows, settings.cols)
    n = len(ordered)
    visits = np.zeros((n, geo.num_edges), np.int16)
    moves = np.full((n, 2), -1, np.int16)
    for index, (_, _, optimal) in enumerate(ordered):
        if optimal:
            visits[index, optimal] = 1
            moves[index] = geo.edge_rc[optimal[0]]
    policy_mask = np.asarray([bool(optimal) for _, _, optimal in ordered], bool)
    return dict(boards=np.stack([board.to_numpy() for board, _, _ in ordered]),
                to_play=np.asarray([board.player() for board, _, _ in ordered], np.int8),
                exact_margin_p1=np.asarray([margin for _, margin, _ in ordered], np.int16),
                moves=moves, result=np.array("Draw"), num_moves=n,
                game=np.array("boxes"), termination=np.array("solver"),
                policy_mask=policy_mask, train_mask=np.ones(n, bool),
                policy_ab_root_mask=policy_mask, exact_policy_root_mask=policy_mask,
                mcts_visits=visits,
                mcts_root_values=np.asarray([float((margin if board.player() == 0 else -margin) > 0)
                                            + 0.5 * float(margin == 0)
                                            for board, margin, _ in ordered], np.float32),
                source_replay_group=np.array(source.family), source_path=np.array(source.path),
                source_sha256=np.array(source.source_sha256), source_id=np.array(source.source_id))


def prepare(paths: list[Path], output: Path, settings: CurriculumSettings, *, workers: int = 1,
            exclude_positions: list[Path] | None = None,
            force_split: str | None = None) -> dict[str, Any]:
    """Create/resume an immutable shard; each completed source is independently persisted."""
    settings.validate()
    assert workers > 0
    assert force_split in (None, "train", "validation")
    output = output.resolve()
    exclusions = exclude_positions or []
    sources, source_records = partition_sources(load_sources(paths, settings), settings, exclusions)
    if force_split is not None:
        for source in sources:
            source.split = force_split
        for record in source_records:
            record["split"] = force_split
    candidates = [candidate_roots(source, settings) for source in sources]
    # Hold out exact root states as well as families, without using any solved labels.
    heldout = {state_key(board) for source, roots in zip(sources, candidates)
               if source.split == "validation" for _, board in roots}
    seen: set[tuple[int, int, int]] = set()
    for source, roots in zip(sources, candidates):
        kept = [(index, board) for index, board in roots if state_key(board) not in seen
                and (source.split == "validation" or state_key(board) not in heldout)]
        roots[:] = kept
        seen.update(state_key(board) for _, board in kept)
    recipe = dict(schema=1, settings=asdict(settings), force_split=force_split,
                  sources=source_records,
                  exclusions={str(path.resolve()): sha256(path) for path in exclusions},
                  labeler_sha256=sha256(Path(__file__)))
    output.mkdir(parents=True, exist_ok=True)
    recipe_path = output / "recipe.json"
    if recipe_path.exists():
        assert json.loads(recipe_path.read_text()) == recipe, "immutable label recipe changed"
    else:
        assert not list(output.iterdir()), "output exists without a label recipe"
        write_json(recipe_path, recipe)
    summary_path = output / "preparation.json"
    if summary_path.exists():
        completed_summary: dict[str, Any] = json.loads(summary_path.read_text())
        assert completed_summary["status"] == "complete"
        for name, digest in completed_summary["artifact_sha256"].items():
            assert sha256(output / name) == digest, f"completed label artifact changed: {name}"
        recorded = {name for name in completed_summary["artifact_sha256"] if name.endswith(".npz")}
        actual = {str(path.relative_to(output)) for path in output.rglob("*.npz")}
        assert recorded == actual, "unrecorded or missing completed label NPZ"
        return completed_summary
    for folder in ["train", "validation", "completed"]:
        (output / folder).mkdir(exist_ok=True)
    started = time.monotonic()
    results: dict[int, dict[str, Any]] = {}
    pending_indices = []
    for index, source in enumerate(sources):
        done_path = output / "completed" / f"source-{index:06d}.json"
        if done_path.exists():
            result = json.loads(done_path.read_text())
            for name, digest in result["artifacts"].items():
                assert sha256(output / name) == digest, f"completed label artifact changed: {name}"
            results[index] = result
        else:
            pending_indices.append(index)
    print(f"Labels: {len(sources)} sources; {len(results)} reused; {workers} workers; "
          f"{settings.table_entries * 16 / 1e6:.1f} MB/table; bounded root/child solves "
          f"{settings.node_budget:,}/{settings.child_node_budget:,} nodes", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for begin in range(0, len(pending_indices), workers):
            batch = pending_indices[begin:begin + workers]
            futures = {pool.submit(solve_roots, sources[index], candidates[index], settings): index
                       for index in batch}
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    index = futures[future]
                    source = sources[index]
                    records, labels, counts = future.result()
                    assert sha256(Path(source.path)) == source.source_sha256, (
                        "source changed during solve")
                    artifacts = {}
                    if labels:
                        name = f"{source.split}/source-{index:06d}.npz"
                        arrays = label_arrays(source, labels, settings)
                        save_game(output / name, arrays)
                        positions = {(sum(1 << edge for edge, (row, col) in
                                          enumerate(geometry(settings.rows, settings.cols).edge_rc)
                                          if grid[row, col] == 1), int(mover)):
                                     row_index for row_index, (grid, mover) in
                                     enumerate(zip(arrays["boards"], arrays["to_play"]))}
                        for record in records:
                            record["file"] = name
                            record["row_index"] = positions[int(record["edges"]), record["mover"]]
                        counts["saved_positions"] = len(arrays["boards"])
                        artifacts[name] = sha256(output / name)
                    result = dict(source=asdict(source), counts=dict(counts), roots=records,
                                  artifacts=artifacts)
                    write_json(output / "completed" / f"source-{index:06d}.json", result)
                    results[index] = result
                root_count = sum(r["counts"]["complete_roots"] for r in results.values())
                print(f"Labels ready {len(results)}/{len(sources)} sources; "
                      f"{root_count} exact roots; "
                      f"{time.monotonic() - started:.0f}s", flush=True)
    root_records = [root for index in sorted(results) for root in results[index]["roots"]]
    for split in ["train", "validation"]:
        target = output / f"{split}-positions.jsonl"
        selected = [{key: root[key] for key in
                     ["rows", "cols", "prefix", "source", "source_sha256", "family"]}
                    for root in root_records if root["split"] == split]
        target.write_text("".join(json.dumps(row) + "\n" for row in selected))
        (output / f"{split}-manifest.txt").write_text(split + "\n")
        source_rows = [dict(rows=settings.rows, cols=settings.cols, prefix=source.prefix,
                            source=source.path, source_sha256=source.source_sha256,
                            family=source.family) for source in sources if source.split == split]
        (output / f"{split}-sources.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in source_rows))
    (output / "roots.jsonl").write_text("".join(json.dumps(row) + "\n" for row in root_records))
    totals: Counter[str] = Counter()
    for result in results.values():
        totals.update(result["counts"])
    totals.update({f"{split}_roots": sum(root["split"] == split for root in root_records)
                   for split in ["train", "validation"]})
    summary = dict(schema=1, status="complete", settings=asdict(settings), counts=dict(totals),
                   source_families={split: len({s.family for s in sources if s.split == split})
                                    for split in ["train", "validation"]},
                   recipe_sha256=sha256(recipe_path), elapsed_seconds=time.monotonic() - started,
                   artifact_sha256={str(path.relative_to(output)): sha256(path)
                                    for path in sorted(output.rglob("*"))
                                    if path.is_file() and path.name != "preparation.json"})
    write_json(output / "preparation.json", summary)
    return summary


def main() -> None:
    """Prepare one bounded, resumable data shard from immutable histories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--min-undrawn", type=int, default=31)
    parser.add_argument("--max-undrawn", type=int, default=34)
    parser.add_argument("--node-budget", type=int, default=2_000_000)
    parser.add_argument("--child-node-budget", type=int, default=2_000_000)
    parser.add_argument("--table-entries", type=int, default=1 << 20)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20261016)
    parser.add_argument("--include-children", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=["train", "validation", "stable"], default="stable")
    parser.add_argument("--exclude-prepared", type=Path, nargs="*", default=[])
    parser.add_argument("--exclude-positions", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    paths = [path for root in args.sources for path in
             (sorted(root.rglob("*.npz")) if root.is_dir() else [root])]
    settings = CurriculumSettings(args.rows, args.cols or args.rows, args.min_undrawn,
                                  args.max_undrawn, args.node_budget, args.child_node_budget,
                                  args.table_entries, args.validation_fraction, args.split_seed,
                                  args.include_children)
    result = prepare(paths, args.output, settings, workers=args.workers,
                     exclude_positions=args.exclude_positions + [
                         path / "validation-sources.jsonl" for path in args.exclude_prepared],
                     force_split=None if args.split == "stable" else args.split)
    print("===RESULT===", flush=True)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
