"""Exact policies at consequential mistakes in training games, without modifying sources.

Run ``python -m alpha_go.boxes.policy_labels --games DIR --output NEW_DIR``. Manifests
use the training script's convention: relative entries are under GAME_DATA_DIR. Only
sequential games are scanned; existing solver-position collections are reported/skipped.
The known evaluation collection ``5x5-mid`` is rejected, including resolved symlink paths.
Other inputs must be training data: file names cannot establish their provenance.

Each game's bounded line solves share a transposition table. Winning-to-losing mistakes
rank first, then exact margin regret; only --max-per-game roots receive all-child solves.
A policy is saved only when every legal child is solved, never from a partial best set.
Regret is in FINAL SCORE-MARGIN POINTS: moving one box between players changes it by two.
"""
from __future__ import annotations

import argparse
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

from alpha_go.boxes.dataset import save_labelled_positions
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.validation import replay_group


@dataclass(frozen=True)
class LabelSettings:
    """Finite budgets and selection criteria for one board size."""

    rows: int = 5
    cols: int = 5
    min_undrawn: int = 35
    max_undrawn: int = 38
    node_budget: int = 20_000_000
    child_node_budget: int = 20_000_000
    table_entries: int = 1 << 22
    max_per_game: int = 2
    min_regret: int = 2

    def validate(self) -> None:
        """Reject unbounded solver budgets and impossible bands."""
        assert self.rows > 0 and self.cols > 0
        edges = geometry(self.rows, self.cols).num_edges
        assert edges <= 64, "the C++ exact solver supports at most 64 edges"
        assert 1 <= self.min_undrawn <= self.max_undrawn <= edges
        assert min(self.node_budget, self.child_node_budget, self.table_entries,
                   self.max_per_game, self.min_regret) > 0


@dataclass
class Mistake:
    """One move whose exact successor is worse for its mover."""

    index: int
    board: BoxesBoard
    played_edge: int
    before: int
    after: int

    @property
    def regret(self) -> int:
        return self.before - self.after

    @property
    def win_to_loss(self) -> bool:
        return self.before > 0 and self.after < 0


def scan_game(game: dict[str, Any], settings: LabelSettings, solver: Any
              ) -> tuple[list[Mistake], Counter[str]]:
    """Validate/replay one game, finding mistakes using bounded adjacent exact values.

    ``before`` and ``after`` share the ORIGINAL mover's perspective, including captures
    that retain the turn. The successor of the final stored move is solved too, even if
    collection stopped early at the solver threshold.
    """
    board = BoxesBoard(settings.rows, settings.cols)
    counts: Counter[str] = Counter()
    for row, col in np.asarray(game.get("start_moves", []), dtype=np.int64).reshape(-1, 2):
        assert board.play(int(row), int(col)), "illegal source branch prefix"
    moves = np.asarray(game["moves"])
    assert len(moves) == len(game["boards"]) == len(game["to_play"])
    train_mask = np.asarray(game.get("train_mask", np.ones(len(moves), dtype=bool)), dtype=bool)
    assert train_mask.shape == (len(moves),), "source train_mask shape mismatch"
    values: dict[int, int | None] = {}

    def final_p1(position: BoxesBoard) -> int | None:
        if position.edges not in values:
            value = solver.value_within(position.edges, settings.node_budget)
            values[position.edges] = (None if value is None else round(position.score())
                                     + (int(value) if position.player() == 0 else -int(value)))
        return values[position.edges]

    mistakes = []
    for index, (row, col) in enumerate(moves):
        assert board.player() == int(game["to_play"][index]), "source mover mismatch"
        assert np.array_equal(board.to_numpy(), game["boards"][index]), "source board mismatch"
        edge = board.edge_index(int(row), int(col))
        assert board.is_legal_edge(edge), "illegal source move"
        undrawn = board.num_edges() - board.move_count()
        before = board.copy()
        assert board.play_edge(edge)
        if not settings.min_undrawn <= undrawn <= settings.max_undrawn:
            continue
        counts["band_moves"] += 1
        if not train_mask[index]:
            counts["opponent_moves_excluded"] += 1
            continue
        first, second = final_p1(before), final_p1(board)
        if first is None or second is None:
            counts["unknown_adjacent"] += 1
            continue
        counts["known_adjacent"] += 1
        sign = 1 if before.player() == 0 else -1
        mistake = Mistake(index, before, edge, sign * first, sign * second)
        assert mistake.regret >= 0, "exact move improved on minimax value"
        if mistake.regret >= settings.min_regret:
            mistakes.append(mistake)
            counts["qualifying_mistakes"] += 1
            counts["win_to_loss"] += int(mistake.win_to_loss)
    mistakes.sort(key=lambda m: (-int(m.win_to_loss), -m.regret, m.index))
    return mistakes, counts


def exact_policy(mistake: Mistake, solver: Any, node_budget: int) -> list[int] | None:
    """Return ALL optimal legal edges, or None if any child exceeds its finite budget."""
    assert node_budget > 0
    board = mistake.board
    values = []
    for edge in board.get_legal_moves_flat():
        child = board.copy()
        assert child.play_edge(edge)
        rest = solver.value_within(child.edges, node_budget)
        if rest is None:
            return None
        margin = child.margin() + int(rest)
        values.append((edge, margin if child.player() == board.player() else -margin))
    assert max(v for _, v in values) == mistake.before, "root/child solver mismatch"
    return [edge for edge, value in values if value == mistake.before]


def label_game(path: Path, settings: LabelSettings
               ) -> tuple[dict[str, Any], list[tuple[Mistake, list[int]]], Counter[str]]:
    """Solve one source with its own table, so worker scheduling cannot change coverage."""
    with np.load(path, allow_pickle=False) as data:
        game = dict(data)
    assert str(game.get("game", "boxes")) == "boxes", f"not a Boxes game: {path}"
    if str(game.get("termination", "")) == "solver":
        return game, [], Counter({"static_collections": 1})
    solver = alpha_go_cpp.BoxesSolver(settings.rows, settings.cols, settings.table_entries)
    mistakes, counts = scan_game(game, settings, solver)
    labels = []
    for mistake in mistakes[:settings.max_per_game]:
        counts["policy_attempts"] += 1
        optimal = exact_policy(mistake, solver, settings.child_node_budget)
        if optimal is None:
            counts["incomplete_policies"] += 1
        else:
            labels.append((mistake, optimal))
    counts["complete_policies"] = len(labels)
    return game, labels, counts


def save_policy(path: Path, source: dict[str, Any], source_path: Path, source_sha256: str,
                mistake: Mistake, optimal: list[int]) -> None:
    """Write one exact training position with its original replay-family assignment."""
    board = mistake.board
    margin_p1 = mistake.before if board.player() == 0 else -mistake.before
    temporary = path.with_suffix(".tmp.npz")
    save_labelled_positions(temporary, board.to_numpy()[None],
                            np.array([board.player()], np.int8), [optimal], margin_p1, board.geo)
    with np.load(temporary, allow_pickle=False) as data:
        game = dict(data)
    game.update(exact_margin_p1=np.array([margin_p1], np.int16),
                source_replay_group=np.array(replay_group(source, board.geo).hex()),
                source_path=np.array(str(source_path)), source_sha256=np.array(source_sha256),
                source_position=np.array(mistake.index), played_edge=np.array(mistake.played_edge),
                regret_margin_points=np.array(mistake.regret),
                win_to_loss=np.array(mistake.win_to_loss))
    np.savez_compressed(temporary, **game)
    temporary.replace(path)


def source_paths(directories: list[Path], manifests: list[Path], data_root: Path) -> list[Path]:
    """Resolve training manifests, deduplicate files and reject known evaluation inputs."""
    roots = list(directories)
    for manifest in manifests:
        roots.extend(data_root / line.strip() for line in manifest.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith("#"))
    assert roots, "supply --games or --manifest"
    paths: set[Path] = set()
    for root in roots:
        assert root.is_dir(), f"missing training directory: {root}"
        for path in [root, root.resolve()]:
            assert not any("5x5-mid" in part.lower() for part in path.parts), (
                f"evaluation data must not be used for training labels: {path}")
        files = [p for p in sorted(root.rglob("*.npz")) if not p.name.endswith(".tmp.npz")]
        assert files, f"no training NPZ files under {root}"
        for path in files:
            for candidate in [path, path.resolve()]:
                assert not any("5x5-mid" in part.lower() for part in candidate.parts), (
                    f"evaluation data must not be used for training labels: {candidate}")
            paths.add(path.resolve())
    return sorted(paths)


def run(paths: list[Path], output: Path, settings: LabelSettings, workers: int = 1
        ) -> dict[str, Any]:
    """Create a fresh labelled collection, deterministic across worker completion order."""
    settings.validate()
    assert workers > 0 and paths
    assert not output.exists(), f"output must be fresh: {output}"
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output.mkdir(parents=True)
    settings_json = dict(asdict(settings), workers=workers, sources=hashes,
                         labeler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         regret_units="final score-margin points", status="running")
    (output / "settings.json").write_text(json.dumps(settings_json, indent=2) + "\n")
    print(f"{len(paths)} sources; {workers} workers, solver table "
          f"{settings.table_entries * 16 / 1e6:.1f} MB/worker; bounded line/child solves "
          f"{settings.node_budget:,}/{settings.child_node_budget:,} nodes", flush=True)
    started = time.monotonic()
    reported, last_report = 0, started
    totals: Counter[str] = Counter()
    seen: set[tuple[int, int, int]] = set()
    # Results are emitted in source order. Only at most `workers` games are in flight;
    # an exception cannot leave a whole dataset queued for CPU work on executor shutdown.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for begin in range(0, len(paths), workers):
            batch = paths[begin:begin + workers]
            futures = {pool.submit(label_game, path, settings): path for path in batch}
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()  # Surface failures immediately, before writing this batch.
                completed = begin + len(batch) - len(pending)
                now = time.monotonic()
                if completed - reported >= 5 or now - last_report >= 30 or completed == len(paths):
                    print(f"scanned {completed}/{len(paths)} games; "
                          f"{totals['saved']} policies saved; {now - started:.0f}s", flush=True)
                    reported, last_report = completed, now
            for future, path in futures.items():
                source, labels, counts = future.result()
                assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)], (
                    f"source changed during labelling: {path}")
                totals.update(counts)
                for mistake, optimal in labels:
                    board = mistake.board
                    key = (board.edges, board.player(), round(board.score()))
                    if key in seen:
                        totals["duplicates"] += 1
                        continue
                    seen.add(key)
                    destination = output / f"policy-{totals['saved']:07d}.npz"
                    save_policy(destination, source, path, hashes[str(path)], mistake, optimal)
                    totals["saved"] += 1
                with (output / "coverage.jsonl").open("a") as report:
                    report.write(json.dumps(dict(source=str(path), **counts)) + "\n")
    summary = dict(status="complete", source_files=len(paths), **totals,
                   elapsed_seconds=round(time.monotonic() - started, 2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    settings_json["status"] = "complete"
    (output / "settings.json").write_text(json.dumps(settings_json, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    """Command-line entry point; generated NPZs can be added directly to EXTRA_DATA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, nargs="+", default=[])
    parser.add_argument("--manifest", type=Path, action="append", default=[])
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
    parser.add_argument("--min-regret", type=int, default=2,
                        help="minimum final score-margin points lost (not boxes captured)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    settings = LabelSettings(args.rows, args.cols or args.rows, args.min_undrawn,
                             args.max_undrawn, args.node_budget, args.child_node_budget,
                             args.table_entries, args.max_per_game, args.min_regret)
    paths = source_paths(args.games, args.manifest, args.data_root)
    run(paths, args.output, settings, args.workers)


if __name__ == "__main__":
    main()
