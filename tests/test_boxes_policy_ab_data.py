"""Frozen policy-only A/B preparation keeps exact values, lineage and source data matched."""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import pytest

from alpha_go.boxes.agents import BoxesRandomAgent
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.policy_ab_data import (
    canonical_opening,
    filter_sources,
    load_game,
    prepare,
    sha256,
    solve_children,
    solve_game,
    write_arms,
)
from alpha_go.boxes.policy_labels import LabelSettings, Mistake
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.validation import replay_group, validation_game
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


def settings() -> LabelSettings:
    return LabelSettings(2, 2, 1, 12, 1_000_000, 1_000_000, 1 << 12, 2, 2)


def source_game(folder: Path, seed: int = 2) -> Path:
    """A shared-gameplay trajectory with explicit test visit distributions on legal edges."""
    record = play_game(BoxesRandomAgent(), BoxesRandomAgent(), game=get_game("boxes"),
                       board_size=2, seed=seed, render_debug_on_error=False)
    path = save_game_data(record, folder, seed, "ab-test")
    game = load_game(path)
    geo = geometry(2, 2)
    game["mcts_visits"] = np.asarray([
        [int(grid[r, c] == 0) for r, c in geo.edge_rc] for grid in game["boards"]], np.int16)
    game["mcts_root_values"] = np.full(len(game["moves"]), 0.5, np.float32)
    game["train_mask"] = np.ones(len(game["moves"]), bool)
    np.savez_compressed(path, **game)
    return path


def exclusion_file(path: Path, game_path: Path, family: str | None = None) -> Path:
    game = load_game(game_path)
    geo = geometry(2, 2)
    prefix = [int(geo.lattice_edge[r, c]) for r, c in game["moves"][:8]]
    row = dict(rows=2, cols=2, family=family or replay_group(game, geo).hex(), prefix=prefix,
               source=str(game_path), source_sha256=sha256(game_path))
    path.write_text(json.dumps(row) + "\n")
    return path


class TestCompleteChildren:
    def test_every_tiny_state_covers_capture_switch_double_and_terminal(self) -> None:
        oracle = Oracle(1, 2)
        solver = alpha_go_cpp.BoxesSolver(1, 2, 1 << 12)
        pending = [BoxesBoard(1, 2)]
        seen = set()
        captures = doubles = switches = terminals = collapses = 0
        while pending:
            board = pending.pop()
            key = board.edges, board.player(), tuple(board.owner)
            if key in seen or board.is_game_over():
                continue
            seen.add(key)
            mistake = Mistake(0, board, 0, oracle.final_margin(board), 0)
            children = solve_children(mistake, solver, 1_000_000)
            sign = 1 if board.player() == 0 else -1
            best = [c.edge for c in children if sign * int(c.margin_p1) == mistake.before]
            assert best == oracle.best_edges(board)
            for child in children:
                raw, collapsed = child.raw, child.collapsed
                raw_p1 = oracle.final_margin(raw) * (1 if raw.player() == 0 else -1)
                collapsed_p1 = oracle.final_margin(collapsed) * (
                    1 if collapsed.player() == 0 else -1)
                assert child.margin_p1 == raw_p1 == collapsed_p1
                gained = sum(raw.boxes) - sum(board.boxes)
                captures += gained > 0
                doubles += gained == 2
                switches += raw.player() != board.player()
                terminals += collapsed.is_game_over()
                collapses += collapsed.edges != raw.edges
                pending.append(raw)
        assert min(captures, doubles, switches, terminals, collapses) > 0

    def test_unknown_child_retains_other_values_but_no_policy_group(self, tmp_path: Path) -> None:
        path = source_game(tmp_path)
        game, labels, _ = solve_game(path, settings())
        assert labels
        mistake, complete = labels[0]
        unknown = replace(complete[0], margin_p1=None)
        output = tmp_path / "partial"
        for name in ["control", "exact", "children"]:
            (output / name).mkdir(parents=True)
        counts = write_arms(output, 0, path, game, [(mistake, [unknown, *complete[1:]])],
                            sha256(path), geometry(2, 2))
        assert counts["replaced_roots"] == counts["child_positions"] == 0
        assert not list((output / "children").iterdir())
        left = load_game(output / "control/source-0000000.npz")
        right = load_game(output / "exact/source-0000000.npz")
        assert left.keys() == right.keys()
        assert all(np.array_equal(left[k], right[k]) for k in left)
        assert not left["policy_ab_root_mask"].any()
        with (output / "children.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows[0]["exact_margin_p1"] == ""
        assert all(row["group_complete"] == "False" for row in rows)

    def test_solver_root_mismatch_rejected(self) -> None:
        board = BoxesBoard(1, 2)
        oracle = Oracle(1, 2)
        solver = alpha_go_cpp.BoxesSolver(1, 2, 1 << 12)
        wrong = Mistake(0, board, 0, oracle.final_margin(board) + 2, 0)
        with pytest.raises(AssertionError, match="root/child"):
            solve_children(wrong, solver, 100_000)


class TestSourceMasks:
    def test_learner_mask_and_positive_visits_required(self, tmp_path: Path) -> None:
        path = source_game(tmp_path)
        game, labels, _ = solve_game(path, settings())
        assert labels
        game["train_mask"][:] = False
        np.savez_compressed(path, **game)
        _, labels, counts = solve_game(path, settings())
        assert not labels and counts["opponent_moves_excluded"] > 0
        game["train_mask"][:] = True
        game["mcts_visits"][:] = 0
        np.savez_compressed(path, **game)
        _, labels, counts = solve_game(path, settings())
        assert not labels and counts["unsearched_roots_excluded"] > 0

    def test_existing_wrong_exact_value_rejected(self, tmp_path: Path) -> None:
        path = source_game(tmp_path)
        game, labels, _ = solve_game(path, settings())
        game["exact_margin_p1"] = np.full(len(game["moves"]), 99, np.int16)
        output = tmp_path / "bad"
        output.mkdir()
        with pytest.raises(AssertionError, match="saved exact root"):
            write_arms(output, 0, path, game, labels, sha256(path), geometry(2, 2))


class TestPreparation:
    def test_matched_arrays_children_lineage_hashes_and_portable_manifests(self,
                                                                       tmp_path: Path) -> None:
        paths = [source_game(tmp_path, seed) for seed in [2, 3]]
        hashes = [sha256(path) for path in paths]
        output = tmp_path / "prepared"
        summary = prepare(paths, output, settings(), workers=2)
        assert summary["status"] == "complete"
        assert summary["counts"]["replaced_roots"] > 0
        assert hashes == [sha256(path) for path in paths]
        assert summary["manifest_paths"] == {"control": "control-manifest.txt",
                                             "exact": "exact-manifest.txt"}
        assert (output / "control-manifest.txt").read_text() == "control\nchildren\n"
        assert (output / "exact-manifest.txt").read_text() == "exact\nchildren\n"
        for name, expected in summary["artifact_sha256"].items():
            assert sha256(output / name) == expected
        for control_path in sorted((output / "control").glob("*.npz")):
            control = load_game(control_path)
            exact = load_game(output / "exact" / control_path.name)
            original = load_game(Path(str(control["source_path"])))
            assert control.keys() == exact.keys()
            for key in control.keys() - {"mcts_visits"}:
                assert np.array_equal(control[key], exact[key]), key
            assert np.array_equal(control["mcts_visits"], original["mcts_visits"])
            mask = control["policy_ab_root_mask"]
            assert np.array_equal(control["mcts_visits"][~mask], exact["mcts_visits"][~mask])
            child_path = output / "children" / control_path.name
            if child_path.exists():
                children = load_game(child_path)
                assert not children["policy_mask"].any()
                assert np.all(children["moves"] == -1)
                for seed in [0, 1, 20]:
                    assert validation_game(children, geometry(2, 2), 0.2, seed) == (
                        validation_game(original, geometry(2, 2), 0.2, seed))
        # Value-only children are consumed by the actual dataset without a move fallback.
        children_data = BoxesDataset([output / "children"])
        for i in range(len(children_data)):
            row = children_data[i]
            assert row["policy_weight"] == 0
            assert row["policy"].sum() == 0
            assert row["exact"]
        with pytest.raises(AssertionError, match="fresh"):
            prepare(paths, output, settings())

    def test_worker_count_same_targets_and_frozen_replay(self, tmp_path: Path) -> None:
        fresh = [source_game(tmp_path, 2), source_game(tmp_path, 3)]
        replay = source_game(tmp_path, 4)
        for workers in [1, 2]:
            prepare(fresh, tmp_path / f"out-{workers}", settings(),
                    replay_paths=[replay], workers=workers)
        for name in ["control", "exact", "children", "replay"]:
            left = sorted((tmp_path / "out-1" / name).glob("*.npz"))
            right = sorted((tmp_path / "out-2" / name).glob("*.npz"))
            assert len(left) == len(right)
            for a, b in zip(left, right):
                aa, bb = load_game(a), load_game(b)
                assert aa.keys() == bb.keys()
                assert all(np.array_equal(aa[k], bb[k]) for k in aa)
        assert (tmp_path / "out-1/replay/replay-0000000.npz").read_bytes() == replay.read_bytes()


class TestExclusions:
    def test_opening_match_excludes_declared_pair_and_replay(self, tmp_path: Path) -> None:
        a, b, c = [source_game(tmp_path, seed) for seed in [2, 3, 4]]
        for path in [a, b]:
            game = load_game(path)
            game["split_group"] = np.array("paired-games")
            np.savez_compressed(path, **game)
        # A diagnostic from another grouping namespace still matches the opening of a.
        excluded = exclusion_file(tmp_path / "excluded.jsonl", a, family="abcd")
        fresh, replay, records = filter_sources([b, c], [a], [excluded], geometry(2, 2))
        assert fresh == [c.resolve()] and not replay
        reasons = {Path(r["path"]): r["excluded_reason"] for r in records}
        assert reasons[a.resolve()] == "diagnostic_opening"
        assert reasons[b.resolve()] == "paired_family"

    def test_family_and_exact_root_state_excluded(self, tmp_path: Path) -> None:
        a, b = [source_game(tmp_path, seed) for seed in [2, 3]]
        excluded = exclusion_file(tmp_path / "excluded.jsonl", a)
        fresh, _, records = filter_sources([a, b], [], [excluded], geometry(2, 2))
        assert fresh == [b.resolve()]
        assert records[0]["excluded_reason"] == "diagnostic_family"
        record: dict[str, Any] = json.loads(excluded.read_text())
        record["family"] = "abcd"
        # Change ordered prefix without changing its resulting state or the mover/score.
        record["prefix"][0], record["prefix"][1] = record["prefix"][1], record["prefix"][0]
        excluded.write_text(json.dumps(record) + "\n")
        _, _, records = filter_sources([a], [], [excluded], geometry(2, 2))
        assert records[0]["excluded_reason"] == "diagnostic_root_state"

    def test_canonical_openings_and_evaluation_path_guard(self, tmp_path: Path) -> None:
        geo = geometry(2, 2)
        edges = [0, 3, 7, 8, 2, 5, 6, 9]
        assert canonical_opening(edges, geo) == canonical_opening(edges + [1], geo)
        folder = tmp_path / "5x5-mid"
        folder.mkdir()
        path = source_game(folder)
        with pytest.raises(AssertionError, match="evaluation data"):
            filter_sources([path], [], [], geo)
