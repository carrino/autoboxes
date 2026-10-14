"""Consequential policy labels preserve exact signs, complete move sets and game lineage."""
from __future__ import annotations

import hashlib
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
from alpha_go.boxes.policy_labels import (
    LabelSettings,
    Mistake,
    exact_policy,
    label_game,
    run,
    scan_game,
    source_paths,
)
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.boxes.validation import replay_group, validation_game
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


def training_game(folder: Path, seed: int = 0, start: int = 0) -> Path:
    """Generate a real sequential NPZ through the shared gameplay entry point."""
    record = play_game(BoxesRandomAgent(), BoxesRandomAgent(), game=get_game("boxes"),
                       board_size=2, seed=seed, render_debug_on_error=False)
    path = save_game_data(record, folder, seed, "policy-test")
    if start:
        with np.load(path) as data:
            game = dict(data)
        game["start_moves"] = game["moves"][:start]
        for key in ["moves", "boards", "to_play"]:
            game[key] = game[key][start:]
        game["num_moves"] = len(game["moves"])
        np.savez_compressed(path, **game)
    return path


def tiny_settings() -> LabelSettings:
    return LabelSettings(2, 2, 1, 12, 1_000_000, 1_000_000, 1 << 12, 2, 2)


class TestExactPolicy:
    @pytest.mark.parametrize("rows,cols", [(1, 1), (1, 2)])
    def test_every_reachable_tiny_state_matches_oracle(self, rows: int, cols: int) -> None:
        oracle = Oracle(rows, cols)
        solver = alpha_go_cpp.BoxesSolver(rows, cols, 1 << 12)
        pending = [BoxesBoard(rows, cols)]
        seen = set()
        captures = switches = 0
        while pending:
            board = pending.pop()
            key = board.edges, board.player(), tuple(board.owner)
            if key in seen or board.is_game_over():
                continue
            seen.add(key)
            mistake = Mistake(0, board, 0, oracle.final_margin(board), 0)
            assert exact_policy(mistake, solver, 1_000_000) == oracle.best_edges(board)
            for edge in board.get_legal_moves_flat():
                child = board.copy()
                assert child.play_edge(edge)
                captures += child.player() == board.player()
                switches += child.player() != board.player()
                pending.append(child)
        assert captures and switches

    def test_incomplete_child_never_produces_partial_policy(self) -> None:
        board = BoxesBoard(1, 2)
        oracle = Oracle(1, 2)

        class IncompleteSolver:
            def value_within(self, mask: int, budget: int) -> int | None:
                assert budget == 7
                return None if mask == 1 << 3 else oracle.value(mask)

        mistake = Mistake(0, board, 0, oracle.final_margin(board), 0)
        assert exact_policy(mistake, IncompleteSolver(), 7) is None


class TestMistakeScan:
    @pytest.mark.parametrize("start", [0, 4, 9])
    def test_adjacent_values_and_ranking_match_oracle(self, tmp_path: Path, start: int) -> None:
        settings = tiny_settings()
        oracle = Oracle(2, 2)
        for seed in range(8):
            path = training_game(tmp_path, seed, start)
            with np.load(path) as data:
                game = dict(data)
            solver = alpha_go_cpp.BoxesSolver(2, 2, 1 << 12)
            mistakes, counts = scan_game(game, settings, solver)
            expected = []
            board = BoxesBoard(2, 2)
            for row, col in game["start_moves"]:
                assert board.play(int(row), int(col))
            for index, (row, col) in enumerate(game["moves"]):
                before = oracle.final_margin(board)
                player = board.player()
                assert board.play(int(row), int(col))
                after = oracle.final_margin(board) * (1 if player == board.player() else -1)
                if before - after >= settings.min_regret:
                    expected.append((index, before, after))
            expected.sort(key=lambda m: (-int(m[1] > 0 and m[2] < 0), -(m[1] - m[2]), m[0]))
            assert [(m.index, m.before, m.after) for m in mistakes] == expected
            assert counts["known_adjacent"] == len(game["moves"])
            assert counts["unknown_adjacent"] == 0

    def test_unknown_line_and_opponent_moves_are_excluded(self, tmp_path: Path) -> None:
        path = training_game(tmp_path)
        with np.load(path) as data:
            game = dict(data)

        class UnknownSolver:
            def value_within(self, mask: int, budget: int) -> None:
                return None

        mistakes, counts = scan_game(game, tiny_settings(), UnknownSolver())
        assert not mistakes and counts["unknown_adjacent"] == 12
        game["train_mask"] = np.zeros(12, dtype=bool)
        mistakes, counts = scan_game(game, tiny_settings(), UnknownSolver())
        assert not mistakes and counts["opponent_moves_excluded"] == 12
        assert counts["unknown_adjacent"] == 0

    def test_last_stored_successor_is_solved(self, tmp_path: Path) -> None:
        path = training_game(tmp_path, seed=2)
        with np.load(path) as data:
            game = dict(data)
        solver = alpha_go_cpp.BoxesSolver(2, 2, 1 << 12)
        mistakes, _ = scan_game(game, tiny_settings(), solver)
        assert mistakes
        last = mistakes[0].index
        for key in ["boards", "to_play", "moves"]:
            game[key] = game[key][:last + 1]
        game["termination"] = "solved"
        short, _ = scan_game(game, tiny_settings(), solver)
        assert last in [m.index for m in short]


class TestCollection:
    def test_written_labels_split_with_source_and_train(self, tmp_path: Path) -> None:
        source = tmp_path / "training"
        source.mkdir()
        paths = [training_game(source, seed, start=4) for seed in range(8)]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        output = tmp_path / "labels"
        summary = run(paths, output, tiny_settings(), workers=2)
        assert 0 < summary["saved"] <= 16
        assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        assert json.loads((output / "settings.json").read_text())["status"] == "complete"
        dataset = BoxesDataset([output])
        assert dataset.num_exact() == len(dataset) == summary["saved"]
        oracle = Oracle(2, 2)
        for index, game in enumerate(dataset.games):
            with np.load(str(game["source_path"])) as data:
                original: dict[str, Any] = dict(data)
            assert replay_group(game, dataset.geo) == replay_group(original, dataset.geo)
            for seed in range(20):
                assert validation_game(game, dataset.geo, 0.2, seed) == validation_game(
                    original, dataset.geo, 0.2, seed)
            mask = sum(1 << e for e, (r, c) in enumerate(dataset.geo.edge_rc)
                       if game["boards"][0, r, c] == 1)
            optimal = [e for e in range(dataset.geo.num_edges) if not (mask >> e) & 1
                       and oracle.child_value(mask, e) == oracle.value(mask)]
            expected = np.zeros(dataset.geo.num_edges, np.float32)
            expected[optimal] = 1 / len(optimal)
            assert np.array_equal(dataset[index]["policy"].numpy(), expected)
        with pytest.raises(AssertionError, match="fresh"):
            run(paths, output, tiny_settings())

    def test_dedup_and_worker_count_do_not_change_labels(self, tmp_path: Path) -> None:
        source = tmp_path / "sources"
        source.mkdir()
        path = training_game(source, seed=2)
        duplicate = source / "duplicate.npz"
        duplicate.write_bytes(path.read_bytes())
        for workers in [1, 2]:
            output = tmp_path / f"labels-{workers}"
            summary = run([path, duplicate], output, tiny_settings(), workers)
            assert summary["duplicates"] == summary["saved"] > 0
        left = BoxesDataset([tmp_path / "labels-1"])
        right = BoxesDataset([tmp_path / "labels-2"])
        assert len(left) == len(right)
        for a, b in zip(left.games, right.games):
            for key in a:
                assert np.array_equal(a[key], b[key])

    def test_budget_and_source_validation(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError):
            replace(tiny_settings(), node_budget=0).validate()
        path = training_game(tmp_path)
        with np.load(path) as data:
            game = dict(data)
        game["boards"][0, 0, 1] = 1
        np.savez_compressed(path, **game)
        with pytest.raises(AssertionError, match="board mismatch"):
            label_game(path, tiny_settings())

    def test_manifest_resolution_and_evaluation_rejection(self, tmp_path: Path) -> None:
        train = tmp_path / "training"
        train.mkdir()
        path = training_game(train)
        manifest = tmp_path / "manifest.txt"
        manifest.write_text("# Training only\ntraining\n")
        assert source_paths([train], [manifest], tmp_path) == [path]
        evaluation = tmp_path / "5x5-mid"
        evaluation.mkdir()
        (evaluation / "positions.npz").write_bytes(path.read_bytes())
        alias = tmp_path / "disguised"
        alias.symlink_to(evaluation, target_is_directory=True)
        for directory in [evaluation, alias, tmp_path]:
            with pytest.raises(AssertionError, match="evaluation data"):
                source_paths([directory], [], tmp_path)
