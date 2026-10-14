"""Frozen search probes preserve extra turns, exact margins and common coverage."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import pytest
import torch

from alpha_go.boxes.agents import BoxesRandomAgent
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import save_boxes_net
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.search_probe import (
    ExactLabels,
    load_positions,
    main,
    sample_positions,
    selected_final,
    summarize,
)
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


def game_file(directory: Path, seed: int, branch: int = 0) -> Path:
    """Create a sequential training trajectory through the shared game loop."""
    directory.mkdir(parents=True, exist_ok=True)
    record = play_game(BoxesRandomAgent(), BoxesRandomAgent(), game=get_game("boxes"),
                       board_size=2, seed=seed, render_debug_on_error=False)
    path = save_game_data(record, directory, seed, "probe")
    if branch:
        with np.load(path) as loaded:
            game = dict(loaded)
        game["start_moves"] = game["moves"][:branch]
        for key in ("moves", "boards", "to_play"):
            game[key] = game[key][branch:]
        np.savez_compressed(path, **game)
    return path


class TestExactSelectedMoves:
    def test_all_tiny_masks_against_exhaustive_oracle(self) -> None:
        oracle = Oracle(1, 2)
        labels = ExactLabels(1, 2, 1_000_000, 1 << 12)
        captures = switches = scored = 0
        for mask in range(1 << oracle.geo.num_edges):
            board = alpha_go_cpp.BoxesBoard(1, 2)
            for edge in range(oracle.geo.num_edges):
                if mask >> edge & 1:
                    board.play_edge(edge)
            expected = board.margin() + oracle.value(mask)
            assert labels.final(board, board.to_play()) == expected
            assert labels.final(board, 3 - board.to_play()) == -expected
            scored += board.margin() != 0
            for edge in board.get_legal_moves_flat():
                child = selected_final(labels, board, edge)
                assert child == board.margin() + oracle.child_value(mask, edge)
                assert expected >= child
                captures += oracle.captures(mask, edge) > 0
                switches += oracle.captures(mask, edge) == 0
        assert captures and switches and scored

    def test_budget_unknown_and_readonly_geometry_cache(self, tmp_path: Path) -> None:
        labels = ExactLabels(2, 3, 1, 1 << 12)
        assert labels.remaining(0) is None
        assert labels.calls == 1
        assert labels.remaining(0) is None
        assert labels.calls == 1
        full = (1 << 17) - 1
        assert labels.remaining(full) == 0
        path = tmp_path / "labels.json"
        labels.save(path)
        before = path.read_bytes()
        cached = ExactLabels(2, 3, 1_000_000, 1 << 12, path)
        assert cached.remaining(full) == 0
        assert cached.calls == 0
        assert cached.remaining(0) is not None  # failures are retried in a new run
        assert path.read_bytes() == before
        with pytest.raises(ValueError, match="geometry"):
            ExactLabels(2, 2, 1, 1 << 12, path)


class TestFrozenPositions:
    def test_branches_static_collections_and_round_trip(self, tmp_path: Path) -> None:
        game_file(tmp_path / "full", 4)
        branch = game_file(tmp_path / "branch", 4, branch=1)
        with np.load(branch) as loaded:
            game = dict(loaded)
        game["termination"] = "solver"
        np.savez_compressed(tmp_path / "static.npz", **game)
        whole = sample_positions([tmp_path / "full"], 2, 2, 10, 11, 100, 9)
        branched = sample_positions([tmp_path / "branch"], 2, 2, 10, 11, 100, 9)
        assert whole and branched
        assert [p.prefix for p in whole] == [p.prefix for p in branched]
        assert [p.family for p in whole] == [p.family for p in branched]
        together = sample_positions([tmp_path], 2, 2, 10, 11, 100, 9)
        assert len(together) == len(whole)
        frozen = tmp_path / "positions.jsonl"
        frozen.write_text("".join(json.dumps(asdict(p)) + "\n" for p in whole))
        assert load_positions(frozen, 2, 2, 10, 11) == whole
        with pytest.raises(ValueError, match="dimensions"):
            load_positions(frozen, 1, 2, 7, 9)


class TestCommonMetrics:
    def test_missing_children_excluded_from_every_case(self) -> None:
        records: list[dict[str, Any]] = []
        for case in ("old", "new"):
            for position in range(3):
                records.append(dict(case=case, position=position, root_final=1,
                                    policy_final=1, search_final=1, search_seconds=0.1))
        records[-1]["search_final"] = None
        records[-2]["search_final"] = -1
        summary = summarize(records, ["old", "new"], 3)
        assert summary["common_known"] == 2
        assert summary["excluded"] == 1
        assert summary["cases"]["old"]["search_optimal"] == 1
        assert summary["cases"]["new"]["search_optimal"] == 0.5
        assert summary["cases"]["new"]["search_margin_lost"] == 1
        assert summary["cases"]["new"]["search_win_to_loss"] == 1
        assert summary["cases"]["new"]["search_wins_preserved"] == 1
        assert summary["cases"]["new"]["winning_roots"] == 2

    def test_no_known_is_not_zero_accuracy(self) -> None:
        summary = summarize([], ["old", "new"], 3)
        assert summary["common_known"] == 0
        assert summary["cases"]["old"]["search_optimal"] is None


class TestProbeCli:
    def test_tiny_cpu_sweep_and_reuse(self, tmp_path: Path) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(1)
        game_file(tmp_path / "games", 1)
        reference, candidate = tmp_path / "old.pt", tmp_path / "new.pt"
        save_boxes_net(BoxesNet(2, channels=8, n_blocks=1, value_hidden=8), reference)
        save_boxes_net(BoxesNet(2, channels=8, n_blocks=1, value_hidden=8), candidate)
        shared = ["--reference", str(reference), "--candidate", str(candidate),
                  "--rows", "2", "--min-undrawn", "8", "--max-undrawn", "10",
                  "--positions", "2", "--simulations", "2", "4", "--lambdas", "0.5", "0",
                  "--solver", "0", "--label-budget", "1000000",
                  "--label-table-entries", "4096", "--cpu"]
        output = tmp_path / "probe"
        main([*shared, "--games", str(tmp_path / "games"), "--output", str(output)])
        summary = json.loads((output / "summary.json").read_text())
        assert summary["complete"]
        assert len(summary["cases"]) == 16
        assert summary["common_known"] == 2
        assert summary["sampled"] == 2
        assert len(summary["checkpoint_sha256"]["reference"]) == 64
        frozen = (output / "positions.jsonl").read_bytes()
        cached = (output / "labels.json").read_bytes()
        repeat = tmp_path / "repeat"
        main([*shared, "--positions-file", str(output / "positions.jsonl"),
              "--label-cache", str(output / "labels.json"), "--output", str(repeat)])
        assert (output / "positions.jsonl").read_bytes() == frozen
        assert (repeat / "positions.jsonl").read_bytes() == frozen
        assert (output / "labels.json").read_bytes() == cached
        with pytest.raises(ValueError, match="fresh"):
            main([*shared, "--games", str(tmp_path / "games"), "--output", str(output)])
