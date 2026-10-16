"""Exact-root curriculum fits legal policies without monitoring-family contamination."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from alpha_go.boxes.curriculum_train import (
    CurriculumSettings,
    advance_curriculum,
    evaluate_roots,
    load_roots,
    rehearsal,
    root_keys,
    run,
    select_candidate,
    select_roots,
)
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import save_boxes_net
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.policy_ab_data import canonical_opening
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.validation import replay_group


def make_bundle(folder: Path, prefixes: list[list[int]], split: str,
                *, families: list[str] | None = None) -> Path:
    """Write small real exact policies with child margins in the root mover's frame."""
    directory = folder / split
    directory.mkdir(parents=True)
    geo = geometry(2, 2)
    oracle = Oracle(2, 2)
    records = []
    for index, prefix in enumerate(prefixes):
        board = BoxesBoard(2, 2)
        for edge in prefix:
            assert board.play_edge(edge)
        best = oracle.final_margin(board)
        children = {edge: board.margin() + oracle.child_value(board.edges, edge)
                    for edge in board.get_legal_moves_flat()}
        optimal = [edge for edge, margin in children.items() if margin == best]
        family = families[index] if families else (
            b"opening:" + bytes.fromhex(canonical_opening(prefix, geo))).hex()
        visits = np.zeros((1, geo.num_edges), np.int16)
        visits[0, optimal] = 1
        filename = f"root-{index}.npz"
        margin_p1 = best if board.player() == 0 else -best
        np.savez_compressed(directory / filename,
                            boards=np.asarray([board.to_numpy()]),
                            to_play=np.asarray([board.player()], np.int8),
                            moves=np.asarray([geo.edge_rc[optimal[0]]], np.int16),
                            result="Draw", game="boxes", termination="solver",
                            mcts_visits=visits, mcts_root_values=np.asarray([float(best > 0)]),
                            policy_ab_root_mask=np.asarray([True]),
                            exact_margin_p1=np.asarray([margin_p1]),
                            source_replay_group=np.array(family))
        records.append(dict(rows=2, cols=2, prefix=prefix, family=family,
                            root_margin=best, child_margins=children, file=f"{split}/{filename}",
                            row_index=0, split=split))
    (folder / "roots.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return directory


def metric(regret: float, wins: int = 4) -> dict[str, Any]:
    return dict(positions=10, winning_roots=5, mean_regret=regret, wins_preserved=wins)


class TestSelection:
    def test_regret_and_win_preservation_both_required(self) -> None:
        assert select_candidate(metric(2), metric(1.9))["accepted"]
        assert not select_candidate(metric(2), metric(1.99))["accepted"]
        assert not select_candidate(metric(2), metric(1.9, 3))["accepted"]
        assert not select_candidate(metric(2), metric(2, 5))["accepted"]

    def test_matching_monitor_required(self) -> None:
        candidate = metric(1)
        candidate["positions"] = 9
        with pytest.raises(AssertionError):
            select_candidate(metric(2), candidate)

    def test_curriculum_advances_only_on_supported_accepted_read(self) -> None:
        current = dict(positions=32, families=8, winning_roots=10,
                       win_preservation=0.9, mean_regret=1.0)
        assert advance_curriculum(True, current, 34)
        assert not advance_curriculum(False, current, 34)
        assert not advance_curriculum(True, current, 36)
        for name, value in [("positions", 31), ("families", 7), ("winning_roots", 7),
                            ("win_preservation", 0.89), ("mean_regret", 1.1)]:
            assert not advance_curriculum(True, dict(current, **{name: value}), 34)


class TestRootData:
    def test_policy_checked_against_all_children(self, tmp_path: Path) -> None:
        directory = make_bundle(tmp_path, [[0, 2, 8, 5, 6, 10, 7, 4]], "train")
        dataset, labels, hashes = load_roots([directory], "chains")
        assert len(dataset) == 1 and len(labels) == 1 and len(hashes) == 2
        path = directory / "root-0.npz"
        with np.load(path) as arrays:
            game = dict(arrays)
        game["mcts_visits"][:] = 1
        np.savez_compressed(path, **game)
        with pytest.raises(AssertionError, match="policy/child"):
            load_roots([directory], "chains")

    def test_duplicate_states_and_family_aliases(self, tmp_path: Path) -> None:
        prefix = [0, 2, 8, 5, 6, 10, 7, 4]
        directory = make_bundle(tmp_path, [prefix, prefix], "train", families=["aa", "bb"])
        dataset, labels, _ = load_roots([directory], "chains")
        selected = select_roots(dataset, labels)
        assert len(dataset) == 2 and len(selected) == 1
        assert len(select_roots(dataset, labels, blocked_states=root_keys(selected))) == 0
        assert len(select_roots(dataset, labels, blocked_families={"aa", "bb"})) == 0
        opening = canonical_opening(prefix, dataset.geo)
        assert len(select_roots(dataset, labels, blocked_openings={opening})) == 0

    def test_exhaustive_regret_uses_root_mover_after_capture(self, tmp_path: Path) -> None:
        directory = make_bundle(tmp_path, [[0, 2, 8, 5, 6, 10, 7, 4],
                                            [8, 9, 5, 7, 2, 1, 0, 4, 6]], "validation")
        dataset, labels, _ = load_roots([directory], "chains")
        torch.manual_seed(7)
        model = BoxesNet(2, 2, channels=8, n_blocks=1, value_hidden=8, features="chains")
        result = evaluate_roots(model, dataset, labels, 8, torch.device("cpu"))
        assert result["positions"] == 2 and result["families"] == 2
        oracle = Oracle(2, 2)
        for row in result["per_position"]:
            expected = oracle.value(row["edges"]) - oracle.child_value(row["edges"], row["edge"])
            assert row["regret"] == expected >= 0
        assert 0 <= result["optimal_mass"] <= 1
        assert result["mean_regret"] >= 0

    def test_rehearsal_is_distinct_capped_and_family_excluded(self, tmp_path: Path) -> None:
        prefixes = [[0, 2, 8, 5, 6, 10, 7, 4], [8, 9, 5, 7, 2, 1, 0, 4],
                    [11, 9, 5, 6, 2, 1, 0, 4]]
        directory = make_bundle(tmp_path / "exact", prefixes, "train")
        dataset, _, _ = load_roots([directory], "chains")
        games = tmp_path / "rehearsal"
        games.mkdir()
        geo = geometry(2, 2)
        rng = np.random.default_rng(71)
        for index in range(6):
            board = BoxesBoard(2, 2)
            grids, movers, moves, visits = [], [], [], []
            for edge in rng.permutation(geo.num_edges):
                grids.append(board.to_numpy())
                movers.append(board.player())
                moves.append(geo.edge_rc[edge])
                visits.append([int(e in board.get_legal_moves_flat()) for e in range(12)])
                assert board.play_edge(int(edge))
            np.savez_compressed(games / f"{index}.npz", boards=np.asarray(grids),
                                to_play=np.asarray(movers), moves=np.asarray(moves),
                                mcts_visits=np.asarray(visits), result="B+2.0",
                                mcts_root_values=np.full(12, 0.5), game="boxes")
        source = BoxesDataset([games])
        blocked = {replay_group(g, source.geo).hex() for g in source.games[:2]}
        settings = CurriculumSettings(rehearsal_min_undrawn=9)
        combined, stats = rehearsal(dataset, [games], settings, blocked, set(), root_keys(dataset))
        assert stats["selected"] == 1  # Three roots + one rehearsal row = exactly 25%.
        assert len(combined) == 4 and len(root_keys(combined)) == 4
        assert all(replay_group(g, source.geo).hex() not in blocked
                   for g in combined.games[len(dataset.games):])
        again, stats_again = rehearsal(dataset, [games], settings, blocked, set(),
                                        root_keys(dataset))
        assert stats == stats_again and root_keys(combined) == root_keys(again)


def test_training_fixed_monitor_and_source_immutability(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(19)
    train = make_bundle(tmp_path / "training", [[0, 2, 8, 5, 6, 10, 7, 4],
                                               [11, 9, 5, 6, 2, 1, 0, 4]], "train")
    validation = make_bundle(tmp_path / "monitor", [[8, 9, 5, 7, 2, 1, 0, 4]], "validation")
    checkpoint = tmp_path / "reference.pt"
    save_boxes_net(BoxesNet(2, 2, channels=8, n_blocks=1, value_hidden=8, features="chains"),
                   checkpoint)
    original = checkpoint.read_bytes()
    output = tmp_path / "result"
    settings = CurriculumSettings(epochs=2, batch_size=2, max_undrawn=4)
    summary = run(checkpoint, [train], [validation], output, settings, cpu=True)
    assert summary["status"] == "complete"
    assert summary["training"]["roots"] == 2
    assert summary["training_run"]["steps"] == 2
    assert summary["candidate_training"]["positions"] == 2
    assert summary["reference_validation"]["current_band"]["positions"] == 1
    assert summary["candidate_validation"]["current_band"]["positions"] == 1
    assert summary["candidate_validation"]["35..36"]["positions"] == 0
    assert checkpoint.read_bytes() == original == (output / "reference.pt").read_bytes()
    assert json.loads((output / "summary.json").read_text()) == summary
    with pytest.raises(AssertionError, match="fresh"):
        run(checkpoint, [train], [validation], output, settings, cpu=True)


def test_unsolved_monitor_family_is_excluded(tmp_path: Path) -> None:
    train = make_bundle(tmp_path / "training", [[0, 2, 8, 5, 6, 10, 7, 4]], "train")
    validation = make_bundle(tmp_path / "monitor", [[8, 9, 5, 7, 2, 1, 0, 4]], "validation")
    excluded = json.loads((train.parent / "roots.jsonl").read_text())
    (validation.parent / "validation-sources.jsonl").write_text(json.dumps(excluded) + "\n")
    checkpoint = tmp_path / "reference.pt"
    save_boxes_net(BoxesNet(2, 2, channels=8, n_blocks=1, value_hidden=8, features="chains"),
                   checkpoint)
    with pytest.raises(AssertionError, match="distinct training"):
        run(checkpoint, [train], [validation], tmp_path / "result", cpu=True)


def test_actual_labeler_bundle_loads(tmp_path: Path) -> None:
    from alpha_go.boxes.curriculum_labels import CurriculumSettings as LabelSettings
    from alpha_go.boxes.curriculum_labels import prepare

    source = tmp_path / "searched-history.jsonl"
    source.write_text(json.dumps(dict(rows=2, cols=2,
                                     prefix=[0, 2, 8, 5, 6, 10, 7, 4, 1, 3, 9, 11])) + "\n")
    output = tmp_path / "labels"
    prepare([source], output, LabelSettings(rows=2, cols=2, min_undrawn=7, max_undrawn=9,
                                            table_entries=4096), force_split="train")
    dataset, labels, _ = load_roots([output / "train"], "chains")
    assert len(dataset) == len(labels) > 0
    assert len(select_roots(dataset, labels)) == len(dataset)
