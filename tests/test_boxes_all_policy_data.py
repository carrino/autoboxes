"""The pilot changes only complete policy targets, with first-four-family exclusion."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from alpha_go.boxes.agents import BoxesRandomAgent
from alpha_go.boxes.all_policy_data import (
    canonical_opening4,
    filter_sources,
    prepare,
    replay_source,
    solve_source,
)
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import save_boxes_net
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.policy_ab_data import load_game, sha256
from alpha_go.boxes.policy_labels import LabelSettings
from alpha_go.boxes.policy_target_ab import (
    TrainSettings,
    bundle_files,
    load_arm,
    matched_datasets,
    run,
)
from alpha_go.boxes.relabel import EXACT_UNKNOWN
from alpha_go.boxes.rules import geometry
from alpha_go.boxes.symmetry import apply, edge_permutation
from alpha_go.boxes.validation import replay_group
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


def settings() -> LabelSettings:
    return LabelSettings(2, 2, 4, 7, 100_000, 100_000, 1 << 12, 4, 2)


def source_game(folder: Path, seed: int = 0, start: list[tuple[int, int]] | None = None
                ) -> Path:
    """Shared gameplay generates real histories; test search counts cover legal edges."""
    folder.mkdir(parents=True, exist_ok=True)
    record = play_game(BoxesRandomAgent(), BoxesRandomAgent(), game=get_game("boxes"),
                       board_size=2, seed=seed, start_moves=start,
                       render_debug_on_error=False)
    path = save_game_data(record, folder, seed, "all-policy-test")
    game = load_game(path)
    game["mcts_visits"] = np.asarray([
        [int(grid[r, c] == 0) for r, c in geometry(2, 2).edge_rc]
        for grid in game["boards"]], np.int16)
    game["mcts_root_values"] = np.full(len(game["moves"]), 0.5, np.float32)
    game["train_mask"] = np.ones(len(game["moves"]), bool)
    np.savez_compressed(path, **game)
    return path


class BoundedOracle:
    """Deterministic budget failures exercise unknown recovery without timing assumptions."""

    def __init__(self, unknown: set[int]) -> None:
        self.oracle = Oracle(2, 2)
        self.unknown = unknown
        self.calls: Counter[int] = Counter()

    def value_within(self, mask: int, budget: int) -> int | None:
        assert budget > 0
        self.calls[mask] += 1
        return None if mask in self.unknown else self.oracle.value(mask)


class TestRootAndChildLabels:
    def test_unknown_root_recovered_without_repeating_child_solves(self, tmp_path: Path) -> None:
        path = source_game(tmp_path)
        original = sha256(path)
        states, _ = replay_source(load_game(path), geometry(2, 2))
        solver = BoundedOracle({states[5].edges})
        game, labels, counts, diagnostics = solve_source(
            path, replace(settings(), min_undrawn=7, max_per_game=1), solver)
        assert counts["complete_policies"] == counts["recovered_roots"] == 1
        root, children = labels[0]
        assert root.before == solver.oracle.final_margin(root.board)
        assert {c.edge for c in children if c.margin_p1 == diagnostics[0]["root_margin_p1"]} == (
            set(solver.oracle.best_edges(root.board)))
        for child in children:
            sign = 1 if child.collapsed.player() == 0 else -1
            assert child.margin_p1 == sign * solver.oracle.final_margin(child.collapsed)
        assert max(solver.calls.values()) == 1
        assert game["exact_margin_p1"][5] == diagnostics[0]["root_margin_p1"]
        assert sha256(path) == original

    @pytest.mark.parametrize("unknown_root", [False, True])
    def test_unknown_child_never_becomes_a_partial_policy(self, tmp_path: Path,
                                                         unknown_root: bool) -> None:
        path = source_game(tmp_path)
        original = load_game(path)
        states, _ = replay_source(original, geometry(2, 2))
        root = states[5]
        child_mask = root.edges | (1 << root.get_legal_moves_flat()[0])
        solver = BoundedOracle({child_mask} | ({root.edges} if unknown_root else set()))
        game, labels, counts, diagnostics = solve_source(
            path, replace(settings(), min_undrawn=7, max_per_game=1), solver)
        assert not labels and counts["incomplete_policies"] == 1
        assert counts["complete_policies"] == 0
        assert np.array_equal(game["mcts_visits"], original["mcts_visits"])
        assert list(diagnostics[0]["child_margin_p1"].values()).count(None) == 1
        assert diagnostics[0]["root_margin_p1"] is None if unknown_root else (
            diagnostics[0]["root_margin_p1"] is not None)
        assert (game["exact_margin_p1"][5] == EXACT_UNKNOWN) == unknown_root

    def test_all_known_source_values_preserved_and_disagreement_rejected(self,
                                                                        tmp_path: Path) -> None:
        path = source_game(tmp_path)
        original = load_game(path)
        states, _ = replay_source(original, geometry(2, 2))
        oracle = Oracle(2, 2)
        values = np.asarray([oracle.final_margin(b) * (1 if b.player() == 0 else -1)
                             for b in states], np.int16)
        original["exact_margin_p1"] = values
        np.savez_compressed(path, **original)
        game, _, _, _ = solve_source(path, settings())
        assert np.array_equal(game["exact_margin_p1"], values)
        original["exact_margin_p1"][5] = -4 if values[5] != -4 else 4
        np.savez_compressed(path, **original)
        with pytest.raises(AssertionError, match="saved exact value"):
            solve_source(path, settings())

    def test_bad_direct_root_disagrees_with_complete_children(self, tmp_path: Path) -> None:
        path = source_game(tmp_path)
        states, _ = replay_source(load_game(path), geometry(2, 2))
        wrong_mask = states[5].edges

        class WrongOracle(BoundedOracle):
            def value_within(self, mask: int, budget: int) -> int | None:
                value = super().value_within(mask, budget)
                assert value is not None
                return value + 2 if mask == wrong_mask else value

        with pytest.raises(AssertionError, match="root/child solver mismatch"):
            solve_source(path, settings(), WrongOracle(set()))


class TestSearchedEligibility:
    def test_raw_unsearched_rows_cannot_gain_policy_from_positive_saved_targets(
            self, tmp_path: Path) -> None:
        path = source_game(tmp_path)
        game = load_game(path)
        game["mcts_raw_visits"] = np.zeros_like(game["mcts_visits"])
        np.savez_compressed(path, **game)
        result, labels, counts, _ = solve_source(path, settings())
        assert not labels and result["policy_mask"].all()
        assert counts["unsearched_roots_excluded"] == 4

    def test_missing_visits_do_not_create_synthetic_policies(self, tmp_path: Path) -> None:
        path = source_game(tmp_path)
        game = load_game(path)
        del game["mcts_visits"]
        np.savez_compressed(path, **game)
        solver = BoundedOracle(set())
        result, labels, counts, _ = solve_source(path, settings(), solver)
        assert not labels and not solver.calls
        assert counts["unsearched_roots_excluded"] == 4
        assert result["policy_mask"].all()

    def test_train_and_policy_masks_are_respected_before_cap(self, tmp_path: Path) -> None:
        path = source_game(tmp_path)
        game = load_game(path)
        game["train_mask"][5] = False
        game["policy_mask"] = np.ones(len(game["moves"]), bool)
        game["policy_mask"][6] = False
        states, _ = replay_source(load_game(path), geometry(2, 2))
        game["exact_margin_p1"] = np.full(len(states), EXACT_UNKNOWN, np.int16)
        game["exact_margin_p1"][6] = (Oracle(2, 2).final_margin(states[6]) *
                                      (1 if states[6].player() == 0 else -1))
        game["mcts_visits"][7] = 0
        np.savez_compressed(path, **game)
        result, labels, counts, _ = solve_source(path, replace(settings(), max_per_game=1))
        assert [m.index for m, _ in labels] == [8]
        assert counts["opponent_roots_excluded"] == 1
        assert counts["unsearched_roots_excluded"] == 2
        assert not result["policy_mask"][6] and result["policy_mask"][7]

    def test_value_only_source_mask_requires_an_existing_exact_value(self, tmp_path: Path) -> None:
        path = source_game(tmp_path)
        game = load_game(path)
        game["policy_mask"] = np.zeros(len(game["moves"]), bool)
        np.savez_compressed(path, **game)
        with pytest.raises(AssertionError, match="value-only source row"):
            solve_source(path, settings())

    @pytest.mark.parametrize("corruption", ["train_mask", "policy_mask", "to_play",
                                             "boards", "visits", "moves"])
    def test_invalid_historical_replay_or_supervision_rejected(self, tmp_path: Path,
                                                              corruption: str) -> None:
        path = source_game(tmp_path)
        game = load_game(path)
        if corruption in ("train_mask", "policy_mask"):
            game[corruption] = np.ones(len(game["moves"]), np.int8)
        elif corruption == "visits":
            game["mcts_visits"][1] = 1  # Includes the already drawn edge.
        elif corruption == "moves":
            game["moves"] = game["moves"].astype(float)
        elif corruption == "to_play":
            game["to_play"][0] = 1
        else:
            game["boards"][0, 0, 1] = 1
        np.savez_compressed(path, **game)
        with pytest.raises(AssertionError):
            solve_source(path, settings())


class TestMatchedPreparation:
    def test_zero_visit_forced_moves_outside_band_keep_shared_fallback(
            self, tmp_path: Path) -> None:
        path = source_game(tmp_path / "source")
        source = load_game(path)
        source["mcts_visits"][[0, 11]] = 0
        source["mcts_raw_visits"] = source["mcts_visits"].copy()
        np.savez_compressed(path, **source)
        output = tmp_path / "prepared"
        prepare([path], output, settings(), workers=1)
        control, exact = [load_arm(output, arm, "basic", 0) for arm in ("control", "exact")]
        matched_datasets(control, exact)  # In particular value-only validation must not fail.
        ordinary = load_game(output / "games/0000000/control/source-0000000.npz")
        assert ordinary["policy_mask"][[0, 11]].all()
        assert (ordinary["exact_margin_p1"][[0, 11]] == EXACT_UNKNOWN).all()
        assert not ordinary["mcts_visits"][[0, 11]].any()
        assert not ordinary["policy_ab_root_mask"][[0, 11]].any()

    def test_real_cpu_training_has_identical_updates_and_separate_families(self,
                                                                        tmp_path: Path) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(421)
        paths = [source_game(tmp_path / "sources", seed=seed) for seed in range(12)]
        prepared = tmp_path / "prepared"
        prepare(paths, prepared, settings(), workers=2)
        reference = tmp_path / "reference.pt"
        save_boxes_net(BoxesNet(2, 2, channels=8, n_blocks=1, value_hidden=8,
                               features="chains"), reference)
        before = bundle_files(prepared)
        config = TrainSettings(epochs=2, batch_size=32, val_fraction=0.4, min_undrawn=0)
        summary = run(reference, prepared, tmp_path / "trained", config, cpu=True)
        assert summary["status"] == "complete"
        control, exact = (summary["arms"][arm] for arm in ("control", "exact"))
        assert control["steps"] == exact["steps"] > 1
        assert control["update_trace_sha256"] == exact["update_trace_sha256"]
        assert summary["training"]["changed_policy_rows"] > 0
        assert summary["validation"]["exact_roots"] > 0
        data = load_arm(prepared, "exact", "chains", 0)
        train, val = data.split_stable(config.val_fraction, config.split_seed)
        assert {replay_group(g, data.geo) for g in train.games}.isdisjoint(
            {replay_group(g, data.geo) for g in val.games})
        prepare(paths, prepared, settings(), workers=2)
        assert bundle_files(prepared) == before == summary["prepared_sha256"]

    def test_full_rows_values_children_and_masks_match_control_changes_only_mistakes(
            self, tmp_path: Path) -> None:
        path = source_game(tmp_path / "source", seed=0)
        original = load_game(path)
        original["source_replay_group"] = np.array(b"explicit-lineage".hex())
        np.savez_compressed(path, **original)
        source_hash = sha256(path)
        output = tmp_path / "prepared"
        result = prepare([path], output, settings(), workers=1)
        assert result["counts"]["control_exact_policies"] == 2
        assert result["counts"]["additional_exact_policies"] == 2
        left, right = (load_game(output / f"games/0000000/{a}/source-0000000.npz")
                       for a in ("control", "exact"))
        assert len(left["boards"]) == len(original["boards"])
        assert left.keys() == right.keys()
        for key in left:
            if key != "mcts_visits":
                assert np.array_equal(left[key], right[key]), key
        for target in (left, right):
            assert replay_group(target, geometry(2, 2)) == b"explicit-lineage"
        diagnostics = json.loads((output / "games/0000000/diagnostics.json").read_text())
        for row in diagnostics:
            index = row["position"]
            if row["played_regret"] >= 2:
                assert np.array_equal(left["mcts_visits"][index], right["mcts_visits"][index])
            else:
                assert np.array_equal(left["mcts_visits"][index], original["mcts_visits"][index])
        unmarked = ~left["policy_ab_root_mask"]
        assert np.array_equal(left["mcts_visits"][unmarked], original["mcts_visits"][unmarked])
        assert sha256(path) == source_hash
        control = load_arm(output, "control", "basic", 0)
        exact = load_arm(output, "exact", "basic", 0)
        matched = matched_datasets(control, exact)
        assert matched["exact_roots"] == 4
        assert matched["changed_policy_rows"] == result["counts"]["changed_policy_rows"] == 1
        assert matched["value_only_rows"] > 0
        child = load_game(output / "games/0000000/children/source-0000000.npz")
        assert not child["policy_mask"].any()
        assert replay_group(child, geometry(2, 2)) == b"explicit-lineage"
        before = bundle_files(output)
        again = prepare([path], output, settings(), workers=2)
        assert again == result and bundle_files(output) == before

    def test_resume_completed_shard_and_reject_tampering(self, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
        import alpha_go.boxes.all_policy_data as module

        path = source_game(tmp_path / "source")
        output = tmp_path / "prepared"
        result = prepare([path], output, settings(), workers=1)
        metadata = output / "preparation.json"
        result["status"] = "running"
        metadata.write_text(json.dumps(result))
        incomplete = output / "games/0000001.partial"
        incomplete.mkdir()
        (incomplete / "partial.npz").write_bytes(b"unfinished")

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("completed game was solved again")

        monkeypatch.setattr(module, "solve_source", forbidden)
        result = prepare([path], output, settings(), workers=1)
        assert result["status"] == "complete" and not incomplete.exists()
        bundle_files(output)
        artifact = output / "games/0000000/diagnostics.json"
        artifact.write_text("[]")
        with pytest.raises(AssertionError, match="completed preparation changed"):
            prepare([path], output, settings(), workers=1)

    def test_changed_source_is_not_reused(self, tmp_path: Path) -> None:
        path = source_game(tmp_path / "source")
        output = tmp_path / "prepared"
        prepare([path], output, settings(), workers=1)
        game = load_game(path)
        game["mcts_root_values"][0] = 0.75
        np.savez_compressed(path, **game)
        with pytest.raises(AssertionError, match="recipe changed"):
            prepare([path], output, settings(), workers=1)


class TestExclusions:
    @pytest.mark.parametrize("transform", range(8))
    def test_first_four_catches_different_continuations_and_symmetries(self, tmp_path: Path,
                                                                    transform: int) -> None:
        path = source_game(tmp_path / "original")
        original = load_game(path)
        geo = geometry(2, 2)
        _, history = replay_source(original, geo)
        prefix = history[:4]
        branch = source_game(tmp_path / "branch", seed=51,
                             start=[tuple(map(int, rc)) for rc in original["moves"][:4]])
        game = load_game(branch)
        perm = edge_permutation(geo, transform)
        game["boards"] = apply(game["boards"], transform).copy()
        for field in ("moves", "start_moves"):
            edges = [int(geo.lattice_edge[r, c]) for r, c in game[field]]
            game[field] = np.asarray([geo.edge_rc[perm[e]] for e in edges], np.int16)
        transformed = np.empty_like(game["mcts_visits"])
        transformed[:, perm] = game["mcts_visits"]
        game["mcts_visits"] = transformed
        np.savez_compressed(branch, **game)
        frozen = tmp_path / "probe.jsonl"
        frozen.write_text(json.dumps(dict(rows=2, cols=2, family="different-namespace",
                                          prefix=prefix)) + "\n")
        kept, records = filter_sources([branch], [frozen], geo)
        assert not kept and records[0]["excluded_reason"] == "diagnostic_opening4"
        _, transformed_history = replay_source(game, geo)
        assert canonical_opening4(transformed_history, geo) == canonical_opening4(history, geo)

    def test_explicit_lineage_exclusion_propagates_across_supplied_directories(
            self, tmp_path: Path) -> None:
        paths = [source_game(tmp_path / str(i), seed=i) for i in (0, 2)]
        for path in paths:
            game = load_game(path)
            game["source_replay_group"] = np.array(b"shared-source".hex())
            np.savez_compressed(path, **game)
        geo = geometry(2, 2)
        _, prefix = replay_source(load_game(paths[0]), geo)
        frozen = tmp_path / "probe.jsonl"
        frozen.write_text(json.dumps(dict(rows=2, cols=2, family="probe-namespace",
                                          prefix=prefix[:7])) + "\n")
        kept, records = filter_sources(paths, [frozen], geo)
        assert not kept
        assert {r["excluded_reason"] for r in records} == {"diagnostic_opening4", "paired_family"}
