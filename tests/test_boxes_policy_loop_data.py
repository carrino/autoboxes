"""Per-iteration exact policy bundles resume without changing completed training data."""
from __future__ import annotations

import json
import shutil
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alpha_go.boxes import policy_loop_data
from alpha_go.boxes.agents import BoxesRandomAgent
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.policy_ab_data import load_game, sha256
from alpha_go.boxes.policy_labels import LabelSettings
from alpha_go.boxes.policy_loop_data import prepare_iteration
from alpha_go.boxes.rules import geometry
from alpha_go.boxes.validation import replay_group
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


def settings() -> LabelSettings:
    return LabelSettings(2, 2, 1, 12, 1_000_000, 1_000_000, 1 << 12, 2, 2)


def source_game(folder: Path, seed: int) -> Path:
    """Save a real trajectory with explicit synthetic legal visit targets for testing."""
    record = play_game(BoxesRandomAgent(), BoxesRandomAgent(), game=get_game("boxes"),
                       board_size=2, seed=seed, render_debug_on_error=False)
    path = save_game_data(record, folder, seed, "loop-policy-test")
    game = load_game(path)
    geo = geometry(2, 2)
    game["mcts_visits"] = np.asarray([
        [int(grid[r, c] == 0) for r, c in geo.edge_rc] for grid in game["boards"]], np.int16)
    game["mcts_root_values"] = np.full(len(game["moves"]), 0.5, np.float32)
    game["train_mask"] = np.ones(len(game["moves"]), bool)
    np.savez_compressed(path, **game)
    return path


class TestLoopBundles:
    def test_game_without_qualifying_mistake_keeps_ordinary_rows(self, tmp_path: Path) -> None:
        path = source_game(tmp_path, 2)
        output = tmp_path / "prepared"
        result = prepare_iteration([path], output, replace(settings(), min_regret=99))
        assert result["counts"].get("replaced_roots", 0) == 0
        assert (output / "dataset.txt").read_text() == "games/game-0000000/exact\n"
        original = load_game(path)
        saved = load_game(output / "games/game-0000000/exact/source-0000000.npz")
        assert np.array_equal(saved["mcts_visits"], original["mcts_visits"])
        assert np.array_equal(saved["train_mask"], original["train_mask"])
        assert not saved["policy_ab_root_mask"].any()

    def test_targets_lineage_portable_manifest_and_no_solve_reuse(self, tmp_path: Path,
                                                               monkeypatch: Any) -> None:
        paths = [source_game(tmp_path, seed) for seed in [2, 3]]
        hashes = [sha256(path) for path in paths]
        output = tmp_path / "prepared"
        summary = prepare_iteration(paths, output, settings(), workers=2)
        assert summary["status"] == "complete" and summary["completed_games"] == 2
        assert summary["counts"]["replaced_roots"] > 0
        assert hashes == [sha256(path) for path in paths]
        manifest = (output / "dataset.txt").read_text().splitlines()
        assert manifest and all(not Path(line).is_absolute() for line in manifest)
        data = BoxesDataset([output / line for line in manifest])
        assert len(data) > 24 and data.num_exact() > 0
        assert not list(output.rglob("control"))
        for directory in sorted((output / "games").iterdir()):
            saved = json.loads((directory / "receipt.json").read_text())
            original = load_game(Path(saved["source"]["path"]))
            game = load_game(next((directory / "exact").glob("*.npz")))
            assert np.array_equal(original["boards"], game["boards"])
            assert np.array_equal(original["train_mask"], game["train_mask"])
            marked = game["policy_ab_root_mask"]
            assert np.array_equal(original["mcts_visits"][~marked], game["mcts_visits"][~marked])
            for child_file in (directory / "children").glob("*.npz"):
                child = load_game(child_file)
                assert not child["policy_mask"].any()
                assert replay_group(child, geometry(2, 2)) == replay_group(original, geometry(2, 2))
        before = {str(path): sha256(path) for path in (output / "games").rglob("*")
                  if path.is_file()}

        def never_solve(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("a complete source game was solved again")

        monkeypatch.setattr(policy_loop_data, "solve_game", never_solve)
        again = prepare_iteration(paths, output, settings(), workers=1)
        assert again["reused_games"] == 2 and again["counts"] == summary["counts"]
        assert before == {str(path): sha256(path) for path in (output / "games").rglob("*")
                          if path.is_file()}

    def test_interruption_reuses_published_game_and_rebuilds_only_partial(self, tmp_path: Path,
                                                                      monkeypatch: Any) -> None:
        paths = [source_game(tmp_path, seed) for seed in [2, 3]]
        output = tmp_path / "prepared"
        original_publish = policy_loop_data.publish_game

        def interrupted(folder: Path, index: int, source: dict[str, Any], config: LabelSettings,
                        identity: str) -> dict[str, Any]:
            if index == 1:
                stage = folder / ".partial-game-0000001" / "exact"
                stage.mkdir(parents=True)
                (stage / "unfinished.npz").write_bytes(b"interrupted")
                raise RuntimeError("simulated interruption")
            return original_publish(folder, index, source, config, identity)

        monkeypatch.setattr(policy_loop_data, "publish_game", interrupted)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            prepare_iteration(paths, output, settings(), workers=1)
        assert not (output / "dataset.txt").exists()
        assert json.loads((output / "preparation.json").read_text())["completed_games"] == 1
        first = output / "games/game-0000000/receipt.json"
        first_hash = sha256(first)
        monkeypatch.setattr(policy_loop_data, "publish_game", original_publish)
        original_solve = policy_loop_data.solve_game
        solved = []

        def counted(path: Path, config: LabelSettings) -> Any:
            solved.append(path)
            return original_solve(path, config)

        monkeypatch.setattr(policy_loop_data, "solve_game", counted)
        summary = prepare_iteration(paths, output, settings(), workers=2)
        assert summary["status"] == "complete" and summary["reused_games"] == 1
        assert solved == [sorted(paths)[1].resolve()]
        assert sha256(first) == first_hash
        assert not list(output.glob(".partial-*"))

    @pytest.mark.parametrize("mutation", ["source", "settings", "artifact", "manifest", "receipt",
                                         "deleted_bundle", "extra_file"])
    def test_resume_rejects_input_and_completed_output_mutation(self, tmp_path: Path,
                                                              mutation: str) -> None:
        path = source_game(tmp_path, 2)
        output = tmp_path / "prepared"
        prepare_iteration([path], output, settings())
        config = settings()
        bundle = output / "games/game-0000000"
        if mutation == "source":
            game = load_game(path)
            game["note"] = np.array("changed")
            np.savez_compressed(path, **game)
        elif mutation == "settings":
            config = replace(config, child_node_budget=100)
        elif mutation == "artifact":
            next((bundle / "exact").glob("*.npz")).write_bytes(b"changed")
        elif mutation == "manifest":
            (output / "dataset.txt").write_text("games/game-0000000\n")
        elif mutation == "receipt":
            (bundle / "receipt.json").write_text("{}\n")
        elif mutation == "deleted_bundle":
            shutil.rmtree(bundle)
        else:
            (bundle / "new.npz").write_bytes(b"extra")
        with pytest.raises(AssertionError):
            prepare_iteration([path], output, config)

    def test_exclusion_identity_and_heldout_families(self, tmp_path: Path) -> None:
        paths = [source_game(tmp_path, seed) for seed in [2, 3]]
        first = load_game(paths[0])
        geo = geometry(2, 2)
        exclusion = tmp_path / "heldout.jsonl"
        excluded = dict(rows=2, cols=2, family=replay_group(first, geo).hex(),
                        prefix=[int(geo.lattice_edge[r, c]) for r, c in first["moves"][:8]])
        exclusion.write_text(json.dumps(excluded) + "\n")
        output = tmp_path / "prepared"
        result = prepare_iteration(paths, output, settings(), exclude_positions=[exclusion])
        assert result["excluded_files"] == 1 and result["included_files"] == 1
        exclusion.write_text(exclusion.read_text() + "\n")
        with pytest.raises(AssertionError, match="inputs/settings/exclusions"):
            prepare_iteration(paths, output, settings(), exclude_positions=[exclusion])

    def test_worker_bound_and_one_solve_per_game(self, tmp_path: Path, monkeypatch: Any) -> None:
        paths = [source_game(tmp_path, seed) for seed in [2, 3, 4, 5]]
        real_solve = policy_loop_data.solve_game
        calls: Counter[str] = Counter()
        lock = threading.Lock()
        active = maximum = 0

        def counted(path: Path, config: LabelSettings) -> Any:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                calls[str(path)] += 1
            time.sleep(0.01)
            result = real_solve(path, config)
            with lock:
                active -= 1
            return result

        monkeypatch.setattr(policy_loop_data, "solve_game", counted)
        prepare_iteration(paths, tmp_path / "prepared", settings(), workers=2)
        assert 1 <= maximum <= 2
        assert calls == Counter({str(path.resolve()): 1 for path in paths})
