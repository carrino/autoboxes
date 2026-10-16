"""Strong-history curriculum targets retain exact moves and isolated source families."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alpha_go.boxes import curriculum_labels as labels
from alpha_go.boxes.agents import BoxesRandomAgent
from alpha_go.boxes.curriculum_labels import CurriculumSettings, prepare
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.policy_ab_data import load_game, sha256
from alpha_go.boxes.rules import geometry
from alpha_go.boxes.search_probe import load_positions
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game


def settings() -> CurriculumSettings:
    return CurriculumSettings(2, 2, 1, 9, 100_000, 100_000, 1 << 12, 0.3, 0)


def source(folder: Path, seed: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    record = play_game(BoxesRandomAgent(), BoxesRandomAgent(), game=get_game("boxes"),
                       board_size=2, seed=seed, render_debug_on_error=False)
    return save_game_data(record, folder, seed, "curriculum-test")


def json_source(path: Path, game_path: Path, truncate: int = 12) -> Path:
    game = load_game(game_path)
    geo = geometry(2, 2)
    prefix = [int(geo.lattice_edge[r, c]) for r, c in game["moves"][:truncate]]
    board = labels.replay_prefix(prefix, 2, 2)
    row = dict(rows=2, cols=2, prefix=prefix, scores=board.boxes, to_play=board.to_play(),
               source_family="not-trusted-as-split")
    path.write_text(json.dumps(row) + "\n")
    return path


class TestExactCurriculumTargets:
    @pytest.mark.parametrize("include_children", [False, True])
    def test_targets_and_child_signs_match_oracle(self, tmp_path: Path,
                                                  include_children: bool) -> None:
        inputs = [source(tmp_path / "games", seed) for seed in range(8)]
        hashes = [sha256(path) for path in inputs]
        output = tmp_path / "labels"
        config = replace(settings(), include_children=include_children)
        summary = prepare(inputs, output, config, workers=2)
        assert hashes == [sha256(path) for path in inputs]
        assert summary["counts"]["complete_roots"] > 0
        oracle = Oracle(2, 2)
        records = [json.loads(line) for line in (output / "roots.jsonl").read_text().splitlines()]
        for record in records:
            board = labels.replay_prefix(record["prefix"], 2, 2)
            assert record["root_margin"] == oracle.final_margin(board)
            assert record["optimal_edges"] == oracle.best_edges(board)
            assert set(map(int, record["child_margins"])) == set(board.get_legal_moves_flat())
            for edge, expected in record["child_margins"].items():
                child = board.copy()
                assert child.play_edge(int(edge))
                sign = 1 if child.player() == board.player() else -1
                assert expected == sign * oracle.final_margin(child)
            game = load_game(output / record["file"])
            row = record["row_index"]
            assert np.array_equal(game["boards"][row], board.to_numpy())
            assert game["source_replay_group"].item() == record["family"]
        families = {split: {r["family"] for r in records if r["split"] == split}
                    for split in ["train", "validation"]}
        assert not families["train"] & families["validation"]
        states = {split: {(r["edges"], r["mover"], r["score_p1"]) for r in records
                          if r["split"] == split} for split in ["train", "validation"]}
        assert not states["train"] & states["validation"]
        data = BoxesDataset([output / "train", output / "validation"])
        assert data.num_exact() == len(data)
        value_only = 0
        for row in data:
            if row["policy_weight"] == 0:
                assert row["policy"].sum() == 0
                value_only += 1
            else:
                assert np.isclose(row["policy"].sum().item(), 1)
        assert (value_only > 0) == include_children
        for split in ["train", "validation"]:
            positions = load_positions(output / f"{split}-positions.jsonl", 2, 2, 1, 9)
            assert len(positions) == summary["counts"][f"{split}_roots"]

    def test_json_history_endpoint_and_npz_equivalent(self, tmp_path: Path) -> None:
        path = source(tmp_path / "games", 2)
        path = source(tmp_path / "games", 0)
        prefix_path = json_source(tmp_path / "prefix.jsonl", path, truncate=8)
        loaded = labels.load_sources([prefix_path], settings())[0]
        candidates = labels.candidate_roots(loaded, settings())
        assert any(index == 8 for index, _ in candidates)  # endpoint has no stored next move
        npz_source = labels.load_sources([path], settings())[0]
        assert loaded.family == npz_source.family
        row = json.loads(prefix_path.read_text())
        row["scores"][0] += 1
        prefix_path.write_text(json.dumps(row) + "\n")
        with pytest.raises(AssertionError, match="scores"):
            labels.load_sources([prefix_path], settings())

    def test_incomplete_child_never_emits_root_policy(self, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
        path = source(tmp_path / "games", 2)
        real = labels.solve_children

        def incomplete(*args: Any) -> Any:
            children = real(*args)
            children[0].margin_p1 = None
            return children

        monkeypatch.setattr(labels, "solve_children", incomplete)
        output = tmp_path / "labels"
        result = prepare([path], output, settings())
        assert result["counts"]["incomplete_children"] > 0
        assert result["counts"]["complete_roots"] == 0
        assert not list(output.rglob("*.npz"))
        assert not (output / "roots.jsonl").read_text()


class TestFrozenResumeAndExclusion:
    def test_resume_preserves_completed_sources_and_rejects_changed_recipe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        inputs = [source(tmp_path / "games", seed) for seed in [2, 3]]
        real = labels.solve_roots
        calls = 0

        def interrupted(*args: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("interrupted")
            return real(*args)

        monkeypatch.setattr(labels, "solve_roots", interrupted)
        output = tmp_path / "labels"
        with pytest.raises(RuntimeError, match="interrupted"):
            prepare(inputs, output, settings(), workers=1)
        completed = list((output / "completed").glob("*.json"))
        assert len(completed) == 1
        saved = {path: sha256(path) for path in output.rglob("*.npz")}
        monkeypatch.setattr(labels, "solve_roots", real)
        prepare(inputs, output, settings(), workers=2)
        assert saved == {path: sha256(path) for path in saved}

        def forbidden(*args: Any) -> Any:
            raise AssertionError("completed source must not be solved twice")

        monkeypatch.setattr(labels, "solve_roots", forbidden)
        prepare(inputs, output, settings(), workers=1)
        with pytest.raises(AssertionError, match="recipe"):
            prepare(inputs, output, replace(settings(), max_undrawn=8))
        any_path = next(output.rglob("*.npz"))
        any_path.write_bytes(b"changed")
        with pytest.raises(AssertionError, match="artifact changed"):
            prepare(inputs, output, settings())

    def test_fixed_monitor_excludes_entire_family_even_unsolved(
        self, tmp_path: Path,
    ) -> None:
        a, b = [source(tmp_path / "games", seed) for seed in [2, 3]]
        monitor = tmp_path / "monitor"
        prepare([a], monitor, settings(), force_split="validation")
        assert not list((monitor / "train").iterdir())
        output = tmp_path / "training"
        result = prepare([a, b], output, settings(), force_split="train",
                         exclude_positions=[monitor / "validation-sources.jsonl"])
        assert result["counts"]["validation_roots"] == 0
        recipe = json.loads((output / "recipe.json").read_text())
        assert sum(row["excluded"] for row in recipe["sources"]) == 1
        monitor_sources = [json.loads(line) for line in
                           (monitor / "validation-sources.jsonl").read_text().splitlines()]
        records = [json.loads(line) for line in (output / "roots.jsonl").read_text().splitlines()]
        assert all(r["family"] != monitor_sources[0]["family"] for r in records)

    def test_full_and_shortened_same_family_are_deduplicated(self, tmp_path: Path) -> None:
        path = source(tmp_path / "games", 2)
        prefix = json_source(tmp_path / "short.jsonl", path, truncate=10)
        output = tmp_path / "labels"
        result = prepare([prefix, path], output, settings())
        rows = [json.loads(line) for line in (output / "roots.jsonl").read_text().splitlines()]
        assert len({r["family"] for r in rows}) == 1
        assert len({(r["edges"], r["mover"], r["score_p1"]) for r in rows}) == len(rows)
        assert result["counts"]["complete_roots"] == len(rows)
