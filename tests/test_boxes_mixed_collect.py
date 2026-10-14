"""Mixed training replay: quotas, extra turns, complete games and learner-only targets."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import alpha_go_cpp
import numpy as np
import pytest
import torch

from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.mixed_collect import collect, learner_train_mask, pair_schedule, parser
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import save_boxes_net
from alpha_go.boxes.rules import geometry
from alpha_go.boxes.validation import replay_group
from alpha_go.gameplay import GameRecord, MoveMetric


class TestPairSchedule:
    def test_balanced_quotas_and_repeatable_openings(self) -> None:
        plan = pair_schedule(2, 3, 20, [3, 1, 1], 4, 123)
        assert plan == pair_schedule(2, 3, 20, [3, 1, 1], 4, 123)
        assert Counter(pair.opponent for pair in plan) == {0: 6, 1: 2, 2: 2}
        assert len({pair.seed for pair in plan}) == 10
        assert len({pair.split_group for pair in plan}) > 1
        for pair in plan:
            board = alpha_go_cpp.BoxesBoard(2, 3)
            for row, col in pair.opening:
                assert board.is_legal(row, col)
                board.play(row, col)
            assert board.move_count() == 4

    def test_symmetric_openings_share_group(self) -> None:
        # Every one-edge opening on 1x1 is related by a board symmetry.
        plan = pair_schedule(1, 1, 20, [1], 1, 7)
        assert len({tuple(pair.opening) for pair in plan}) > 1
        assert len({pair.split_group for pair in plan}) == 1
        empty = pair_schedule(2, 2, 8, [1], 0, 7)
        assert len({pair.split_group for pair in empty}) == 4

    @pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [float("nan")], [float("inf")]])
    def test_invalid_weights_fail(self, weights: list[float]) -> None:
        with pytest.raises(AssertionError):
            pair_schedule(2, 2, 4, weights, 2, 0)

    def test_requires_whole_pairs(self) -> None:
        with pytest.raises(AssertionError, match="even"):
            pair_schedule(2, 2, 3, [1], 2, 0)


class TestTrainingMask:
    def test_extra_turns_and_unsearched_rows(self) -> None:
        record = GameRecord(board_size=2, black_agent="learner", white_agent="old")
        record.to_play = [0, 1, 1, 0, 0, 0]
        record.moves = [(0, 1)] * 6
        for i in range(6):
            visits = np.asarray([3, 1], dtype=np.int16) if i != 4 else None
            record.move_metrics.append(MoveMetric(
                move_step=i, policy_entropy=None, agent_color="black", visit_counts=visits))
        assert learner_train_mask(record, 0).tolist() == [True, False, False, True, False, True]
        assert learner_train_mask(record, 1).tolist() == [False, True, True, False, False, False]
        assert learner_train_mask(record, None).tolist() == [True, True, True, True, False, True]


class TestCollection:
    def test_tiny_board_collection_replay_and_training(self, tmp_path: Path) -> None:
        torch.manual_seed(4)
        checkpoint = tmp_path / "learner.pt"
        older = tmp_path / "older.pt"
        model = BoxesNet(rows=2, cols=2, channels=8, n_blocks=1, value_hidden=8)
        save_boxes_net(model, checkpoint)
        save_boxes_net(model, older)
        output = tmp_path / "mixed"
        args = parser().parse_args([
            "--checkpoint", str(checkpoint), "--output", str(output), "--rows", "2",
            "--num-games", "6", "--opening-moves", "2", "--seed", "42", "--sims", "4",
            "--opponent", "self", "--opponent", f"ckpt:{older}",
            "--opponent", "boxes-classic", "--weights", "1", "1", "1",
            "--solver", "12", "--solver-budget", "100000", "--solver_table_entries", "1024",
            "--margin_utility_lambda", "0.5", "--shape_terminals", "1", "--cpu",
        ])
        result = collect(args)
        assert result["status"] == "complete" and len(result["games"]) == 6
        assert len(result["checkpoint_sha256"]["learner"]) == 64
        assert len(result["source_sha256"]["alpha_go_cpp"]) == 64
        assert result["root_value_semantics"] == "shaped_utility"
        assert json.loads((output / "collection.json").read_text())["status"] == "complete"
        files = sorted(output.glob("*.npz"))
        assert len(files) == 6
        groups = []
        saw_extra_turn = False
        saw_opponent_search = False
        saw_shaped_utility = False
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                game = dict(archive)
            groups.append(replay_group(game, geometry(2, 2)))
            board = alpha_go_cpp.BoxesBoard(2, 2)
            for row, col in game["start_moves"]:
                board.play(int(row), int(col))
            for i, (row, col) in enumerate(game["moves"]):
                assert int(game["to_play"][i]) == board.player()
                assert np.array_equal(game["boards"][i], board.to_numpy())
                board.play(int(row), int(col))
            assert board.is_game_over() and len(game["moves"]) + len(game["start_moves"]) == 12
            saw_extra_turn |= bool(np.any(np.diff(game["to_play"]) == 0))
            visits = game["mcts_visits"].sum(axis=1)
            assert str(game["root_value_semantics"]) == "shaped_utility"
            saw_shaped_utility |= bool(np.any((game["mcts_root_values"] < 0)
                                             | (game["mcts_root_values"] > 1)))
            color = int(game["learner_color"])
            expected = (visits > 0) & ((game["to_play"] == color) | (color == -1))
            assert np.array_equal(game["train_mask"], expected)
            if str(game["opponent_spec"]).startswith("ckpt:"):
                saw_opponent_search |= bool(np.any((visits > 0) & ~expected))
            # The dataset must apply both masks by actual row index, not truncate to a prefix.
            dataset = BoxesDataset([], games=[game], min_undrawn=4)
            selected = np.flatnonzero(expected & (np.arange(10, 0, -1) >= 4))
            assert len(dataset) == len(selected)
            for sample, row in zip(dataset, selected):
                assert sample["has_mcts"]
                np.testing.assert_allclose(sample["policy"].numpy(),
                                           game["mcts_visits"][row] / visits[row])
                assert sample["root_value"].item() == pytest.approx(game["mcts_root_values"][row])
                expected_margin = int(board.score()) * (1 if game["to_play"][row] == 0 else -1)
                assert sample["margin"].item() == expected_margin
        assert saw_extra_turn and saw_opponent_search and saw_shaped_utility
        assert all(groups[i] == groups[i + 1] for i in range(0, 6, 2))
        full = BoxesDataset([output])
        assert len(full) == sum(game["training_rows"] for game in result["games"])
        before = (output / "collection.json").read_bytes()
        with pytest.raises(AssertionError, match="already exists"):
            collect(args)
        assert (output / "collection.json").read_bytes() == before
