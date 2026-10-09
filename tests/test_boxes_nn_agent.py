"""Boxes NN evaluators, MCTS agent, batched engine and dataset."""
from __future__ import annotations

import sys
from pathlib import Path

import alpha_go_cpp
import numpy as np
import pytest
import torch

from alpha_go import self_play
from alpha_go.agents import get_agent
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.inference import PlaneBatchedEngine
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import (
    BoxesEngineEvaluator,
    BoxesLeafEvaluator,
    BoxesMCTSAgent,
    add_search_flags,
    load_boxes_net,
    register_boxes_mcts_agent,
    save_boxes_net,
    search_flags,
)
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.gameplay import play_game, save_game_data
from alpha_go.games import get_game

CPU = torch.device("cpu")


def small_net(rows: int = 3, cols: int | None = None) -> BoxesNet:
    torch.manual_seed(0)
    return BoxesNet(rows, cols, channels=16, n_blocks=2, value_hidden=8)


class TestEvaluators:
    def test_leaf_evaluator_policy_is_legal_and_normalised(self) -> None:
        evaluator = BoxesLeafEvaluator(small_net(), CPU)
        a, b = alpha_go_cpp.BoxesBoard(3), alpha_go_cpp.BoxesBoard(3)
        b.play_edge(0)
        (pa, va), (pb, vb) = evaluator.batch_evaluate([a, b])
        assert set(pa) == set(range(24)) and 0 not in pb and len(pb) == 23
        assert sum(pa.values()) == pytest.approx(1.0) and sum(pb.values()) == pytest.approx(1.0)
        assert va == pytest.approx(0.5) and vb == pytest.approx(0.5)  # zero-init value head

    def test_margin_utility_shapes_value(self) -> None:
        net = small_net()
        with torch.no_grad():
            net.value_fc2.bias[net.num_boxes + 4] = 20.0  # peaked at margin +4
        plain = BoxesLeafEvaluator(net, CPU).evaluate(alpha_go_cpp.BoxesBoard(3))[1]
        shaped = BoxesLeafEvaluator(net, CPU, margin_utility_lambda=0.5, margin_utility_k=6.0)
        assert plain == pytest.approx(1.0, abs=1e-4)
        assert shaped.evaluate(alpha_go_cpp.BoxesBoard(3))[1] == pytest.approx(
            1.0 + 0.5 * np.tanh(4 / 6), abs=1e-3
        )

    def test_engine_evaluator_matches_leaf_evaluator(self) -> None:
        net = small_net()
        engine = PlaneBatchedEngine(net, CPU, batch_size=4, batch_timeout_ms=5.0)
        engine.start()
        boards = [alpha_go_cpp.BoxesBoard(3) for _ in range(3)]
        boards[2].play_edge(5)
        via_engine = BoxesEngineEvaluator(engine).batch_evaluate(boards)
        direct = BoxesLeafEvaluator(net, CPU).batch_evaluate(boards)
        engine.stop()
        for (pe, ve), (pd, vd) in zip(via_engine, direct):
            assert pe.keys() == pd.keys() and ve == pytest.approx(vd, abs=1e-5)
            for e in pe:
                assert pe[e] == pytest.approx(pd[e], abs=1e-5)
        assert engine.total_requests == 3 and engine.footprint_bytes() == 4 * 11 * 49 * 4


class TestAgent:
    def test_registered_untrained_agent_plays_a_full_game(self) -> None:
        record = play_game(
            get_agent("boxes-mcts-untrained-3x3"), get_agent("boxes-greedy"), board_size=3,
            seed=0, max_moves=24, game=get_game("boxes"), collect_metrics=True,
        )
        assert record.num_moves == 24 and record.termination == "all_edges"
        first = record.move_metrics[0]
        assert first.visit_counts is not None and first.visit_counts.shape == (24,)
        assert first.visit_counts.sum() == 32  # leaf-batched: every simulation visits a child
        assert first.root_value is not None and 0.0 <= first.root_value <= 1.0
        assert record.move_metrics[1].visit_counts is None  # greedy has no search

    def test_agent_finds_forced_wins_with_search_only(self) -> None:
        """Random net, terminal rewards only: the search still finds the exact best move."""
        agent = BoxesMCTSAgent(BoxesLeafEvaluator(small_net(2, 2), CPU), num_simulations=400,
                               temperature=0.0, leaf_batch_size=4)
        oracle = Oracle(2, 2)
        for seed in range(6):
            py, cpp = BoxesBoard(2, 2), alpha_go_cpp.BoxesBoard(2, 2)
            rng = np.random.default_rng(seed)
            while cpp.num_edges() - cpp.move_count() > 3:
                e = int(rng.choice(cpp.get_legal_moves_flat()))
                py.play_edge(e)
                cpp.play_edge(e)
            edge = cpp.edge_index(*agent.select_move(cpp, seed))
            outcomes = {e: np.sign(py.margin() + oracle.child_value(py.edges, e))
                        for e in py.get_legal_moves_flat()}
            assert outcomes[edge] == max(outcomes.values())

    def test_temperature_cutoff_and_resign(self) -> None:
        evaluator = BoxesLeafEvaluator(small_net(), CPU)
        agent = BoxesMCTSAgent(evaluator, num_simulations=8, temperature=1.0,
                               temperature_cutoff=1, resign_threshold=0.01, resign_consec_turns=1)
        board = alpha_go_cpp.BoxesBoard(3)
        assert board.is_legal(*agent.select_move(board, 0))  # win prob ~0.5 is above 0.01
        board.play_edge(0)
        assert board.is_legal(*agent.select_move(board, 1))  # past the cutoff: argmax
        quitter = BoxesMCTSAgent(evaluator, num_simulations=8, resign_threshold=0.9,
                                 resign_consec_turns=2)
        quitter.start_game(3)
        assert board.is_legal(*quitter.select_move(board, 2))  # first turn below threshold
        assert quitter.select_move(board, 3) == (-2, -2)  # second consecutive: resign

    def test_checkpoint_round_trip_and_registration(self, tmp_path: Path) -> None:
        net = small_net(2, 3)
        path = tmp_path / "net.pt"
        save_boxes_net(net, path, iteration=3)
        loaded = load_boxes_net(path, CPU)
        assert loaded.rows == 2 and loaded.cols == 3 and loaded.channels == 16
        planes = torch.zeros(1, 11, 5, 7)
        assert torch.allclose(net(planes)[0], loaded(planes)[0])
        name = register_boxes_mcts_agent("boxes-mcts-test-2x3", path, 2, 3, device="cpu",
                                         num_simulations=4)
        agent = get_agent(name)
        assert agent.checkpoint_path == str(path)
        board = alpha_go_cpp.BoxesBoard(2, 3)
        assert board.is_legal(*agent.select_move(board, 0))

    def test_search_flags_reach_the_agent_and_its_evaluator(self, tmp_path: Path) -> None:
        import argparse
        path = tmp_path / "net.pt"
        save_boxes_net(small_net(2, 3), path)
        parser = argparse.ArgumentParser()
        add_search_flags(parser)
        defaults = search_flags(parser.parse_args([]))
        assert defaults == ({"c_puct": 1.5, "leaf_batch_size": 16},
                            {"policy_temperature": 1.0, "margin_utility_lambda": 0.0,
                             "margin_utility_k": 6.0})
        mcts, evaluator = search_flags(parser.parse_args(
            ["--c_puct", "2.5", "--leaf_batch_size", "4", "--policy_temperature", "0.7",
             "--margin_utility_lambda", "0.75", "--margin_utility_k", "3"]))
        name = register_boxes_mcts_agent("boxes-mcts-test-flags", path, 2, 3, device="cpu",
                                         num_simulations=4, evaluator_kwargs=evaluator, **mcts)
        agent = get_agent(name)
        assert (agent.cpp_config.c_puct, agent.leaf_batch_size) == (2.5, 4)
        assert (agent.evaluator.policy_temperature, agent.evaluator.margin_utility_lambda,
                agent.evaluator.margin_utility_k) == (0.7, 0.75, 3.0)
        board = alpha_go_cpp.BoxesBoard(2, 3)
        assert board.is_legal(*agent.select_move(board, 0))


class TestDataset:
    def test_samples_from_searched_and_unsearched_games(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(self_play, "GAME_DATA_DIR", tmp_path)
        monkeypatch.setattr(sys, "argv", [
            "self_play", "--game", "boxes", "--board_size", "3",
            "--black", "boxes-mcts-untrained-3x3", "--white", "boxes-greedy",
            "--num_games", "1", "--save-name", "searched", "--seed", "0", "--collect-metrics",
        ])
        self_play.main()
        record = play_game(get_agent("boxes-random"), get_agent("boxes-random"), board_size=3,
                           seed=5, max_moves=24, game=get_game("boxes"))
        (tmp_path / "plain").mkdir()
        save_game_data(record, tmp_path / "plain", 0, "t")
        ds = BoxesDataset([tmp_path / "searched", tmp_path / "plain"])
        assert len(ds) == 48 and ds.num_with_mcts == 1 and ds.footprint_bytes() > 0
        searched, plain = ds[0], ds[47]
        assert searched["planes"].shape == (11, 7, 7) and searched["policy"].shape == (24,)
        assert searched["has_mcts"] and searched["policy"].sum() == pytest.approx(1.0)
        # The policy target is the normalised visit distribution, whatever the sampling
        # temperature was (the first game's first position is searched by the NN agent).
        visits = ds.games[0]["mcts_visits"][0].astype(float)
        assert np.allclose(searched["policy"].numpy(), visits / visits.sum())
        # Positions of the searched game that were played without a search (the greedy
        # opponent's moves, forced captures) carry a zero visit row: their target is the
        # played move, label-smoothed, never a zero vector or a one-hot at edge 0.
        game = ds.games[0]
        unsearched = [i for i in range(int(game["num_moves"])) if game["mcts_visits"][i].sum() == 0]
        assert unsearched
        for i in unsearched[:5]:
            sample = ds[i]
            r, c = (int(x) for x in game["moves"][i])
            assert not sample["has_mcts"]
            assert sample["policy"].argmax().item() == int(ds.geo.lattice_edge[r, c])
            assert sample["policy"].max() == pytest.approx(0.9 + 0.1 / 24)
        assert not plain["has_mcts"] and plain["policy"].max() == pytest.approx(0.9 + 0.1 / 24)
        # Margin target is the final box difference for the side to move at that position.
        score_p1 = float(record.result[2:]) * (1 if record.result.startswith("B") else -1)
        assert plain["margin"].item() == (score_p1 if record.to_play[-1] == 0 else -score_p1)
        assert plain["win"].item() == float(plain["margin"].item() > 0)
        # Held-out split by game: one game each way, positions never straddle the sets.
        train, val = ds.split(0.5, seed=1)
        assert len(train.games) == 1 and len(val.games) == 1
        assert len(train) + len(val) == 48 and train.games[0] is not val.games[0]
        empty_train, _ = ds.split(0.0)
        assert len(empty_train) == 48
