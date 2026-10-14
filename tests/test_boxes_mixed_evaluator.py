"""Policy/value routing works with different feature sets and real game search."""
from __future__ import annotations

import alpha_go_cpp  # type: ignore[import-not-found]
import pytest
import torch

from alpha_go.agents import get_agent
from alpha_go.boxes.mixed_evaluator import MixedLeafEvaluator
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import BoxesLeafEvaluator, BoxesMCTSAgent
from alpha_go.gameplay import play_game
from alpha_go.games import get_game

CPU = torch.device("cpu")


def nets() -> tuple[BoxesNet, BoxesNet]:
    torch.manual_seed(14)
    first = BoxesNet(2, 2, channels=8, n_blocks=1, value_hidden=8).eval()
    second = BoxesNet(2, 2, channels=8, n_blocks=1, value_hidden=8, features="chains").eval()
    with torch.no_grad():
        first.policy_conv.weight.normal_()
        second.policy_conv.weight.normal_()
        first.value_fc2.bias[0] = 10
        second.value_fc2.bias[-1] = 10
    return first, second


def test_policy_and_value_come_from_requested_complete_networks() -> None:
    first, second = nets()
    boards = [alpha_go_cpp.BoxesBoard(2, 2), alpha_go_cpp.BoxesBoard(2, 2)]
    boards[1].play_edge(0)
    kwargs = {"policy_temperature": 0.7, "margin_utility_lambda": 0.5}
    old = BoxesLeafEvaluator(first, CPU, **kwargs).batch_evaluate(boards)
    new = BoxesLeafEvaluator(second, CPU, **kwargs).batch_evaluate(boards)
    mixed = MixedLeafEvaluator(first, second, CPU, **kwargs).batch_evaluate(boards)
    reverse = MixedLeafEvaluator(second, first, CPU, **kwargs).batch_evaluate(boards)
    for a, b, ab, ba in zip(old, new, mixed, reverse):
        assert a[0] != b[0] and a[1] != b[1]
        assert ab[0] == a[0] and ab[1] == pytest.approx(b[1])
        assert ba[0] == b[0] and ba[1] == pytest.approx(a[1])
    same = MixedLeafEvaluator(first, first, CPU, **kwargs).batch_evaluate(boards)
    assert same == old


def test_mixed_evaluator_runs_through_shared_gameplay() -> None:
    first, second = nets()
    learner = BoxesMCTSAgent(MixedLeafEvaluator(first, second, CPU),
                            num_simulations=16, temperature=0, leaf_batch_size=4)
    game = play_game(learner, get_agent("boxes-greedy"), board_size=2, seed=10,
                     max_moves=12, game=get_game("boxes"), collect_metrics=True)
    assert game.num_moves == 12 and game.termination == "all_edges"
    assert game.move_metrics[0].visit_counts is not None


def test_mismatched_boards_fail_before_search() -> None:
    first, _ = nets()
    other = BoxesNet(1, 2, channels=8, n_blocks=1, value_hidden=8)
    with pytest.raises(ValueError, match="board size"):
        MixedLeafEvaluator(first, other, CPU)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(bad: float) -> None:
    first, second = nets()
    with pytest.raises(ValueError, match="settings"):
        MixedLeafEvaluator(first, second, CPU, policy_temperature=bad)
