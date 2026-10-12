"""MCTS must not flip the value when a capture keeps the turn, and must agree with the oracle.

Hand-built position on 1x2 after edges 0,1,2,3,4: PLAYER_2 to move. Playing edge 5
captures box A, keeps the turn, and edge 6 then captures box B (2-0 win). Playing edge 6
first lets PLAYER_1 take both boxes with edge 5. A search that negates the value at every
ply scores the capture line as a loss and picks edge 6.
"""
from __future__ import annotations

import random

import numpy as np

from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard, BoxesState
from alpha_go.mcts import MCTSConfig, get_action_probabilities, run_mcts


def uniform_policy_and_value(state: BoxesState) -> tuple[dict[int, float], float]:
    actions = state.get_legal_actions()
    return {a: 1.0 / len(actions) for a in actions}, 0.5


def capture_position() -> BoxesState:
    board = BoxesBoard(1, 2)
    for e in (0, 1, 2, 3, 4):
        board.play_edge(e)
    return BoxesState(board)


class TestPythonMCTSSign:
    def test_capture_line_backs_up_without_flip(self) -> None:
        root = run_mcts(capture_position(), 200, MCTSConfig(c_puct=1.0), uniform_policy_and_value)
        take, decline = root.children[5], root.children[6]
        assert take.children[6].Q == 1.0  # terminal 2-0 for PLAYER_2, scored for PLAYER_2
        assert take.Q > 0.9  # same mover, no flip on the way up
        assert decline.children[5].Q == 1.0  # PLAYER_1 takes both: 1.0 for PLAYER_1 ...
        assert decline.Q < 0.2  # ... flipped to 0.0 for PLAYER_2 (its first visit averaged in 0.5)
        assert get_action_probabilities(root, temperature=0)[5] == 1.0

    def test_agrees_with_oracle_near_the_end(self) -> None:
        # The search optimises win probability, so it must pick a move whose exact
        # win/draw/loss outcome is the best available; it need not maximise the margin.
        oracle = Oracle(2, 2)
        rng = random.Random(5)
        for _ in range(30):
            board = BoxesBoard(2, 2)
            for _ in range(board.num_edges() - rng.randint(2, 4)):
                board.play_edge(rng.choice(board.get_legal_moves_flat()))
            root = run_mcts(
                BoxesState(board), 400, MCTSConfig(c_puct=1.0), uniform_policy_and_value
            )
            probs = get_action_probabilities(root, temperature=0)
            chosen = max(probs, key=lambda a: probs[a])
            outcomes = {
                e: np.sign(board.margin() + oracle.child_value(board.edges, e))
                for e in board.get_legal_moves_flat()
            }
            assert outcomes[chosen] == max(outcomes.values()), (board.render(), chosen, outcomes)


class TestPythonProofPropagation:
    """`prove_terminals`: exact leaves back up by minimax and settle the root."""

    def test_capture_line_is_proven(self) -> None:
        cfg = MCTSConfig(c_puct=1.0, prove_terminals=True)
        root = run_mcts(capture_position(), 50, cfg, uniform_policy_and_value)
        take = root.children[5]
        assert (take.proven, take.proven_value, take.Q) == (True, 1.0, 1.0)
        # The root is proven the moment its winning line is, before edge 6 is ever looked at:
        # a loss for the opponent, who "moved" into it.
        assert root.proven and root.Q == 0.0
        probs = get_action_probabilities(root, temperature=1.0)
        assert probs[5] == 1.0 and probs.get(6, 0.0) == 0.0

    def test_exact_on_late_positions(self) -> None:
        # Two to four undrawn edges on 2x2: at most 65 nodes, so the tree is exhausted and the
        # root's value is the oracle's; only optimal edges keep probability mass.
        oracle = Oracle(2, 2)
        rng = random.Random(11)
        cfg = MCTSConfig(c_puct=1.0, prove_terminals=True)
        for _ in range(30):
            board = BoxesBoard(2, 2)
            for _ in range(board.num_edges() - rng.randint(2, 4)):
                board.play_edge(rng.choice(board.get_legal_moves_flat()))
            root = run_mcts(BoxesState(board), 600, cfg, uniform_policy_and_value)
            outcomes = {
                e: np.sign(board.margin() + oracle.child_value(board.edges, e))
                for e in board.get_legal_moves_flat()
            }
            best = max(outcomes.values())
            assert root.proven, board.render()
            assert max(c.proven_value for c in root.children.values() if c.proven) == (best + 1) / 2
            probs = get_action_probabilities(root, temperature=1.0)
            kept = [a for a, p in probs.items() if p > 0]
            assert all(outcomes[a] == best for a in kept), (board.render(), probs)

    def test_flag_off_is_unchanged(self) -> None:
        on = run_mcts(capture_position(), 50, MCTSConfig(c_puct=1.0), uniform_policy_and_value)
        assert not on.proven and not any(c.proven for c in on.children.values())
