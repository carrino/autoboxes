"""The MCTS backup must flip the value only when the mover changes.

Go alternates every ply, so an unconditional `1 - value` flip is correct there. Dots and
Boxes gives the mover another turn after a capture; this toy game has the same property
and checks that a two-capture win is scored as a win for the player who makes it.
"""
from __future__ import annotations

from dataclasses import dataclass

from alpha_go.mcts import MCTSConfig, Node, get_action_probabilities, run_mcts

GRAB = 0  # take a token and move again
HAND_OVER = 1  # take nothing and pass the turn


@dataclass
class GrabGame:
    """Two tokens on the table; a grab keeps the turn, so grab-grab wins 2-0."""

    tokens: int = 2
    taken: tuple[int, int] = (0, 0)
    current: int = 0

    def get_legal_actions(self) -> list[int]:
        return [] if self.tokens == 0 else [GRAB, HAND_OVER]

    def apply_action(self, action: int) -> GrabGame:
        if action == HAND_OVER:
            return GrabGame(self.tokens, self.taken, 1 - self.current)
        taken = list(self.taken)
        taken[self.current] += 1
        return GrabGame(self.tokens - 1, (taken[0], taken[1]), self.current)

    def is_terminal(self) -> bool:
        return self.tokens == 0

    def get_reward(self, player: int) -> float:
        mine, theirs = self.taken[player], self.taken[1 - player]
        return 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0

    def current_player(self) -> int:
        return self.current

    def clone(self) -> GrabGame:
        return GrabGame(self.tokens, self.taken, self.current)


def uniform_policy_and_value(state: GrabGame) -> tuple[dict[int, float], float]:
    actions = state.get_legal_actions()
    return {a: 1.0 / len(actions) for a in actions}, 0.5


def _search() -> Node[int]:
    return run_mcts(GrabGame(), 400, MCTSConfig(c_puct=1.0), uniform_policy_and_value)


class TestExtraMoveBackup:
    def test_same_mover_keeps_sign(self) -> None:
        root = _search()
        grab = root.children[GRAB]
        # Player 0 grabs twice and wins 2-0. Both nodes on that line are player 0's moves,
        # so the terminal reward 1.0 must reach `grab` without a flip.
        assert grab.children[GRAB].Q == 1.0
        assert grab.Q > 0.8
        assert get_action_probabilities(root, temperature=0)[GRAB] == 1.0

    def test_mover_change_flips_sign(self) -> None:
        root = _search()
        hand_over = root.children[HAND_OVER]
        grab = hand_over.children[GRAB]
        # Player 1 grabs twice after the hand-over and wins 2-0: scored 1.0 for player 1
        # on its own line (no flip), and flipped once when backed up to player 0's node.
        assert grab.children[GRAB].Q == 1.0
        assert grab.Q > 0.8
        assert hand_over.Q < 0.5
