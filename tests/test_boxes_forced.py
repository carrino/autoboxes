"""Forced-move collapse: hand-built remainders and minimax invariance against the oracle."""
from __future__ import annotations

import random

import alpha_go_cpp
import pytest

from alpha_go.boxes.forced import BoxesSearchState, collapse
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard, geometry


def with_edges(rows: int, cols: int, drawn: list[int], cpp: bool = False) -> object:
    board = alpha_go_cpp.BoxesBoard(rows, cols) if cpp else BoxesBoard(rows, cols)
    for e in drawn:
        assert board.play_edge(e)
    return board


def border(rows: int, cols: int) -> list[int]:
    geo = geometry(rows, cols)
    return [e for e in range(geo.num_edges) if len(geo.edge_boxes[e]) == 1]


class TestHandBuilt:
    @pytest.mark.parametrize("cpp", [False, True])
    def test_half_hearted_handout_offers_take_or_control(self, cpp: bool) -> None:
        # 1x2 after edges 0,1,2,3,4: A has 3 sides (capturable via 5), B has 2 (5 and 6).
        board = with_edges(1, 2, [0, 1, 2, 3, 4], cpp)
        c = collapse(board)
        assert c.prefix == [] and c.decision is not None
        assert c.decision.take == [5, 6] and c.decision.control == 6
        state = BoxesSearchState.from_board(board)
        assert state.get_legal_actions() == [5, 6]
        taken = state.apply_action(5)
        assert taken.is_terminal()
        assert (taken.board.boxes(1) if cpp else taken.board.boxes[1]) == 2
        declined = state.apply_action(6)  # opponent (player 0) auto-captures both via edge 5
        assert declined.is_terminal() and declined.get_reward(0) == 1.0

    def test_hard_hearted_handout_is_just_taken(self) -> None:
        # Both boxes have 3 sides and share the only undrawn edge: no control option.
        board = with_edges(1, 2, [0, 1, 2, 3, 4, 6])
        c = collapse(board)
        assert c.decision is None and c.prefix == [5] and c.board.is_game_over()

    def test_doubly_open_three_chain_is_taken_in_full(self) -> None:
        board = with_edges(1, 3, border(1, 3))
        c = collapse(board)
        assert c.decision is None and len(c.prefix) == 2 and c.board.is_game_over()

    def test_opened_four_loop_offers_middle_edge(self) -> None:
        geo = geometry(2, 2)
        board = with_edges(2, 2, border(2, 2) + [geo.box_edges[0][3]])  # open the loop at 0|1
        c = collapse(board)
        assert c.decision is not None and c.prefix == []
        assert len(c.decision.take) == 3  # 4 boxes with 3 edges (last edge takes two)
        assert c.decision.control == geo.box_edges[2][3]  # middle edge, shared by boxes 2 and 3
        after = BoxesSearchState.from_board(board).apply_action(c.decision.control)
        assert after.is_terminal()
        assert after.board.boxes[1 - c.board.player()] == 4  # the opponent took all four

    def test_long_chain_is_taken_down_to_two(self) -> None:
        # 1x4 with all border edges drawn except box 0's left side, then draw it: a
        # 4-chain opened at box 0 with box 3 ending at ground on the right.
        geo = geometry(1, 4)
        right_of_3 = geo.box_edges[3][3]
        board = with_edges(1, 4, [e for e in border(1, 4) if e != right_of_3])
        c = collapse(board)
        assert c.decision is not None
        assert len(c.prefix) == 2 and c.board.boxes[c.board.player()] == 2
        assert c.decision.control == right_of_3

    def test_normal_position_has_no_prefix(self) -> None:
        state = BoxesSearchState.from_board(BoxesBoard(3))
        assert state.get_legal_actions() == list(range(24)) and state.decision is None
        child = state.apply_action(0)
        assert child.current_player() == 1 and len(child.get_legal_actions()) == 23


class TestMinimaxInvariance:
    @pytest.mark.parametrize("rows,cols", [(1, 3), (2, 2), (2, 3)])
    def test_collapsed_tree_has_the_oracle_value(self, rows: int, cols: int) -> None:
        oracle = Oracle(rows, cols)
        memo: dict[tuple[int, int], int] = {}

        def value(state: BoxesSearchState) -> int:
            """Remaining margin for the side to move in the collapsed game."""
            if state.is_terminal():
                return 0
            key = (state.board.edges, state.current_player())
            if key in memo:
                return memo[key]
            best = -99
            for a in state.get_legal_actions():
                child = state.apply_action(a)
                gained = (child.board.boxes[state.current_player()]
                          - state.board.boxes[state.current_player()])
                lost = (child.board.boxes[1 - state.current_player()]
                        - state.board.boxes[1 - state.current_player()])
                rest = value(child)
                same = child.current_player() == state.current_player()
                best = max(best, gained - lost + (rest if same else -rest))
            memo[key] = best
            return best

        rng = random.Random(rows * 7 + cols)
        for _ in range(40):
            board = BoxesBoard(rows, cols)
            for _ in range(rng.randint(0, board.num_edges() - 1)):
                board.play_edge(rng.choice(board.get_legal_moves_flat()))
            c = collapse(board)
            me = board.player()
            prefix_gain = c.board.boxes[me] - board.boxes[me]
            state = BoxesSearchState(c.board, c.decision)
            assert prefix_gain + value(state) == oracle.value(board.edges), board.render()
