"""Chain and loop analysis tests on hand-built positions."""
from __future__ import annotations

from alpha_go.boxes.chains import GROUND, OPEN, analyse, degrees
from alpha_go.boxes.rules import BoxesBoard, geometry


def board_with(rows: int, cols: int, drawn: list[int]) -> BoxesBoard:
    board = BoxesBoard(rows, cols)
    for e in drawn:
        board.play_edge(e)
    return board


class TestComponents:
    def test_empty_board_has_no_chains(self) -> None:
        assert analyse(BoxesBoard(3)) == []

    def test_one_by_three_open_chain(self) -> None:
        # 1x3: all border edges drawn except none; interior edges v(0,1)=? Build a 3-chain:
        # draw every border edge -> each box has degree 2 (interior sides) except the ends
        # have degree 1 (one interior side each).
        geo = geometry(1, 3)
        border = [e for e in range(geo.num_edges) if len(geo.edge_boxes[e]) == 1]
        board = board_with(1, 3, border)
        assert degrees(board.edges, geo) == [1, 2, 1]
        [comp] = analyse(board)
        assert comp.boxes in ([0, 1, 2], [2, 1, 0]) and not comp.is_loop
        assert comp.ends == (OPEN, OPEN) and comp.opened

    def test_chain_between_ground_and_junction(self) -> None:
        # 2x2: box 0 degree 2 (left and top undrawn), connected to ground on both? Draw
        # box 0's right and bottom -> box 0 has top+left undrawn: a 1-box chain ground-ground.
        geo = geometry(2, 2)
        top, bottom, left, right = geo.box_edges[0]
        board = board_with(2, 2, [right, bottom])
        comps = analyse(board)
        chain = next(c for c in comps if 0 in c.boxes)
        assert chain.boxes == [0] and chain.ends == (GROUND, GROUND) and not chain.opened

    def test_loop(self) -> None:
        # 2x2 with all 8 border edges drawn: the four boxes form a 4-loop through the
        # 4 interior edges.
        geo = geometry(2, 2)
        border = [e for e in range(geo.num_edges) if len(geo.edge_boxes[e]) == 1]
        board = board_with(2, 2, border)
        [loop] = analyse(board)
        assert loop.is_loop and sorted(loop.boxes) == [0, 1, 2, 3] and not loop.opened
        # Open it: draw one interior edge -> an opened 4-chain with two open ends.
        board.play_edge(geo.box_edges[0][3])  # right side of box 0 (shared with box 1)
        [opened] = analyse(board)
        assert not opened.is_loop and opened.opened and opened.size == 4
        assert opened.ends == (OPEN, OPEN)

    def test_cpp_board_is_accepted(self) -> None:
        import alpha_go_cpp

        geo = geometry(1, 3)
        border = [e for e in range(geo.num_edges) if len(geo.edge_boxes[e]) == 1]
        cpp = alpha_go_cpp.BoxesBoard(1, 3)
        for e in border:
            cpp.play_edge(e)
        [comp] = analyse(cpp)
        assert comp.size == 3 and comp.opened
