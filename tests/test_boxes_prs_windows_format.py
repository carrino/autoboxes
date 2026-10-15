"""State conversion tests. These do not validate the Windows GUI bridge."""

import importlib.util
import random
from pathlib import Path

import pytest

from alpha_go.boxes.rules import BoxesBoard

_worker = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "2026-10-14_19-52-boxes-external-probe"
    / "windows"
    / "prsboxes_windows_smoke.py"
)
_spec = importlib.util.spec_from_file_location("prsboxes_windows_smoke", _worker)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
parse_position = _module.parse_position
position_text = _module.position_text
replay = _module.replay


def native_header_width(row):
    """Characterize the pinned PE loader's first-row branch at 0x406c26..0x406cab.

    Its line reader retains LF. Normal loop exhaustion leaves the constructor's
    12-dot width unchanged; a non-edge after the last dot commits the real width.
    This intentionally differs from our parser, which receives explicit dimensions.
    """
    line = row + "\n"
    dots = (len(line) + 1) // 2  # strlen including the terminating NUL, divided by two
    for x in range(dots):
        if line[2 * x] != "+":
            return x
        if x < dots - 1 and line[2 * x + 1] not in " -=":
            return x + 1
    return 12


def test_unterminated_header_reproduces_native_width_bug():
    assert native_header_width("+ + +-+ +-+") == 12
    assert native_header_width("+ + +-+ +-+#The first row determines width") == 6


@pytest.mark.parametrize("rows,cols", [(1, 1), (2, 3), (3, 3), (5, 5)])
def test_exporter_sets_native_width(rows, cols):
    row = next(line for line in position_text(rows, cols, []).splitlines()
               if line.startswith("+"))
    assert native_header_width(row) == cols + 1


@pytest.mark.parametrize("rows,cols", [(1, 1), (2, 3), (3, 3), (5, 5)])
def test_replay_and_text_agree_with_boxesboard_at_every_ply(rows, cols):
    rng = random.Random(20261015)
    capture_count = 0
    double_capture_count = 0
    for _ in range(20):
        board = BoxesBoard(rows, cols)
        order = list(range(board.num_edges()))
        rng.shuffle(order)
        prefix = []
        for edge in order:
            before = sum(board.boxes)
            assert board.play_edge(edge)
            capture_count += sum(board.boxes) > before
            double_capture_count += sum(board.boxes) - before == 2
            prefix.append(edge)
            edges, owners, mover = replay(rows, cols, prefix)
            assert sum(1 << e for e in edges) == board.edges
            assert owners == board.owner
            assert mover == board.to_play()
            text_edges, text_owners = parse_position(position_text(rows, cols, prefix), rows, cols)
            assert text_edges == edges
            assert text_owners == [
                " " if owner == 0 else "C" if owner == mover else "H" for owner in board.owner
            ]
    assert capture_count > 0
    if rows * cols > 1:
        assert double_capture_count > 0


def test_native_comment_and_computer_line_characters():
    text = ("+=+ #The first row determines width\n|C|\n+-+\n"
            "Anything illegal in column one determines height.\n")
    edges, owners = parse_position(text, 1, 1)
    assert edges == {0, 1, 2, 3}
    assert owners == ["C"]


def test_duplicate_edge_rejected():
    with pytest.raises(AssertionError, match="repeated edge"):
        replay(2, 2, [0, 0])
