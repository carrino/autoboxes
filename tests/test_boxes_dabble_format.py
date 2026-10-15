"""Validate the genuine Dabble bridge independently of its Windows runtime."""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from alpha_go.boxes.rules import BoxesBoard

WORKER = (Path(__file__).resolve().parents[1] / "experiments" /
          "2026-10-14_19-52-boxes-external-probe" / "windows" / "dabble_worker.py")
SPEC = importlib.util.spec_from_file_location("dabble_format_worker", WORKER)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def channel(prefix: list[int], rows: int, cols: int, stdout: bool) -> str:
    """Match the format strings recovered from the pinned real executable."""
    lines = []
    for index, edge in enumerate(prefix):
        x1, y1, x2, y2 = worker.coins(edge, rows, cols)
        label = "DABBLE MOVE: " if stdout else ""
        lines.append(f"{label}{index}: ({x1}, {y1}) - ({x2}, {y2})")
    return "\n".join(lines)+"\n"


class TestDabbleState:
    def test_coordinates_round_trip(self) -> None:
        for rows, cols in [(1, 1), (1, 2), (2, 3), (3, 2), (5, 5)]:
            for edge in range((rows+1)*cols+rows*(cols+1)):
                c = worker.coins(edge, rows, cols)
                assert worker.edge_from_coins(c, rows, cols) == edge
                assert worker.edge_from_coins((c[2], c[3], c[0], c[1]), rows, cols) == edge

    def test_random_prefixes_match_reference(self) -> None:
        rng = random.Random(17)
        captures = double_captures = 0
        for rows, cols in [(1, 2), (2, 3), (3, 2), (5, 5)]:
            for _ in range(4):
                board = BoxesBoard(rows, cols)
                prefix: list[int] = []
                while not board.is_game_over():
                    request = dict(rows=rows, cols=cols, prefix=prefix, to_play=board.to_play(),
                                   scores=board.boxes, board=board.to_numpy().tolist())
                    mover, scores, lattice = worker.replay(request)
                    assert mover == board.to_play() and scores == board.boxes
                    assert lattice == board.to_numpy().tolist()
                    before = sum(board.boxes)
                    edge = rng.choice(board.get_legal_moves_flat())
                    board.play_edge(edge)
                    gained = sum(board.boxes)-before
                    captures += gained > 0
                    double_captures += gained == 2
                    prefix.append(edge)
        assert captures > 0 and double_captures > 0

    def test_rejects_wrong_board_scores_or_mover(self) -> None:
        request = dict(rows=2, cols=2, prefix=[1])
        with pytest.raises(ValueError, match="board differs"):
            worker.replay({**request, "board": [[0]*5 for _ in range(5)]})
        with pytest.raises(ValueError, match="scores differ"):
            worker.replay({**request, "scores": [1, 0]})
        with pytest.raises(ValueError, match="mover differs"):
            worker.replay({**request, "to_play": 1})

    def test_nonterminal_capture_sequence(self) -> None:
        request = dict(rows=2, cols=2, prefix=[1, 0, 6, 8, 9, 2, 5, 4, 7, 10, 11])
        mover, scores, _ = worker.replay(request)
        assert (mover, scores) == (2, [2, 0])
        assert worker.savegame(request).startswith("3,3\n(2, 0) - (2, 1)\n")


class TestDabbleOutput:
    @pytest.mark.parametrize("stdout,log", [(True, True), (True, False), (False, True)])
    def test_real_format_channels(self, stdout: bool, log: bool) -> None:
        request = dict(rows=2, cols=2, prefix=[1, 0])
        history = [1, 0, 6]
        out = channel(history, 2, 2, True) if stdout else "diagnostic only\n"
        saved = channel(history, 2, 2, False) if log else ""
        assert worker.chosen_edge(out, saved, request) == 6

    def test_conflicting_choices_fail(self) -> None:
        request = dict(rows=2, cols=2, prefix=[1, 0])
        with pytest.raises(ValueError, match="disagree"):
            worker.chosen_edge(channel([1, 0, 6], 2, 2, True),
                               channel([1, 0, 7], 2, 2, False), request)

    def test_incomplete_channel_fails_even_when_other_is_valid(self) -> None:
        request = dict(rows=2, cols=2, prefix=[1, 0])
        with pytest.raises(ValueError, match="complete prefix"):
            worker.chosen_edge(channel([1], 2, 2, True),
                               channel([1, 0, 7], 2, 2, False), request)

    def test_no_move_fails(self) -> None:
        with pytest.raises(ValueError, match="missing a move"):
            worker.chosen_edge("", "", dict(rows=2, cols=2, prefix=[]))

    def test_first_new_edge_of_capture_turn(self) -> None:
        request = dict(rows=1, cols=3, prefix=[0, 1, 2, 3, 4, 6, 8])
        history = [*request["prefix"], 7, 5]
        assert worker.chosen_edge(channel(history, 1, 3, True), "", request) == 7

    def test_unpinned_executable_fails(self, tmp_path: Path) -> None:
        executable = tmp_path/"dabble.exe"
        executable.write_bytes(b"not the pinned executable")
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            worker.verify_executable(executable)
