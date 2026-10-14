"""Conversion, forced-turn, and optional upstream-source parity checks."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from alpha_go.boxes.boxeszero_worker import (
    PublishedSearch,
    canonical_board,
    create_search,
    lattice_action,
    load_upstream,
    upstream_position_key,
)
from alpha_go.boxes.rules import BoxesBoard


def _request(board: BoxesBoard, last: tuple[int, int] | None = None) -> dict[str, Any]:
    return {
        "board": board.to_numpy().tolist(),
        "to_play": board.to_play(),
        "last_move": None if last is None else list(last),
        "seed": 103,
    }


class TestBoardConversion:
    def test_every_edge_coordinate_and_extra_turns(self) -> None:
        board = BoxesBoard(5)
        rng = random.Random(103)
        captures = 0
        for _ in range(60):
            old_player = board.to_play()
            old_score = sum(board.boxes)
            edge = rng.choice(board.get_legal_moves_flat())
            row, col = board.row_col(edge)
            assert board.play_edge(edge)
            converted = canonical_board(board.to_numpy(), board.to_play()).reshape(11, 11)
            assert lattice_action([row, col], converted.reshape(-1)) == row * 11 + col
            for idx, (r, c) in enumerate(board.geo.edge_rc):
                assert converted[r, c] == (0 if board.is_legal_edge(idx) else 2)
            mine = np.count_nonzero(converted == 1)
            theirs = np.count_nonzero(converted == -1)
            assert mine - theirs == board.margin()
            captured = sum(board.boxes) > old_score
            assert (board.to_play() == old_player) == captured
            captures += captured
        assert captures > 0

    def test_both_player_perspectives(self) -> None:
        board = BoxesBoard(5)
        for edge in range(60):
            board.play_edge(edge)
        a = canonical_board(board.to_numpy(), 1).reshape(11, 11)
        b = canonical_board(board.to_numpy(), 2).reshape(11, 11)
        np.testing.assert_array_equal(a[1::2, 1::2], -b[1::2, 1::2])
        np.testing.assert_array_equal(a[::2], b[::2])

    @pytest.mark.parametrize("cell,value", [((0, 0), 1), ((0, 1), 2), ((1, 1), 3)])
    def test_rejects_bad_cell_values(self, cell: tuple[int, int], value: int) -> None:
        board = np.zeros((11, 11), dtype=np.int8)
        board[cell] = value
        with pytest.raises(ValueError):
            canonical_board(board, 1)

    def test_rejects_missing_or_invented_capture(self) -> None:
        board = np.zeros((11, 11), dtype=np.int8)
        board[1, 1] = 1
        with pytest.raises(ValueError, match="fourth sides"):
            canonical_board(board, 1)
        board[1, 1] = 0
        board[0, 1] = board[2, 1] = board[1, 0] = board[1, 2] = 1
        with pytest.raises(ValueError, match="fourth sides"):
            canonical_board(board, 1)

    @pytest.mark.parametrize("move", [[0, 0], [-1, 0], [0, 1], [0.0, 1], [0]])
    def test_rejects_invalid_last_move(self, move: object) -> None:
        with pytest.raises(ValueError):
            lattice_action(move, np.zeros(121, dtype=np.int8))

    def test_cache_key_preserves_upstream_byte_layout(self) -> None:
        board = np.zeros(121, dtype=np.int8)
        board[[1, 11, 13]] = [2, -2, 2]
        board[[12, 14, 16]] = [1, 1, -1]
        expected = np.append(np.abs(board), 1).tobytes()
        assert upstream_position_key(board) == expected
        assert upstream_position_key(-board) != expected


class _ReferenceGame:
    """Use the existing host rules to validate queued upstream moves."""

    GAME_NOT_END = 0

    def __init__(self, boards: list[BoxesBoard]) -> None:
        self.boards = {canonical_board(b.to_numpy(), b.to_play()).tobytes(): b for b in boards}

    def _board(self, board: Any) -> BoxesBoard:
        return self.boards[np.asarray(board, dtype=np.int8).tobytes()].copy()

    def get_legal_action_dots(self, board: Any) -> list[int]:
        host = self._board(board)
        return [r * 11 + c for r, c in map(host.row_col, host.get_legal_moves_flat())]

    def game_over_dots(self, board: Any) -> int:
        return int(max(self._board(board).boxes) >= 13)

    def get_next_board(self, board: Any, action: int) -> tuple[bool, Any]:
        host = self._board(board)
        old_player = host.to_play()
        assert host.play(*divmod(action, 11))
        converted = canonical_board(host.to_numpy(), host.to_play())
        self.boards[converted.tobytes()] = host
        return host.to_play() == old_player, converted


class _ScriptedMcts:
    def __init__(self, scripts: list[Any]) -> None:
        self.scripts = list(scripts)
        self.calls: list[int] = []
        self.node_info = {"sentinel": 1}
        self.root: Any = None

    def search(self, num_iter: int, board: Any, layers: int, action: int) -> None:
        self.calls.append(action)
        self.root = self.scripts.pop(0)


def _root(quick: list[int], *, fake: int | None = None, best: int = 3) -> Any:
    visits = np.zeros(121, dtype=np.int32)
    visits[best] = 4
    return SimpleNamespace(root_quick_move=quick, fake_last_move=fake, child_N=visits)


class TestForcedCaptureProtocol:
    def test_preserves_sequence_and_fake_last_move(self) -> None:
        board = BoxesBoard(5)
        for edge in [0, 5, 30, 2, 7, 32]:
            board.play_edge(edge)
        # Two independent fourth sides; both moves must keep the same player.
        mcts = _ScriptedMcts([_root([13, 17], fake=15), _root([])])
        search = PublishedSearch(_ReferenceGame([board]), mcts)
        first = search.select(_request(board, (1, 4)))
        assert first["move"] == [1, 2]
        assert board.play(*first["move"])
        second = search.select(_request(board, (1, 2)))
        assert second["move"] == [1, 6]
        assert len(mcts.calls) == 1
        assert board.play(*second["move"])
        search.select(_request(board, (1, 6)))
        assert mcts.calls[-1] == 15
        search.reset()
        assert not search.pending and search.fake_last_move is None
        assert not mcts.node_info

    def test_does_not_reuse_sequence_after_board_changes(self) -> None:
        board = BoxesBoard(5)
        for edge in [0, 5, 30, 2, 7, 32]:
            board.play_edge(edge)
        other = BoxesBoard(5)
        mcts = _ScriptedMcts([_root([13, 17], fake=15), _root([], best=1)])
        search = PublishedSearch(_ReferenceGame([board, other]), mcts)
        search.select(_request(board, (1, 4)))
        result = search.select(_request(other))
        assert result["selection"] == "search"
        assert mcts.calls == [15, -1]

    def test_rejects_non_capture_in_forced_sequence(self) -> None:
        board = BoxesBoard(5)
        search = PublishedSearch(_ReferenceGame([board]), _ScriptedMcts([_root([1])]))
        with pytest.raises(ValueError, match="changes player"):
            search.select(_request(board))

    def test_decided_game_completion_never_calls_search(self) -> None:
        board = BoxesBoard(5)
        rng = random.Random(7)
        while max(board.boxes) < 13:
            board.play_edge(rng.choice(board.get_legal_moves_flat()))
        assert not board.is_game_over()
        mcts = _ScriptedMcts([])
        search = PublishedSearch(_ReferenceGame([board]), mcts)
        result = search.select(_request(board))
        assert result["selection"] == "decided_completion"
        assert board.is_legal(*result["move"])
        assert not mcts.calls


@pytest.fixture(scope="module")
def upstream_root() -> Path:
    supplied = os.environ.get("BOXESZERO_ROOT")
    if not supplied:
        pytest.skip("Set BOXESZERO_ROOT to the reviewed, pinned external checkout")
    return Path(supplied).resolve()


@pytest.fixture(scope="module")
def synthetic_checkpoint(upstream_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    args, _, classes, _ = load_upstream(upstream_root)
    args.input_channels = 4
    torch.manual_seed(19)
    network = classes[0](args)
    destination = tmp_path_factory.mktemp("boxeszero") / "synthetic.pt"
    torch.save({"state_dict": network.state_dict()}, destination)
    return destination


class TestPublishedSource:
    @pytest.mark.parametrize("channels", [2, 26])
    def test_other_published_encodings_preserve_checkpoint(
        self, upstream_root: Path, tmp_path: Path, channels: int
    ) -> None:
        args, _, classes, _ = load_upstream(upstream_root)
        args.input_channels = channels
        network = classes[0](args)
        checkpoint = tmp_path / "synthetic.pt"
        weights = network.state_dict()
        torch.save({"state_dict": weights}, checkpoint)
        search, metadata = create_search(upstream_root, checkpoint, 3, "cpu", 1)
        loaded = search.mcts.trainer.net_work.state_dict()
        for key, expected in weights.items():
            torch.testing.assert_close(loaded[key], expected, rtol=0, atol=0)
        result = search.select(_request(BoxesBoard(5)))
        assert BoxesBoard(5).is_legal(*result["move"])
        assert metadata["input_channels"] == channels

    def test_rules_parity_through_random_games(self, upstream_root: Path) -> None:
        args, game_class, _, _ = load_upstream(upstream_root)
        game = game_class(args)
        for seed in range(8):
            board = BoxesBoard(5)
            rng = random.Random(seed)
            for _ in range(60):
                canonical = canonical_board(board.to_numpy(), board.to_play())
                edge = rng.choice(board.get_legal_moves_flat())
                row, col = board.row_col(edge)
                same, actual = game.get_next_board(canonical, row * 11 + col)
                player = board.to_play()
                board.play_edge(edge)
                assert bool(same) == (board.to_play() == player)
                expected = canonical_board(board.to_numpy(), board.to_play()).reshape(11, 11)
                actual = actual.reshape(11, 11)
                np.testing.assert_array_equal(actual[1::2, 1::2], expected[1::2, 1::2])
                np.testing.assert_array_equal(np.abs(actual), np.abs(expected))

    def test_search_full_game_with_synthetic_weights(
        self, upstream_root: Path, synthetic_checkpoint: Path
    ) -> None:
        search, metadata = create_search(upstream_root, synthetic_checkpoint, 4, "cpu", 1)
        board = BoxesBoard(5)
        last = None
        saw_search = False
        while not board.is_game_over():
            result = search.select(_request(board, last))
            last = tuple(result["move"])
            assert board.play(*last)
            saw_search |= result["selection"] == "search"
        assert saw_search and metadata["input_channels"] == 4

    def test_stdout_is_json_only(self, upstream_root: Path, synthetic_checkpoint: Path) -> None:
        requests = [_request(BoxesBoard(5)), {"command": "reset"}, {"command": "close"}]
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "alpha_go.boxes.boxeszero_worker",
                "--root",
                str(upstream_root),
                "--checkpoint",
                str(synthetic_checkpoint),
                "--simulations",
                "4",
                "--threads",
                "1",
            ],
            input="".join(json.dumps(request) + "\n" for request in requests),
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
        outputs = [json.loads(line) for line in completed.stdout.splitlines()]
        assert len(outputs) == 3
        assert outputs[0]["ready"] is True
        assert BoxesBoard(5).is_legal(*outputs[1]["move"])
        assert outputs[2] == {"reset": True}
        assert "总结点数" in completed.stderr
