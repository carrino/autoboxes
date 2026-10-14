"""Isolated JSON-lines worker for the author's published BoxesZero source snapshot.

The upstream search and network run in their own process because they use generic
top-level module names. Their training/server entry points are never invoked.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

LATTICE_SIZE = 11
_ROW, _COL = np.indices((LATTICE_SIZE, LATTICE_SIZE))
_EDGES = (_ROW + _COL) % 2 == 1
_BOXES = (_ROW % 2 == 1) & (_COL % 2 == 1)
_DOTS = (_ROW % 2 == 0) & (_COL % 2 == 0)


def canonical_board(board: object, to_play: int) -> NDArray[np.int8]:
    """Convert a validated host lattice to upstream's current-player perspective."""
    grid = np.asarray(board)
    if grid.shape != (LATTICE_SIZE, LATTICE_SIZE) or grid.dtype.kind not in "iu":
        raise ValueError("BoxesZero requires an integer 11x11 lattice (5x5 boxes)")
    if to_play not in (1, 2):
        raise ValueError("to_play must be 1 or 2")
    if np.any(grid[_DOTS] != 0) or not np.isin(grid[_EDGES], (0, 1)).all():
        raise ValueError("Host dots must be zero and edges must be zero or one")
    if not np.isin(grid[_BOXES], (0, 1, 2)).all():
        raise ValueError("Host box owners must be zero, one or two")
    complete = (
        (grid[0:10:2, 1:10:2] != 0)
        & (grid[2:11:2, 1:10:2] != 0)
        & (grid[1:10:2, 0:10:2] != 0)
        & (grid[1:10:2, 2:11:2] != 0)
    )
    if not np.array_equal(complete, grid[1::2, 1::2] != 0):
        raise ValueError("Box ownership must agree with completed fourth sides")
    result = np.zeros(grid.shape, dtype=np.int8)
    result[_EDGES] = 2 * grid[_EDGES]
    result[_BOXES & (grid == to_play)] = 1
    result[_BOXES & (grid == 3 - to_play)] = -1
    return result.reshape(-1)


def lattice_action(move: object, board: NDArray[np.int8]) -> int:
    """Validate the last move supplied by the host and return its upstream index."""
    if move is None:
        return -1
    if not isinstance(move, list) or len(move) != 2 or any(type(x) is not int for x in move):
        raise ValueError("last_move must be null or an integer [row, col]")
    row, col = move
    if not (0 <= row < 11 and 0 <= col < 11) or (row + col) % 2 != 1:
        raise ValueError("last_move is not a lattice edge")
    action = row * LATTICE_SIZE + col
    if board[action] == 0:
        raise ValueError("last_move must already be drawn")
    return int(action)


def upstream_position_key(board: NDArray[np.int8]) -> bytes:
    """Preserve upstream's cache key with NumPy's supported tostring replacement."""
    grid = np.asarray(board).reshape(11, 11)
    margin = int(np.count_nonzero(grid == 1) - np.count_nonzero(grid == -1))
    return np.append(np.abs(grid).reshape(-1), margin).tobytes()


def load_upstream(root: Path) -> tuple[Any, Any, Any, Any]:
    """Import reviewed inference modules, using the matching ControlCenter settings."""
    center = root.resolve() / "ComputingCenter1"
    args_path = root.resolve() / "ControlCenter/game_args/dotsboxes_6.py"
    if not center.is_dir() or not args_path.is_file():
        raise ValueError("--root must contain ComputingCenter1 and ControlCenter source")
    # Keep the external, hash-verified source tree read-only even when this helper
    # is invoked directly rather than through the parent process's python -B.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(center))
    importlib.import_module("game_args")
    spec = importlib.util.spec_from_file_location("game_args.dotsboxes_6", args_path)
    assert spec is not None and spec.loader is not None
    args_module = importlib.util.module_from_spec(spec)
    sys.modules["game_args.dotsboxes_6"] = args_module
    spec.loader.exec_module(args_module)
    game_class = importlib.import_module("game_rules.dotsboxes").Game
    network_class = importlib.import_module("model.dotsboxes_6").NetWork
    trainer_class = importlib.import_module("TrainNet").TrainNet
    mcts_class = importlib.import_module("mcts.dotsboxes").Mcts
    return args_module.Args(), game_class, (network_class, trainer_class), mcts_class


class PublishedSearch:
    """Adapt upstream multi-capture actions to one legal host move per request."""

    def __init__(self, game: Any, mcts: Any) -> None:
        self.game = game
        self.mcts = mcts
        self.pending: list[int] = []
        self.expected_board: NDArray[np.int8] | None = None
        self.expected_player: int | None = None
        self.fake_last_move: int | None = None

    def reset(self) -> None:
        """Discard game-local plans and upstream neural evaluation cache."""
        self.pending.clear()
        self.expected_board = None
        self.expected_player = None
        self.fake_last_move = None
        self.mcts.root = None
        self.mcts.node_info.clear()

    def _quick_move(self, board: NDArray[np.int8], to_play: int) -> dict[str, Any]:
        action = self.pending.pop(0)
        if action not in self.game.get_legal_action_dots(board):
            raise ValueError("Upstream forced-capture sequence contains an illegal move")
        same_player, next_board = self.game.get_next_board(board, action)
        if not same_player:
            raise ValueError("Upstream forced-capture sequence unexpectedly changes player")
        self.expected_board = np.asarray(next_board, dtype=np.int8).reshape(-1)
        self.expected_player = to_play
        return {"move": list(divmod(action, LATTICE_SIZE)), "selection": "forced_capture"}

    def select(self, request: dict[str, Any]) -> dict[str, Any]:
        """Search one request, reusing a forced sequence only on the expected board."""
        to_play = request["to_play"]
        if type(to_play) is not int:
            raise ValueError("to_play must be an integer")
        board = canonical_board(request["board"], to_play)
        last_move = lattice_action(request.get("last_move"), board)
        seed = request["seed"]
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed % (2**63))
        if self.expected_board is not None:
            matches = self.expected_player == to_play and np.array_equal(self.expected_board, board)
            if matches and self.pending:
                return self._quick_move(board, to_play)
            if matches and self.fake_last_move is not None:
                last_move = self.fake_last_move
            self.pending.clear()
            self.expected_board = None
            self.expected_player = None
            self.fake_last_move = None
        legal = self.game.get_legal_action_dots(board)
        if not legal:
            raise ValueError("Cannot select a move on a finished board")
        # The upstream engine stops once 13 boxes are owned. Complete the host game
        # deterministically; these moves are not claimed to optimize final margin.
        if self.game.game_over_dots(board) != self.game.GAME_NOT_END:
            return {"move": list(divmod(int(legal[0]), 11)), "selection": "decided_completion"}
        layers = int(np.count_nonzero(board.reshape(11, 11)[_EDGES]))
        self.mcts.search(0, board, layers, action=last_move)
        root = self.mcts.root
        if root.root_quick_move:
            self.pending = [int(action) for action in root.root_quick_move]
            self.fake_last_move = None if root.fake_last_move is None else int(root.fake_last_move)
            return self._quick_move(board, to_play)
        visits = np.asarray(root.child_N)
        if not np.isfinite(visits).all() or visits.sum() <= 0:
            raise ValueError(
                "Upstream search returned no child visits; use at least two simulations"
            )
        action = int(np.argmax(visits))
        if action not in legal:
            raise ValueError("Upstream search selected an illegal move")
        return {
            "move": list(divmod(action, 11)),
            "selection": "search",
            "child_visits": int(visits.sum()),
        }


class _ScalarPredictor:
    """Keep upstream inference, converting its one-element value to a scalar."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self.net_work = trainer.net_work

    def predict(self, board: Any, network: Any) -> tuple[Any, float]:
        policy, value = self.trainer.predict(board, network)
        return policy, float(np.asarray(value).item())


def create_search(
    root: Path, checkpoint: Path, simulations: int, device: str, threads: int
) -> tuple[PublishedSearch, dict[str, Any]]:
    """Load a strict weights-only checkpoint and the published search implementation."""
    if simulations < 2 or threads < 1:
        raise ValueError("simulations must be at least two and threads must be positive")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable")
    torch.set_num_threads(threads)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "state_dict" not in state:
        raise ValueError("Expected an upstream checkpoint containing state_dict")
    weights = state["state_dict"]
    channels = int(weights["layer1.left.0.weight"].shape[1])
    if channels not in (2, 4, 26):
        raise ValueError(f"Unsupported upstream input channel count: {channels}")
    args, game_class, classes, mcts_class = load_upstream(root)
    network_class, trainer_class = classes
    args.CenterName = "Control Center"
    args.input_channels = channels
    args.cuda = device == "cuda"
    args.device = torch.device(device)
    args.gpu_parallel = False
    args.train_num_search = simulations
    args.strong_search_num = simulations
    args.use_chain_loop_prune = True
    args.use_endgame = True
    game = game_class(args)
    game.to_string = upstream_position_key
    network = network_class(args)
    trainer = trainer_class(args, network, None)
    # TrainNet initializes parameters in its constructor, so load only afterward.
    network.load_state_dict(weights, strict=True)
    network.to(args.device)
    network.eval()
    search = PublishedSearch(game, mcts_class(game, args, _ScalarPredictor(trainer), None))
    metadata = {
        "implementation": "published ComputingCenter1 search / ControlCenter settings",
        "checkpoint": str(checkpoint.resolve()),
        "input_channels": channels,
        "simulations": simulations,
        "device": device,
        "threads": threads,
        "chain_loop_pruning": True,
        "endgame": True,
        "compatibility": [
            "ndarray.tostring replaced by equivalent tobytes cache key",
            "single-element network value converted to scalar for NumPy 2",
        ],
        "completion": "first legal edge after either player owns 13 boxes",
    }
    return search, metadata


def main() -> None:
    """Serve newline-delimited JSON on stdin/stdout, keeping upstream logs on stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        search, metadata = create_search(
            args.root, args.checkpoint, args.simulations, args.device, args.threads
        )
    print(json.dumps({"ready": True, "metadata": metadata}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if not isinstance(request, dict):
            raise ValueError("Expected a JSON object")
        if request.get("command") == "close":
            return
        with redirect_stdout(sys.stderr):
            if request.get("command") == "reset":
                search.reset()
                response = {"reset": True}
            else:
                response = search.select(request)
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
