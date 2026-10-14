"""Isolated, full-state arena adapter for the author-repository BoxesZero engine.

Worker/model startup happens in ``start_game`` and is excluded from arena move
latency. Encoding, IPC and the complete upstream move selection are included.
"""

from __future__ import annotations

import json
import math
import os
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_go.agents.base import Agent, register_agent


@dataclass(frozen=True)
class BoxesZeroConfig:
    root: Path
    checkpoint: Path
    simulations: int = 800
    device: str = "cpu"
    threads: int = 1
    response_timeout: float = 120.0

    @classmethod
    def load(cls, path: Path) -> BoxesZeroConfig:
        with path.open() as stream:
            raw = json.load(stream)
        if not isinstance(raw, dict):
            raise ValueError("BoxesZero config must be a JSON object")
        allowed = {"root", "checkpoint", "simulations", "device", "threads", "response_timeout"}
        if set(raw) - allowed:
            raise ValueError(f"Unknown BoxesZero config fields: {sorted(set(raw) - allowed)}")
        paths = []
        for field in ("root", "checkpoint"):
            value = raw.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"BoxesZero config requires a {field} path")
            location = Path(value).expanduser()
            paths.append((path.parent / location).resolve())
        positive = []
        for field, default in (("simulations", 800), ("threads", 1)):
            value = raw.get(field, default)
            if type(value) is not int or value < 1:
                raise ValueError(f"BoxesZero {field} must be a positive integer")
            positive.append(value)
        if positive[0] < 2:
            raise ValueError("BoxesZero simulations must be at least two")
        device = raw.get("device", "cpu")
        if device not in ("cpu", "cuda"):
            raise ValueError("BoxesZero device must be cpu or cuda")
        timeout = raw.get("response_timeout", 120.0)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("BoxesZero response_timeout must be finite and positive")
        return cls(paths[0], paths[1], positive[0], device, positive[1], float(timeout))


def _verify_install(root: Path, checkpoint: Path) -> dict[str, object]:
    from alpha_go.boxes.boxeszero_install import verify_install

    return verify_install(root, checkpoint)


class BoxesZeroAgent(Agent):
    """One isolated upstream process per game; no upstream imports in the arena."""

    def __init__(self, config: BoxesZeroConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._buffer = b""
        self._last_move: tuple[int, int] | None = None
        self._verified: dict[str, object] | None = None
        self._metadata_printed = False

    @property
    def checkpoint_path(self) -> str:
        return str(self.config.checkpoint)

    def _worker_command(self) -> list[str]:
        return [
            sys.executable,
            "-B",
            "-u",
            "-m",
            "alpha_go.boxes.boxeszero_worker",
            "--root",
            str(self.config.root),
            "--checkpoint",
            str(self.config.checkpoint),
            "--simulations",
            str(self.config.simulations),
            "--device",
            self.config.device,
            "--threads",
            str(self.config.threads),
        ]

    def start_game(self, board_size: int) -> None:
        self.end_game()
        if board_size != 5:
            raise ValueError("The BoxesZero checkpoint supports 5x5 boxes only")
        if self._verified is None:
            self._verified = _verify_install(self.config.root, self.config.checkpoint)
            print(
                "BoxesZero configuration: "
                + json.dumps(
                    {
                        "root": str(self.config.root),
                        "checkpoint": str(self.config.checkpoint),
                        "simulations": self.config.simulations,
                        "device": self.config.device,
                        "threads": self.config.threads,
                        "provenance": self._verified,
                        "move_timing": (
                            "includes IPC and search; excludes per-game worker/model startup"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        self._buffer = b""
        self._last_move = None
        self._process = subprocess.Popen(
            self._worker_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        ready = False
        try:
            response = self._read_response()
            if response.get("ready") is not True:
                raise RuntimeError(f"BoxesZero worker did not become ready: {response}")
            metadata = response.get("metadata")
            if not isinstance(metadata, dict):
                raise RuntimeError("BoxesZero worker omitted its runtime metadata")
            if not self._metadata_printed:
                print("BoxesZero worker: " + json.dumps(metadata, sort_keys=True), flush=True)
                self._metadata_printed = True
            ready = True
        finally:
            if not ready:
                self.end_game()

    def _read_response(self) -> dict[str, Any]:
        process = self._process
        assert process is not None and process.stdout is not None
        deadline = time.monotonic() + self.config.response_timeout
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("BoxesZero worker response timed out")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("BoxesZero worker closed its output before replying")
                self._buffer += chunk
                if len(self._buffer) > 1024 * 1024:
                    raise RuntimeError("BoxesZero worker response exceeds 1 MiB")
        line, self._buffer = self._buffer.split(b"\n", 1)
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("BoxesZero worker response must be a JSON object")
        if "error" in response:
            raise RuntimeError(f"BoxesZero worker failed: {response['error']}")
        return response

    def select_move(self, board: Any, seed: int) -> tuple[int, int]:
        if board.rows() != 5 or board.cols() != 5:
            raise ValueError("The BoxesZero checkpoint supports 5x5 boxes only")
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError("BoxesZero worker is not running; call start_game first")
        assert process.stdin is not None
        request = {
            "board": board.to_numpy().tolist(),
            "to_play": int(board.to_play()),
            "seed": int(seed),
            "last_move": self._last_move,
        }
        process.stdin.write((json.dumps(request) + "\n").encode())
        process.stdin.flush()
        response = self._read_response()
        move = response.get("move")
        if (
            not isinstance(move, list)
            or len(move) != 2
            or any(type(coordinate) is not int for coordinate in move)
        ):
            raise RuntimeError(f"BoxesZero returned an invalid move: {move}")
        row, col = move
        if not board.is_legal(row, col):
            raise RuntimeError(f"BoxesZero returned an illegal move: {(row, col)}")
        return int(row), int(col)

    def notify_move(self, row: int, col: int) -> None:
        self._last_move = (row, col)

    def end_game(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        # Closing stdin requests EOF shutdown without risking a write to a broken pipe.
        if process.stdin is not None:
            process.stdin.close()
        deadline = time.monotonic() + 0.25
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if process.poll() is None:
            process.terminate()
        deadline = time.monotonic() + 0.25
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdout is not None:
            process.stdout.close()
        self._buffer = b""
        self._last_move = None


def register_boxeszero(name: str) -> None:
    """Register ``boxeszero:<config.json>`` without loading upstream code in this process."""
    prefix = "boxeszero:"
    if not name.startswith(prefix) or not name[len(prefix) :]:
        raise ValueError("BoxesZero agent name must be boxeszero:<config.json>")
    config = BoxesZeroConfig.load(Path(name[len(prefix) :]).expanduser().resolve())

    @register_agent(name)
    class ConfiguredBoxesZeroAgent(BoxesZeroAgent):
        def __init__(self) -> None:
            super().__init__(config)
