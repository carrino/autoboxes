"""Pinned Linux Qdab adapter; searched histories are not training targets.

Qdab's own timer omits its exact presearch. Every RPC therefore has an external
socket deadline, and each decision owns a fresh process under a global port lock.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from alpha_go.agents.base import Agent
from alpha_go.boxes.rules import BoxesBoard

COMMIT = "57c6a5fd849022bf3754c4aa530a2ab2ee7f300c"
ARCHIVE_URL = ("https://raw.githubusercontent.com/yimmon/dots-and-boxes/"
               f"{COMMIT}/release/Qdab_0.1_lnx_amd64.tar.bz2")
ARCHIVE_SHA256 = "cddc6a6c533acd5c47c99ea3c708b342b78ef31a2e260b3f85e65d1ad041dafa"
ARTIFACTS = {
    "server": "9ee41ab1bffc9ad3a176215cbc3239b08950f2964fb0edddb5297072357cee0a",
    "libfann.so.2": "a0190ec77a5bde66991ce78fe6d297e862989853eaea9be1fb8825d75cfcb575",
    "AnnModels/24.ann": "720c356ab6a2ec8024ab9e5f6ec8879e4bf1b963f6d8e035364463ebfcd96ac1",
    "AnnModels/28.ann": "a6c2b283739040611fdb48e2a7e3f18a96da159b91ca0befac460b1d06b223d7",
    "AnnModels/32.ann": "fd816937f512c8418854b6dfa617a489035b6f8307ef5fada9376792337d9b47",
    "AnnModels/stats": "c5742accc673429a87e351019b3d26812f764b0394df457099d871ff6a7bea2b",
}
PORT = 12345


def sha256(path: Path) -> str:
    """Return the digest of a pinned file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_release(release: Path) -> None:
    """Reject changed or incomplete engine artifacts before executing them."""
    for name, expected in ARTIFACTS.items():
        if sha256(release / name) != expected:
            raise ValueError(f"Qdab checksum mismatch: {release / name}")


def install_release(destination: Path, archive: Path | None = None) -> Path:
    """Install only the checked executable, library and models from the release."""
    destination = destination.resolve()
    if (destination / "server").exists():
        verify_release(destination)
        return destination
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise ValueError("The pinned Qdab release requires Linux x86_64 (WSL is supported)")
    with tempfile.TemporaryDirectory(prefix="qdab-install-") as scratch:
        packed = Path(scratch) / "release.tar.bz2"
        if archive is not None:
            packed.write_bytes(archive.read_bytes())
        else:
            with urlopen(ARCHIVE_URL, timeout=60) as response:
                packed.write_bytes(response.read(32 * 1024 * 1024 + 1))
        if sha256(packed) != ARCHIVE_SHA256:
            raise ValueError("Qdab archive checksum mismatch")
        stage = Path(scratch) / "release"
        with tarfile.open(packed) as tar:
            for name in ARTIFACTS:
                member = tar.getmember(name)
                if not member.isfile() or member.size > 16 * 1024 * 1024:
                    raise ValueError(f"Unexpected Qdab archive member: {name}")
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"Missing Qdab artifact: {name}")
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with source:
                    path.write_bytes(source.read())
        verify_release(stage)
        destination.mkdir(parents=True, exist_ok=True)
        # Write the executable last, so interrupted installation is detectable.
        for name in (*[key for key in ARTIFACTS if key != "server"], "server"):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((stage / name).read_bytes())
        (destination / "server").chmod(0o755)
    return destination


def encode_edge(edge: int) -> tuple[str, int]:
    """Map our 5x5 edge to Qdab's transposed H/V bit numbering."""
    if type(edge) is not int or not 0 <= edge < 60:
        raise ValueError("5x5 edge out of range")
    if edge < 30:
        row, col = divmod(edge, 5)
        return "V", col * 6 + row
    return "H", edge - 30


def replay_prefix(prefix: list[int]) -> tuple[BoxesBoard, dict[str, Any]]:
    """Replay scores, retained turns and noncapturing turn count exactly."""
    board = BoxesBoard(5)
    turn = 0
    masks = {"H": 0, "V": 0}
    for edge in prefix:
        axis, bit = encode_edge(edge)
        before = board.to_play()
        if not board.play_edge(edge):
            raise ValueError("Illegal Qdab source prefix")
        turn += board.to_play() != before
        masks[axis] |= 1 << bit
    return board, dict(**masks, S=list(board.boxes), Now=board.player(), Turn=turn)


def decode_result(result: dict[str, Any], board: BoxesBoard) -> list[int]:
    """Decode a forced-capture bundle, rejecting moves across an opponent turn."""
    pending: list[int] = []
    for axis in ("H", "V"):
        mask = result[axis]
        if type(mask) is not int or mask < 0 or mask >> 30:
            raise ValueError("Invalid Qdab move mask")
        for bit in range(30):
            if mask >> bit & 1:
                edge = 30 + bit if axis == "H" else (bit % 6) * 5 + bit // 6
                if not board.is_legal_edge(edge):
                    raise ValueError("Qdab returned a drawn edge")
                pending.append(edge)
    if not pending:
        raise ValueError("Empty Qdab move")
    copy = board.copy()
    ordered: list[int] = []
    while pending:
        captures = [edge for edge in pending
                    if any(copy.sides[b] == 3 for b in copy.geo.edge_boxes[edge])]
        if len(pending) > 1 and not captures:
            raise ValueError("Ambiguous Qdab bundle without a capture")
        edge = captures[0] if captures else pending[0]
        before = copy.to_play()
        assert copy.play_edge(edge)
        pending.remove(edge)
        ordered.append(edge)
        if pending and copy.to_play() != before:
            raise ValueError("Qdab bundle crosses an opponent turn")
    return ordered


@contextmanager
def port_lock() -> Iterator[None]:
    """Fail promptly if another collector owns Qdab's fixed port."""
    with open(Path(tempfile.gettempdir()) / "autoboxes-qdab-12345.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _stop(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def owns_port(pid: int, proc_root: Path = Path("/proc")) -> bool:
    """Confirm our child owns the listener before sending any board to it."""
    sockets = {os.readlink(fd) for fd in (proc_root / str(pid) / "fd").iterdir()}
    for line in (proc_root / "net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if (fields[1] == f"0100007F:{PORT:04X}" and fields[3] == "0A" and
                f"socket:[{fields[9]}]" in sockets):
            return True
    return False


def query(release: Path, prefix: list[int], seconds: float,
          hard_timeout: float, log_path: Path) -> dict[str, Any]:
    """Run one externally bounded decision; leave no engine behind on failure."""
    if not (math.isfinite(seconds) and seconds > 0 and
            math.isfinite(hard_timeout) and hard_timeout >= seconds):
        raise ValueError("Require 0 < seconds <= hard_timeout, both finite")
    board, state = replay_prefix(prefix)
    if board.is_game_over():
        raise ValueError("Cannot query a terminal board")
    rpc = dict(id=1, method="Server.MakeMove", params=[dict(
        Algorithm="quctann", Board=state, Timeout=max(1, round(seconds * 1000)))])
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(release) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with port_lock(), log_path.open("ab", buffering=0) as log:
        # Never attach to or terminate an existing server.
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", PORT))
        process = subprocess.Popen([str(release / "server")], cwd=release, env=env,
                                   stdout=log, stderr=log)
        try:
            sock: socket.socket | None = None
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(f"Qdab exited during startup; see {log_path}")
                try:
                    sock = socket.create_connection(("127.0.0.1", PORT), timeout=.1)
                    break
                except (ConnectionRefusedError, TimeoutError):
                    time.sleep(.02)
            if sock is None:
                raise TimeoutError("Qdab startup deadline exceeded")
            with sock:
                # A competing uncooperative server causes our child to fail its bind.
                time.sleep(.02)
                if process.poll() is not None or not owns_port(process.pid):
                    raise RuntimeError("Qdab does not own its loopback listening port")
                sock.settimeout(hard_timeout)
                sent = time.monotonic()
                sock.sendall((json.dumps(rpc) + "\n").encode())
                with sock.makefile("rb") as stream:
                    line = stream.readline(65536)
                decision_seconds = time.monotonic() - sent
            response = json.loads(line)
            if response.get("id") != 1 or response.get("error") is not None:
                raise RuntimeError(f"Qdab RPC error: {response}")
            ordered = decode_result(response["result"], board)
            return dict(edges=ordered, decision_seconds=decision_seconds,
                        wall_seconds=time.monotonic() - started,
                        requested_seconds=seconds, hard_timeout_seconds=hard_timeout,
                        rpc_request=rpc, rpc_response=response)
        finally:
            _stop(process)


class QdabAgent(Agent):
    """Qdab through shared gameplay, preserving returned capture bundles."""

    def __init__(self, release: Path, seconds: float, hard_timeout: float,
                 log_path: Path, verbose: bool = False) -> None:
        self.release, self.seconds = release, seconds
        self.hard_timeout, self.log_path = hard_timeout, log_path
        self.verbose = verbose
        self.decisions: list[dict[str, Any]] = []

    def start_game(self, board_size: int) -> None:
        if board_size != 5:
            raise ValueError("Qdab only supports 5x5 boxes")
        self.reference = BoxesBoard(5)
        self.prefix: list[int] = []
        self.pending: list[int] = []
        self.decisions = []

    def notify_move(self, row: int, col: int) -> None:
        edge = self.reference.edge_index(row, col)
        if not self.reference.play_edge(edge):
            raise ValueError("Illegal move notified to Qdab")
        self.prefix.append(edge)
        if self.pending and self.pending.pop(0) != edge:
            raise ValueError("Gameplay diverged from Qdab's retained-turn bundle")

    def select_move(self, board: Any, seed: int) -> tuple[int, int]:
        ref = self.reference
        if (board.to_play() != ref.to_play() or
                board.to_numpy().tolist() != ref.to_numpy().tolist()):
            raise ValueError("Qdab and shared gameplay disagree on the board")
        if not self.pending:
            result = query(self.release, self.prefix, self.seconds,
                           self.hard_timeout, self.log_path)
            self.pending = list(result["edges"])
            self.decisions.append(dict(move_count=len(self.prefix), **result))
            if self.verbose or len(self.prefix) % 5 == 0:
                print(f"  move {len(self.prefix)}: {self.pending}; "
                      f"{result['decision_seconds']:.2f}s", flush=True)
        return ref.row_col(self.pending[0])
