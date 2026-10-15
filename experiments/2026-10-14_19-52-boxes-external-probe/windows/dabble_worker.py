#!/usr/bin/env python3
"""Bridge the pinned KSquares Dabble 1.3 executable to a single JSON request.

Run using native Windows Python and the VS2013 x86 runtime. The upstream executable
has a brief GUI and accepts an integer time budget. Format and state checks are
covered by tests; successful Windows engine execution still requires a smoke run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

EXE_SHA256 = "7e88069a551edae2fc8a404eca9fcb0366b3ba729738e0f4f5bea399a7704c32"
COORDINATES = r"\((\d+), (\d+)\) - \((\d+), (\d+)\)"
STDOUT_MOVE = re.compile(r"^DABBLE MOVE: \d+: " + COORDINATES + r"\s*$")
LOG_MOVE = re.compile(r"^\d+: " + COORDINATES + r"\s*$")


def coins(edge: int, rows: int, cols: int) -> tuple[int, int, int, int]:
    """Map an autoboxes edge to Dabble's strings-and-coins coordinates."""
    horizontal = (rows + 1) * cols
    total = horizontal + rows * (cols + 1)
    if type(edge) is not int or not 0 <= edge < total:
        raise ValueError("Invalid edge")
    if edge < horizontal:
        x, y = edge % cols + 1, edge // cols
        return x, y, x, y + 1
    x, y = (edge-horizontal) % (cols+1), (edge-horizontal) // (cols+1) + 1
    return x, y, x+1, y


def edge_from_coins(c: tuple[int, ...], rows: int, cols: int) -> int:
    """Invert either orientation of a valid strings-and-coins edge."""
    x1, y1, x2, y2 = c
    horizontal = (rows+1)*cols
    if x1 == x2 and abs(y1-y2) == 1 and 1 <= x1 <= cols and 0 <= min(y1, y2) <= rows:
        return min(y1, y2)*cols+x1-1
    if y1 == y2 and abs(x1-x2) == 1 and 1 <= y1 <= rows and 0 <= min(x1, x2) <= cols:
        return horizontal+(y1-1)*(cols+1)+min(x1, x2)
    raise ValueError(f"Invalid strings-and-coins edge: {c}")


def replay(request: dict[str, Any]) -> tuple[int, list[int], list[list[int]]]:
    """Independently reconstruct all edges, owners, scores and the retained turn."""
    rows, cols, prefix = request["rows"], request["cols"], request["prefix"]
    if (type(rows) is not int or type(cols) is not int
            or not (1 <= rows <= 5 and 1 <= cols <= 5)):
        raise ValueError("Hash-enabled Dabble supports at most 5x5 boxes")
    drawn: set[int] = set()
    owners: dict[int, int] = {}
    scores, mover = [0, 0], 0
    horizontal = (rows+1)*cols
    grid = [[0]*(2*cols+1) for _ in range(2*rows+1)]
    for edge in prefix:
        coins(edge, rows, cols)
        if edge in drawn:
            raise ValueError("Repeated prefix edge")
        drawn.add(edge)
        if edge < horizontal:
            grid[2*(edge//cols)][2*(edge % cols)+1] = 1
        else:
            vertical = edge-horizontal
            grid[2*(vertical//(cols+1))+1][2*(vertical % (cols+1))] = 1
        gained = 0
        for r in range(rows):
            for c in range(cols):
                box = r*cols+c
                sides = (r*cols+c, (r+1)*cols+c, horizontal+r*(cols+1)+c,
                         horizontal+r*(cols+1)+c+1)
                if box not in owners and all(e in drawn for e in sides):
                    owners[box] = mover
                    scores[mover] += 1
                    grid[2*r+1][2*c+1] = mover+1
                    gained += 1
        if not gained:
            mover ^= 1
    if sum(scores) == rows*cols:
        raise ValueError("Completed game has no move")
    if "to_play" in request and request["to_play"] != mover+1:
        raise ValueError("Replay mover differs from supplied state")
    if "scores" in request and request["scores"] != scores:
        raise ValueError("Replay scores differ from supplied state")
    if "board" in request and request["board"] != grid:
        raise ValueError("Replay board differs from supplied state")
    return mover+1, scores, grid


def savegame(request: dict[str, Any]) -> str:
    """Serialize a complete history in the upstream DBL savegame format."""
    replay(request)
    rows, cols = request["rows"], request["cols"]
    lines = [f"{cols+1},{rows+1}"]
    for edge in request["prefix"]:
        x1, y1, x2, y2 = coins(edge, rows, cols)
        lines.append(f"({x1}, {y1}) - ({x2}, {y2})")
    return "\n".join(lines)+"\n"


def channel_choice(text: str, pattern: re.Pattern[str], request: dict[str, Any]) -> int | None:
    """Require a complete echoed history in every channel containing moves."""
    moves = []
    for line in text.splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            coordinates = tuple(int(x) for x in match.groups())
            moves.append(edge_from_coins(coordinates, request["rows"], request["cols"]))
    if not moves:
        return None
    prefix = request["prefix"]
    if moves[:len(prefix)] != prefix or len(moves) <= len(prefix):
        raise ValueError("Dabble did not echo the complete prefix and a new move")
    selected = moves[len(prefix)]
    if selected in prefix:
        raise ValueError("Dabble chose a drawn edge")
    return selected


def chosen_edge(stdout: str, dabble_log: str, request: dict[str, Any]) -> int:
    """Use verified stdout/log formats, rejecting incomplete or conflicting output."""
    choices = [channel_choice(stdout, STDOUT_MOVE, request),
               channel_choice(dabble_log, LOG_MOVE, request)]
    known = {choice for choice in choices if choice is not None}
    if len(known) != 1:
        raise ValueError("Dabble output is missing a move or stdout/log choices disagree")
    return known.pop()


def verify_executable(path: Path, expected: str = EXE_SHA256) -> str:
    """Reject an unpinned executable before invoking it."""
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Dabble executable SHA256 mismatch: {actual}")
    return actual


def main() -> None:
    """Invoke one genuine Windows engine decision and preserve all output channels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(sys.stdin.readline())
    seconds = float(request["seconds"])
    if not math.isfinite(seconds) or seconds <= 0 or seconds != int(seconds):
        raise ValueError("This upstream executable accepts positive integer seconds")
    exe = args.exe.resolve(strict=True)
    executable_hash = verify_executable(exe)
    with tempfile.TemporaryDirectory(prefix="dabble-probe-") as temporary:
        folder = Path(temporary)
        game = folder/"input.dabble.dbl"
        game.write_text(savegame(request), encoding="ascii")
        command = [str(exe), str(game), str(int(seconds))]
        start = time.monotonic()
        stdout = stderr = ""
        with subprocess.Popen(command, cwd=folder, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as process:
            try:
                stdout, stderr = process.communicate(timeout=seconds+20)
            finally:
                # Cleanup preserves partial output even when the bounded call times out.
                if process.poll() is None:
                    process.kill()
                    stdout, stderr = process.communicate()
                log_path = folder/"dabble.log"
                dabble_log = log_path.read_text(errors="replace") if log_path.exists() else ""
                sys.stderr.write(json.dumps({"engine_stdout": stdout, "engine_stderr": stderr,
                                             "dabble_log": dabble_log})+"\n")
                sys.stderr.flush()
        elapsed = time.monotonic()-start
        if process.returncode:
            raise RuntimeError(f"Dabble exited with status {process.returncode}")
        mover, scores, _ = replay(request)
        response = dict(edge=chosen_edge(stdout, dabble_log, request), elapsed_seconds=elapsed,
                        engine="Dabble-1.3-KSquares-hash-enabled",
                        executable_sha256=executable_hash, requested_seconds=seconds,
                        input_to_play=mover, input_scores=scores,
                        seed_control="not exposed by upstream command-line interface")
        print(json.dumps(response))


if __name__ == "__main__":
    main()
