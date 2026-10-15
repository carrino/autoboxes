"""Unvalidated Windows GUI smoke bridge for the original 2007 PRsBoxes.exe.

Uses normal File/Open -> Computer Move -> File/Save, never test (-t) mode.
Run only three positions first. Output is adapter validation data, not yet a
certified engine benchmark. No game-playing algorithm is implemented here.
"""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time
import traceback
from ctypes import wintypes
from pathlib import Path

ENGINE_SHA256 = "2f603782446e0df640b832966ac7b19e5ad9e8781aa6bc0c279fa415901b9d55"
WM_COMMAND, WM_SETTEXT, WM_CLOSE = 0x111, 0xC, 0x10
ID_NEW, ID_OPEN, ID_SAVE = 0xE100, 0xE101, 0xE103
ID_COMPUTER, ID_OPTIONS = 0x8004, 0x8009


def replay(rows, cols, prefix):
    h_count = (rows + 1) * cols
    box_edges = []
    for y in range(rows):
        for x in range(cols):
            box_edges.append(
                {
                    y * cols + x,
                    (y + 1) * cols + x,
                    h_count + y * (cols + 1) + x,
                    h_count + y * (cols + 1) + x + 1,
                }
            )
    edges, owners, mover = set(), [0] * (rows * cols), 1
    for edge in prefix:
        assert isinstance(edge, int) and 0 <= edge < h_count + rows * (cols + 1)
        assert edge not in edges, f"repeated edge {edge}"
        edges.add(edge)
        captured = 0
        for box, boundary in enumerate(box_edges):
            if owners[box] == 0 and boundary <= edges:
                owners[box] = mover
                captured += 1
        if not captured:
            mover = 3 - mover
    return edges, owners, mover


def position_text(rows, cols, prefix):
    edges, owners, mover = replay(rows, cols, prefix)
    grid = [[" "] * (2 * cols + 1) for _ in range(2 * rows + 1)]
    for y in range(rows + 1):
        for x in range(cols + 1):
            grid[2 * y][2 * x] = "+"
    for edge in edges:
        if edge < (rows + 1) * cols:
            y, x = divmod(edge, cols)
            grid[2 * y][2 * x + 1] = "-"
        else:
            y, x = divmod(edge - (rows + 1) * cols, cols + 1)
            grid[2 * y + 1][2 * x] = "I"
    for box, owner in enumerate(owners):
        if owner:
            y, x = divmod(box, cols)
            grid[2 * y + 1][2 * x + 1] = "C" if owner == mover else "H"
    lines = ["".join(line) for line in grid]
    # The native loader commits the width only on a non-edge character after
    # the final dot (0x406ca2), not at the end of a well-formed first row.
    # Match its own writer's marker (0x406a48); otherwise width stays 12 dots.
    lines[0] += "#The first row determines width"
    return "\n".join(
        ["#Root mover is Computer; captured boxes preserve score."]
        + lines
        + ["Anything illegal in column one determines height.", ""]
    )


def parse_position(text, rows, cols):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("+"))
    grid = lines[start : start + 2 * rows + 1]
    assert len(grid) == 2 * rows + 1
    assert all(len(line) >= 2 * cols + 1 for line in grid)
    edges, owners = set(), []
    for y in range(rows + 1):
        for x in range(cols + 1):
            assert grid[2 * y][2 * x] == "+"
        for x in range(cols):
            char = grid[2 * y][2 * x + 1]
            assert char in " -=", char
            if char != " ":
                edges.add(y * cols + x)
    for y in range(rows):
        for x in range(cols + 1):
            char = grid[2 * y + 1][2 * x]
            assert char in " I|", char
            if char != " ":
                edges.add((rows + 1) * cols + y * (cols + 1) + x)
        for x in range(cols):
            char = grid[2 * y + 1][2 * x + 1]
            assert char in " HC", char
            owners.append(char)
    return edges, owners


class Windows:
    def __init__(self, pid):
        assert os.name == "nt", "GUI execution requires native Windows Python"
        self.pid = pid
        self.u = ctypes.WinDLL("user32", use_last_error=True)
        self.callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        self.u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.u.GetDlgItem.argtypes = [wintypes.HWND, ctypes.c_int]
        self.u.GetDlgItem.restype = wintypes.HWND
        self.u.PostMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self.u.SendMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self.u.SendMessageW.restype = ctypes.c_ssize_t
        self.u.SendMessageTimeoutW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.u.SendMessageTimeoutW.restype = wintypes.LPARAM
        self.u.IsWindowVisible.argtypes = [wintypes.HWND]

    def text(self, hwnd):
        buf = ctypes.create_unicode_buffer(8192)
        self.u.GetWindowTextW(hwnd, buf, len(buf))
        return buf.value

    def windows(self):
        result = []

        @self.callback
        def visit(hwnd, unused):
            pid = wintypes.DWORD()
            self.u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == self.pid and self.u.IsWindowVisible(hwnd):
                name = ctypes.create_unicode_buffer(256)
                self.u.GetClassNameW(hwnd, name, len(name))
                result.append((hwnd, name.value, self.text(hwnd)))
            return True

        self.u.EnumWindows(visit, 0)
        return result

    def wait(self, predicate, timeout=15):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            result = predicate(self.windows())
            if result:
                return result
            time.sleep(0.05)
        raise RuntimeError(f"Window wait failed; visible windows: {self.windows()}")

    def command(self, hwnd, command):
        assert self.u.PostMessageW(hwnd, WM_COMMAND, command, 0)

    def send(self, hwnd, message, wparam, lparam):
        result = ctypes.c_size_t()
        ok = self.u.SendMessageTimeoutW(
            hwnd, message, wparam, lparam, 0x2, 3000, ctypes.byref(result)
        )
        assert ok, f"Window message timed out: {message}"
        return result.value

    def field(self, dialog, control_id, value):
        hwnd = self.u.GetDlgItem(dialog, control_id)
        assert hwnd, f"Missing control {control_id} in {self.windows()}"
        data = ctypes.create_unicode_buffer(str(value))
        self.send(hwnd, WM_SETTEXT, 0, ctypes.cast(data, ctypes.c_void_p).value)
        received = ctypes.create_unicode_buffer(8192)
        # GetWindowTextW does not read another process's edit control; WM_GETTEXT
        # is a system message and Windows marshals its buffer across processes.
        self.send(hwnd, 0xD, len(received), ctypes.cast(received, ctypes.c_void_p).value)
        assert received.value == str(value), (control_id, received.value, value)

    def dialog_gone(self, hwnd):
        self.wait(lambda windows: not any(w[0] == hwnd for w in windows))

    def new_game(self, dialog, rows, cols, seed):
        self.field(dialog, 1000, cols + 1)
        self.field(dialog, 1001, rows + 1)
        self.send(self.u.GetDlgItem(dialog, 1005), 0xF5, 0, 0)  # BM_CLICK
        self.field(dialog, 1003, seed)
        self.command(dialog, 1)
        self.dialog_gone(dialog)


def run_position(engine, position, root, seconds, seed, watchdog):
    rows, cols, prefix = position["rows"], position["cols"], position["prefix"]
    before = position_text(rows, cols, prefix)
    source = root / "loaded-position.txt"
    source.write_text(before, encoding="ascii", newline="\r\n")
    (root / "input-position.txt").write_text(before, encoding="ascii")
    expected_edges, expected_owners = parse_position(before, rows, cols)
    process = subprocess.Popen([str(engine)], cwd=root)
    ui = Windows(process.pid)
    try:
        first = ui.wait(
            lambda ws: next((w for w in ws if w[1] == "#32770" or w[1].startswith("Afx")), None)
        )
        if first[1] == "#32770":
            assert first[2] == "Parameters for a new game", first
            ui.new_game(first[0], rows, cols, seed)
        main = ui.wait(lambda ws: next((w[0] for w in ws if w[1].startswith("Afx")), None))
        ui.command(main, ID_NEW)
        dialog = ui.wait(
            lambda ws: next((w[0] for w in ws if w[2] == "Parameters for a new game"), None)
        )
        ui.new_game(dialog, rows, cols, seed)
        ui.command(main, ID_OPEN)
        dialog = ui.wait(lambda ws: next((w[0] for w in ws if w[1] == "#32770"), None))
        ui.field(dialog, 0x480, source)
        ui.command(dialog, 1)
        ui.dialog_gone(dialog)
        ui.wait(lambda ws: any(w[0] == main and "loaded-position" in w[2] for w in ws))
        ui.command(main, ID_OPTIONS)
        dialog = ui.wait(lambda ws: next((w[0] for w in ws if w[2] == "Program Options"), None))
        ui.field(dialog, 1014, seconds)
        auto = ui.u.GetDlgItem(dialog, 1027)
        assert auto, "Auto Play control missing"
        ui.send(auto, 0xF1, 0, 0)  # BM_SETCHECK: unchecked
        ui.command(dialog, 1)
        ui.dialog_gone(dialog)
        old_mtime = source.stat().st_mtime_ns
        ui.command(main, ID_SAVE)
        end = time.monotonic() + 10
        while source.stat().st_mtime_ns == old_mtime and time.monotonic() < end:
            time.sleep(0.05)
        assert source.stat().st_mtime_ns != old_mtime, "Initial save never completed"
        saved = source.read_text(encoding="ascii")
        assert parse_position(saved, rows, cols) == (expected_edges, expected_owners), (
            "Native load/save changed root state"
        )
        (root / "native-before.txt").write_text(saved, encoding="ascii")
        started = time.monotonic()
        ui.command(main, ID_COMPUTER)
        while time.monotonic() - started < watchdog:
            if process.poll() is not None:
                raise RuntimeError(f"Engine exited {process.returncode}; windows={ui.windows()}")
            dialogs = [w for w in ui.windows() if w[1] == "#32770"]
            assert not dialogs, f"Unexpected engine dialog: {dialogs}"
            ui.command(main, ID_SAVE)
            time.sleep(0.25)
            # Native File/Save may briefly hold an exclusive write handle.
            try:
                saved = source.read_text(encoding="ascii")
                edges, owners = parse_position(saved, rows, cols)
            except (PermissionError, AssertionError, StopIteration):
                continue
            if edges != expected_edges:
                added = edges - expected_edges
                assert expected_edges <= edges and len(added) == 1, (
                    f"Expected one move; got added={added}"
                )
                edge = added.pop()
                _, _, root_mover = replay(rows, cols, prefix)
                _, child_owners, _ = replay(rows, cols, prefix + [edge])
                assert owners == [
                    " " if owner == 0 else "C" if owner == root_mover else "H"
                    for owner in child_owners
                ], "Native move changed ownership unexpectedly"
                (root / "native-after.txt").write_text(saved, encoding="ascii")
                return {
                    "position": position["position"],
                    "edge": edge,
                    "elapsed_seconds": time.monotonic() - started,
                    "requested_seconds": seconds,
                    "requested_seed_before_load": seed,
                    "seed_after_load_verified": False,
                    "elapsed_includes_gui_and_save_poll": True,
                    "native_root_roundtrip_verified": True,
                    "adapter_status": "unvalidated-smoke-prototype",
                    "mode": "normal-gui-computer-move",
                }
        raise TimeoutError(f"No single saved move within {watchdog}s")
    finally:
        for hwnd, _, _ in ui.windows():
            ui.u.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--positions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seconds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--watchdog", type=int, default=60)
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()
    assert os.name == "nt", "Run this with native Windows Python, not WSL Python"
    assert 1 <= args.limit <= 3, "Validate the three-root smoke before broadening"
    assert args.seconds > 0 and args.watchdog > args.seconds
    engine = args.exe.resolve()
    assert hashlib.sha256(engine.read_bytes()).hexdigest() == ENGINE_SHA256, (
        "Unrecognised PRsBoxes.exe"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    positions = [
        json.loads(line) for line in args.positions.read_text().splitlines() if line.strip()
    ][: args.limit]
    with (args.output / "results.jsonl").open("w") as output:
        for position in positions:
            root = (args.output / f"root-{position['position']:03}").resolve()
            root.mkdir()
            try:
                result = run_position(
                    engine, position, root, args.seconds, args.seed, args.watchdog
                )
            except Exception:
                (root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                raise
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
