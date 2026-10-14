"""Exercise the arena boundary with a real subprocess, without upstream weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from alpha_go.agents.base import get_agent
from alpha_go.boxes import boxeszero
from alpha_go.boxes.boxeszero import BoxesZeroAgent, BoxesZeroConfig, register_boxeszero
from alpha_go.boxes.rules import BoxesBoard


@pytest.fixture
def worker_script(tmp_path: Path) -> Path:
    path = tmp_path / "fake_worker.py"
    path.write_text("""import json, sys, time
mode, requests = sys.argv[1:]
print(json.dumps({"ready": mode != "bad_start", "metadata": {"fake": True}}), flush=True)
if mode == "bad_start":
    time.sleep(30)
for line in sys.stdin:
    request = json.loads(line)
    with open(requests, "a") as stream:
        stream.write(json.dumps(request) + "\\n")
    if mode == "timeout":
        time.sleep(30)
    if mode == "eof":
        sys.exit(0)
    if mode == "error":
        response = {"error": "test failure"}
    elif mode == "illegal":
        response = {"move": [0, 0]}
    elif mode == "occupied":
        response = {"move": [0, 1]}
    elif mode == "malformed":
        response = {"move": [False, 1]}
    else:
        board = request["board"]
        response = {"move": next([r, c] for r in range(11) for c in range(11)
                                 if (r + c) % 2 and board[r][c] == 0)}
    print(json.dumps(response), flush=True)
""")
    return path


@pytest.fixture
def verified(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, Path]]:
    calls: list[tuple[Path, Path]] = []

    def verify(root: Path, checkpoint: Path) -> dict[str, object]:
        calls.append((root, checkpoint))
        return {"source_commit": "fake", "checkpoint_sha256": "fake"}

    monkeypatch.setattr(boxeszero, "_verify_install", verify)
    return calls


def make_agent(
    worker_script: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str = "legal",
) -> BoxesZeroAgent:
    config = BoxesZeroConfig(
        worker_script.parent, worker_script.parent / "weights.pt", response_timeout=2.0
    )
    monkeypatch.setattr(
        BoxesZeroAgent,
        "_worker_command",
        lambda self: [
            sys.executable,
            "-u",
            str(worker_script),
            mode,
            str(worker_script.parent / "requests.jsonl"),
        ],
    )
    return BoxesZeroAgent(config)


def test_full_state_keeps_opening_captures_and_notifies_last_move(
    worker_script: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified: list[tuple[Path, Path]],
) -> None:
    agent = make_agent(worker_script, monkeypatch)
    board = BoxesBoard(5)
    # Fourth side is captured by player 2, who keeps the turn.
    for move in ((0, 1), (1, 0), (1, 2), (2, 1)):
        assert board.play(*move)
    assert board.to_play() == 2 and board.score() == -1
    agent.start_game(5)
    process = agent._process
    try:
        first = agent.select_move(board, 42)
        assert board.is_legal(*first)
        assert board.play(*first)
        agent.notify_move(*first)
        second = agent.select_move(board, 43)
        assert board.is_legal(*second)
    finally:
        agent.end_game()
    assert process is not None and process.poll() is not None
    requests = [
        json.loads(line)
        for line in (worker_script.parent / "requests.jsonl").read_text().splitlines()
    ]
    assert requests[0]["board"][1][1] == 2
    assert requests[0]["to_play"] == 2 and requests[0]["last_move"] is None
    assert requests[0]["seed"] == 42
    assert requests[1]["last_move"] == list(first)
    assert requests[1]["board"][first[0]][first[1]] == 1
    assert len(verified) == 1


def test_new_game_resets_process_and_verifies_once(
    worker_script: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified: list[tuple[Path, Path]],
) -> None:
    agent = make_agent(worker_script, monkeypatch)
    agent.start_game(5)
    previous = agent._process
    agent.notify_move(0, 1)
    agent.start_game(5)
    try:
        assert previous is not None and previous.poll() is not None
        assert agent._process is not previous and agent._last_move is None
        assert len(verified) == 1
        assert BoxesBoard(5).is_legal(*agent.select_move(BoxesBoard(5), 0))
    finally:
        agent.end_game()
        agent.end_game()


@pytest.mark.parametrize(
    "mode,error,match",
    [
        ("timeout", TimeoutError, "timed out"),
        ("eof", RuntimeError, "closed its output"),
        ("error", RuntimeError, "test failure"),
        ("illegal", RuntimeError, "illegal move"),
        ("occupied", RuntimeError, "illegal move"),
        ("malformed", RuntimeError, "invalid move"),
    ],
)
def test_worker_failures_do_not_become_fallback_moves(
    worker_script: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified: list[tuple[Path, Path]],
    mode: str,
    error: type[Exception],
    match: str,
) -> None:
    agent = make_agent(worker_script, monkeypatch, mode)
    board = BoxesBoard(5)
    board.play(0, 1)
    agent.start_game(5)
    if mode == "timeout":
        from dataclasses import replace

        agent.config = replace(agent.config, response_timeout=0.2)
    process = agent._process
    try:
        with pytest.raises(error, match=match):
            agent.select_move(board, 0)
    finally:
        agent.end_game()
    assert process is not None and process.poll() is not None


def test_failed_startup_reaps_worker(
    worker_script: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified: list[tuple[Path, Path]],
) -> None:
    agent = make_agent(worker_script, monkeypatch, "bad_start")
    with pytest.raises(RuntimeError, match="did not become ready"):
        agent.start_game(5)
    assert agent._process is None


def test_reject_wrong_board_size_before_launch(
    worker_script: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified: list[tuple[Path, Path]],
) -> None:
    agent = make_agent(worker_script, monkeypatch)
    with pytest.raises(ValueError, match="5x5"):
        agent.start_game(3)
    with pytest.raises(ValueError, match="5x5"):
        agent.select_move(BoxesBoard(5, 4), 0)
    assert not verified and agent._process is None


def test_config_relative_paths_and_registration(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"root": "upstream", "checkpoint": "upstream/model.pt"}))
    name = f"boxeszero:{path}"
    register_boxeszero(name)
    agent = get_agent(name)
    assert isinstance(agent, BoxesZeroAgent)
    assert agent.config.root == tmp_path / "upstream"
    assert agent.checkpoint_path == str(tmp_path / "upstream/model.pt")
    assert agent.config.simulations == 800 and agent.config.device == "cpu"
    assert "-B" in agent._worker_command()


@pytest.mark.parametrize(
    "field,value",
    [
        ("simulations", 0),
        ("simulations", 1),
        ("simulations", True),
        ("threads", -1),
        ("response_timeout", float("nan")),
        ("response_timeout", 0),
        ("device", "cuda:9"),
        ("unexpected", 3),
    ],
)
def test_bad_config_fails_early(tmp_path: Path, field: str, value: object) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"root": "r", "checkpoint": "c", field: value}))
    with pytest.raises(ValueError):
        BoxesZeroConfig.load(path)


def test_arena_plays_both_colours_from_an_opening_with_a_capture(
    worker_script: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified: list[tuple[Path, Path]],
) -> None:
    from alpha_go.boxes.arena import play_match

    make_agent(worker_script, monkeypatch)  # installs fake worker command
    config = worker_script.parent / "config.json"
    config.write_text(json.dumps({"root": ".", "checkpoint": "weights.pt"}))
    name = f"boxeszero:{config}"
    register_boxeszero(name)
    opening = [(0, 1), (1, 0), (1, 2), (2, 1)]
    result = play_match(
        name, "boxes-greedy", rows=5, num_games=2, num_workers=1, openings=[opening]
    )
    assert result.games == 2 and len(result.margins) == 2
    assert all(len(moves) == 60 for moves in result.moves)
    assert all(moves[:4] == opening for moves in result.moves)
    assert len(verified) == 1  # one cached agent, independent subprocess per game
    requests = [
        json.loads(line)
        for line in (worker_script.parent / "requests.jsonl").read_text().splitlines()
    ]
    assert all(request["board"][1][1] == 2 for request in requests)
    assert {request["to_play"] for request in requests} == {1, 2}
