"""Collect resumable, searched Qdab histories as branch sources, not labels."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import signal
import time
from pathlib import Path
from typing import Any

from alpha_go.boxes.qdab import (
    ARCHIVE_SHA256,
    ARTIFACTS,
    COMMIT,
    QdabAgent,
    install_release,
    replay_prefix,
)
from alpha_go.boxes.validation import replay_group
from alpha_go.gameplay import GameRecord, play_game
from alpha_go.games import get_game


def atomic_json(path: Path, payload: Any) -> None:
    """Replace a small JSON manifest only after the write completes."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def family_fields(prefix: list[int]) -> dict[str, str]:
    """Preserve source identity and the existing canonical opening-family split."""
    board, _ = replay_prefix(prefix)
    digest = hashlib.sha256(bytes(prefix)).hexdigest()
    group = replay_group({"moves": [board.row_col(edge) for edge in prefix]}, board.geo)
    return {"source_game_id": f"qdab-{digest}", "source_family": f"qdab-{digest}",
            "source_replay_group": group.hex()}


def validate_history(row: dict[str, Any]) -> None:
    """Validate resume input and prohibit fabricated truncated-game targets."""
    board, _ = replay_prefix(row["prefix"])
    if row["scores"] != board.boxes or row["to_play"] != board.to_play():
        raise ValueError("Saved Qdab history has wrong scores or mover")
    if row["undrawn"] != 60 - len(row["prefix"]) or row["outcome_target"] is not None:
        raise ValueError("Qdab branch sources must not supply truncated outcomes")
    for key, value in family_fields(row["prefix"]).items():
        if row.get(key) != value:
            raise ValueError("Qdab history family does not match its complete prefix")


def collect(args: argparse.Namespace) -> None:
    """Append complete searched prefixes and resume without replacing saved games."""
    if args.num_games < 1 or not 1 <= args.max_moves <= 60 or args.attempts < 1:
        raise ValueError("Require num-games/attempts >= 1 and max-moves in 1..60")
    if not (math.isfinite(args.seconds) and args.seconds > 0 and
            math.isfinite(args.hard_timeout) and args.hard_timeout >= args.seconds):
        raise ValueError("Require 0 < seconds <= hard-timeout")
    release = install_release(args.release, args.archive)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    recipe = dict(schema=1, source="Qdab-0.1-linux-amd64-quctann", source_commit=COMMIT,
                  archive_sha256=ARCHIVE_SHA256, artifacts=ARTIFACTS,
                  requested_seconds=args.seconds, hard_timeout_seconds=args.hard_timeout,
                  max_moves=args.max_moves, seed=args.seed, max_attempts=args.attempts,
                  seed_control="Qdab uses wall clock; gameplay seed does not control search",
                  scope="searched branch sources only; truncated outcomes are never targets")
    manifest = output.with_suffix(".meta.json")
    game_dir = output.with_suffix(".games")
    game_dir.mkdir(exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            if json.loads(manifest.read_text()) != recipe:
                raise ValueError("Cannot resume Qdab collection with a changed recipe")
        else:
            if output.exists() or any(game_dir.glob("*.json")):
                raise ValueError("Existing Qdab output has no recipe manifest")
            atomic_json(manifest, recipe)
        rows = [json.loads(path.read_text()) for path in sorted(game_dir.glob("*.json"))]
        for index, row in enumerate(rows):
            validate_history(row)
            if row["game_index"] != index:
                raise ValueError("Qdab completed game sequence has a gap")
        if len(rows) > args.num_games:
            raise ValueError("Requested fewer games than already collected")
        print(f"Qdab: {len(rows)}/{args.num_games} searched histories ready; "
              f"{args.max_moves} edges/game, {args.seconds:g}s/decision, "
              f"external {args.hard_timeout:g}s cap; CPU only", flush=True)
        for index in range(len(rows), args.num_games):
            started = time.monotonic()
            agents, record, attempts = searched_game(release, args, index, game_dir)
            prefix = agents[0].prefix
            if prefix != agents[1].prefix or len(prefix) != args.max_moves:
                raise ValueError("Incomplete or inconsistent Qdab source game")
            board = agents[0].reference
            row = dict(schema=1, game_index=index, prefix=prefix, rows=5, cols=5,
                       scores=list(board.boxes), to_play=board.to_play(),
                       undrawn=60 - len(prefix), **family_fields(prefix),
                       source="qdab", source_commit=COMMIT,
                       engine_sha256=ARTIFACTS["server"],
                       requested_seconds=args.seconds, hard_timeout_seconds=args.hard_timeout,
                       seed=args.seed + index, seed_control="uncontrolled wall clock",
                       outcome_target=None, termination=record.termination,
                       elapsed_seconds=time.monotonic() - started, attempts=attempts,
                       decisions=sorted(agents[0].decisions + agents[1].decisions,
                                        key=lambda item: item["move_count"]))
            validate_history(row)
            atomic_json(game_dir / f"game{index:05d}.json", row)
            rows.append(row)
            # The game JSON is authoritative after interruption; rebuild the compact view.
            write_pool(output, rows)
            print(f"Qdab {index + 1}/{args.num_games}: {len(prefix)} searched edges; "
                  f"scores {board.boxes}; {row['elapsed_seconds']:.1f}s; saved {output}",
                  flush=True)
        write_pool(output, rows)


def searched_game(release: Path, args: argparse.Namespace, index: int,
                  game_dir: Path) -> tuple[list[QdabAgent], GameRecord, int]:
    """Retry only externally timed-out games, preserving their incomplete histories."""
    for attempt in range(1, args.attempts + 1):
        started = time.monotonic()
        token = time.time_ns()
        agents = [QdabAgent(release, args.seconds, args.hard_timeout,
                            game_dir / f"game{index:05d}.attempt{attempt}-{token}.engine.log",
                            args.verbose) for _ in range(2)]
        try:
            record = play_game(agents[0], agents[1], board_size=5, board_cols=5,
                               max_moves=args.max_moves, seed=args.seed + index,
                               komi=0, game=get_game("boxes"), collect_boards=False,
                               collect_metrics=False, render_debug_on_error=False,
                               black_agent_name="qdab", white_agent_name="qdab",
                               stop_when_solved=False)
        except TimeoutError as error:
            failure = game_dir / "failures" / f"game{index:05d}.attempt{attempt}-{token}.json"
            failure.parent.mkdir(exist_ok=True)
            atomic_json(failure, dict(
                game_index=index, attempt=attempt, error_type=type(error).__name__,
                error=str(error), prefix=list(agents[0].prefix),
                other_prefix=list(agents[1].prefix),
                scores=list(agents[0].reference.boxes),
                to_play=agents[0].reference.to_play(), outcome_target=None,
                requested_seconds=args.seconds, hard_timeout_seconds=args.hard_timeout,
                source_commit=COMMIT, elapsed_seconds=time.monotonic() - started))
            print(f"Qdab game {index}: attempt {attempt}/{args.attempts} timed out; "
                  f"saved incomplete history {failure}", flush=True)
            if attempt == args.attempts:
                raise
            continue
        return agents, record, attempt
    raise AssertionError("Qdab retry count must be positive")


def write_pool(output: Path, rows: list[dict[str, Any]]) -> None:
    """Write compact portable histories; detailed RPC evidence stays beside them."""
    compact = [{key: value for key, value in row.items() if key != "decisions"} for row in rows]
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in compact))
    temporary.replace(output)


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"Qdab collector interrupted by signal {signum}")


def main() -> None:
    signal.signal(signal.SIGTERM, _interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path("work/qdab-linux"))
    parser.add_argument("--archive", type=Path, help="Optional local pinned release archive")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-games", type=int, default=16)
    parser.add_argument("--max-moves", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--hard-timeout", type=float, default=20)
    parser.add_argument("--attempts", type=int, default=3,
                        help="Bounded attempts per game after RPC timeouts only")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--verbose", action="store_true")
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
