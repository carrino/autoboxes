"""Resumable full games against pinned Qdab; no training or solver adjudication.

The primary rating uses fresh side-swapped four-edge openings. Empty-board games
are a separate descriptive panel, never extra independent samples for that rating.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import FrameType
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import torch

from alpha_go.agents.base import Agent
from alpha_go.boxes.arena import random_opening
from alpha_go.boxes.curriculum import checked_files, package_results, write_json
from alpha_go.boxes.external_rating import summarize_margins
from alpha_go.boxes.inference import PlaneBatchedEngine
from alpha_go.boxes.nn_agent import (
    BoxesEngineEvaluator,
    BoxesMCTSAgent,
    load_boxes_net,
    pick_device,
)
from alpha_go.boxes.qdab import (
    ARCHIVE_SHA256,
    ARTIFACTS,
    COMMIT,
    QdabAgent,
    install_release,
    sha256,
)
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.gameplay import play_game
from alpha_go.games import get_game

SEARCH = dict(num_simulations=400, solver_max_undrawn=34, solver_node_budget=50000,
              c_puct=1.5, leaf_batch_size=16, temperature=1.0, temperature_cutoff=0,
              prove_terminals=True, shape_terminals=True, solver_table_entries=1 << 20,
              merge_equivalent=False)
EVALUATOR = dict(policy_temperature=1.0, margin_utility_lambda=0.5, margin_utility_k=6.0)


class OpeningQdabAgent(QdabAgent):
    """Shared gameplay preplays openings before start_game, without notifications."""

    def __init__(self, *args: Any, opening: list[list[int]], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.opening = opening

    def start_game(self, board_size: int) -> None:
        super().start_game(board_size)
        for row, col in self.opening:
            self.notify_move(row, col)


def schedule(games: int, empty_games: int, seed: int) -> list[dict[str, Any]]:
    """Freeze the full sample before either player is queried."""
    if games <= 0 or games % 2 or empty_games < 0 or empty_games % 2:
        raise ValueError("Require positive even games and nonnegative even empty-games")
    rng = np.random.default_rng(seed)
    result: list[dict[str, Any]] = []
    for panel, count in (("paired", games), ("empty", empty_games)):
        for pair in range(count // 2):
            opening = random_opening(5, 5, 4, rng) if panel == "paired" else []
            for side in (True, False):
                result.append(dict(index=len(result), panel=panel, pair=pair,
                                   a_first=side, seed=seed + len(result),
                                   opening=[list(move) for move in opening]))
    return result


def verify_game(record: dict[str, Any], slot: dict[str, Any]) -> None:
    """Independent Python replay checks moves, final score, sides and timings."""
    if record["slot"] != slot:
        raise ValueError("Completed game's frozen slot changed")
    moves = record["moves"]
    opening = slot["opening"]
    if len(moves) != 60 or moves[:len(opening)] != opening:
        raise ValueError("Expected all 60 edges including the frozen opening")
    board = BoxesBoard(5, 5)
    counts = [0, 0]
    for index, (row, col) in enumerate(moves):
        if index >= len(opening):
            counts[board.player()] += 1
        if not board.play_edge(board.edge_index(row, col)):
            raise ValueError("Illegal benchmark edge")
    first_margin = round(board.score())
    margin = first_margin * (1 if slot["a_first"] else -1)
    expected_result = f"{'B' if first_margin > 0 else 'W'}+{abs(first_margin):.1f}"
    if (not board.is_game_over() or record["a_margin"] != margin or
            record["result"] != expected_result or record["termination"] != "all_edges"):
        raise ValueError("Benchmark outcome disagrees with independent full-game replay")
    a, b = (0, 1) if slot["a_first"] else (1, 0)
    for side, player in (("a", a), ("b", b)):
        seconds = record[f"{side}_seconds"]
        if (record[f"{side}_moves"] != counts[player] or
                not math.isfinite(seconds) or seconds < 0):
            raise ValueError("Invalid move count/timing")


def play_slot(agent: Agent, opponent: QdabAgent, slot: dict[str, Any]) -> dict[str, Any]:
    """Use normal gameplay until every edge is drawn, including retained turns."""
    first, second = (agent, opponent) if slot["a_first"] else (opponent, agent)
    game = play_game(first, second, board_size=5, board_cols=5, max_moves=60,
                     seed=slot["seed"], collect_boards=False, komi=0,
                     game=get_game("boxes"), start_moves=[tuple(m) for m in slot["opening"]],
                     render_debug_on_error=False, stop_when_solved=False,
                     black_agent_name="autoboxes" if slot["a_first"] else "Qdab-quctann",
                     white_agent_name="Qdab-quctann" if slot["a_first"] else "autoboxes")
    first_margin = int(float(game.result[2:])) * (1 if game.result[0] == "B" else -1)
    times = [(game.black_move_seconds, game.black_move_count),
             (game.white_move_seconds, game.white_move_count)]
    (a_sec, a_moves), (b_sec, b_moves) = times if slot["a_first"] else times[::-1]
    result = dict(slot=slot, moves=[list(m) for m in [*game.start_moves, *game.moves]],
                  a_margin=first_margin * (1 if slot["a_first"] else -1), result=game.result,
                  termination=game.termination, a_seconds=a_sec, a_moves=a_moves,
                  b_seconds=b_sec, b_moves=b_moves, qdab_decisions=opponent.decisions)
    verify_game(result, slot)
    return result


def completed(output: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Only a contiguous, verified prefix counts; an unpaired game survives resume."""
    rows = []
    for slot in config["schedule"]:
        path = output / "games" / f"{slot['index']:04d}.json"
        if not path.exists():
            break
        row = json.loads(path.read_text())
        if row["config_sha256"] != sha256(output / "config.json"):
            raise ValueError("Completed game belongs to another protocol")
        verify_game(row, slot)
        rows.append(row)
    if len(list((output / "games").glob("*.json"))) != len(rows):
        raise ValueError("Noncontiguous or unexpected completed games")
    return rows


def report(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Incomplete panels have counts, but no final rating estimate."""
    rows = completed(output, config)
    panels: dict[str, Any] = {}
    for panel in ("paired", "empty"):
        selected = [row for row in rows if row["slot"]["panel"] == panel]
        expected = sum(slot["panel"] == panel for slot in config["schedule"])
        margins = [row["a_margin"] for row in selected]
        stats: dict[str, Any] = dict(planned=expected, completed=len(selected),
                                     wins=sum(m > 0 for m in margins),
                                     losses=sum(m < 0 for m in margins),
                                     status=("complete" if len(selected) == expected
                                             else "incomplete"))
        if selected:
            stats["mean_margin"] = float(np.mean(margins))
            for side in ("a", "b"):
                stats[f"{side}_sec_per_move"] = sum(r[f"{side}_seconds"] for r in selected) / max(
                    1, sum(r[f"{side}_moves"] for r in selected))
            stats["score"] = sum((m > 0) + .5 * (m == 0) for m in margins) / len(margins)
        if panel == "paired" and len(selected) == expected:
            stats.update(summarize_margins(margins, seed=config["seed"]))
        if panel == "empty":
            stats["scope"] = "Repeated empty-board starts; descriptive only, excluded from Elo"
        panels[panel] = stats
    result = dict(status="complete" if len(rows) == len(config["schedule"]) else "incomplete",
                  panels=panels, checkpoint_sha256=config["checkpoint_sha256"],
                  anchor="Pinned Qdab quctann at the recorded settings, assigned relative Elo 0",
                  independent_replay_verified=True,
                  scope="Configured-strength match, not equal-time or a published absolute rating",
                  training_use="Evaluation only. Collect separate games for training.",
                  failed_or_interrupted_attempts=sum(
                      json.loads(p.read_text())["status"] != "complete"
                      for p in (output / "attempts").glob("*/attempt.json")))
    write_json(output / "summary.json", result)
    return result


def runtime() -> dict[str, Any]:
    """Record the software and hardware used by both timed players."""
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                platform=platform.platform(), processor=platform.processor(),
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                extension_sha256=sha256(Path(alpha_go_cpp.__file__)))


def initialize(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Freeze weights, openings, recipe and implementation before the first game."""
    if not (math.isfinite(args.seconds) and 0 < args.seconds <= args.hard_timeout
            and math.isfinite(args.hard_timeout)):
        raise ValueError("Require 0 < seconds <= hard-timeout, both finite")
    slots = schedule(args.games, args.empty_games, args.seed)
    source = args.checkpoint.resolve()
    digest = sha256(source)
    if args.checkpoint_sha256 and digest != args.checkpoint_sha256:
        raise ValueError("Checkpoint differs from the selected baseline")
    device = pick_device("cpu" if args.cpu else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; --cpu is available for a deliberate CPU run")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="qdab-match-", dir=args.output_root)).resolve()
    frozen = output / "inputs" / "checkpoint.pt"
    frozen.parent.mkdir()
    shutil.copy2(source, frozen)
    if sha256(frozen) != digest:
        raise ValueError("Checkpoint changed while copying")
    paths = sorted(Path("src/alpha_go").rglob("*.py"))
    config = dict(version=1, checkpoint=str(frozen), checkpoint_sha256=digest,
                  source_checkpoint=str(source), schedule=slots, seed=args.seed,
                  release=str(args.release.resolve()), seconds=args.seconds,
                  hard_timeout=args.hard_timeout, device=str(device), search=SEARCH,
                  evaluator=EVALUATOR, qdab_commit=COMMIT, qdab_artifacts=ARTIFACTS,
                  qdab_archive_sha256=ARCHIVE_SHA256,
                  qdab_seed_control="not exposed by upstream RPC; timed searches vary across runs",
                  code_sha256={str(path): sha256(path) for path in paths}, runtime=runtime(),
                  code_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], text=True).strip(),
                  git_status=subprocess.check_output(["git", "status", "--short"], text=True),
                  max_attempts_per_game=3)
    write_json(output / "config.json", config)
    command = shlex.join(["uv", "run", "python", "-u", "-m",
                          "alpha_go.boxes.qdab_benchmark", "--resume", str(output)])
    (output / "resume.sh").write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\ncd {shlex.quote(str(Path.cwd()))}\n"
        f"export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n{command}\n")
    return output, config


def run_games(output: Path, config: dict[str, Any], agent: Agent) -> None:
    """Commit each game independently; a failed game is retried only on explicit resume."""
    rows = completed(output, config)
    for slot in config["schedule"][len(rows):]:
        previous = list((output / "attempts").glob(f"game-{slot['index']:04d}-*"))
        if len(previous) >= config["max_attempts_per_game"]:
            raise RuntimeError("Repeated game failure; inspect attempts before continuing")
        attempt = output / "attempts" / f"game-{slot['index']:04d}-{len(previous) + 1}"
        attempt.mkdir(parents=True)
        metadata = dict(slot=slot, status="running", started_unix=time.time())
        write_json(attempt / "attempt.json", metadata)
        opponent = OpeningQdabAgent(Path(config["release"]), config["seconds"],
                                   config["hard_timeout"], attempt / "qdab.log",
                                   opening=slot["opening"])
        print(f"Game {slot['index'] + 1}/{len(config['schedule'])}: {slot['panel']}, "
              f"autoboxes {'first' if slot['a_first'] else 'second'}", flush=True)
        started = time.monotonic()
        try:
            row = play_slot(agent, opponent, slot)
            row.update(config_sha256=sha256(output / "config.json"), attempt=attempt.name)
            write_json(output / "games" / f"{slot['index']:04d}.json", row)
            metadata["status"] = "complete"
        except BaseException as error:
            metadata.update(status="failed", error_type=type(error).__name__, error=str(error),
                            traceback=traceback.format_exc(),
                            prefix=getattr(opponent, "prefix", []), decisions=opponent.decisions)
            raise
        finally:
            metadata["elapsed_seconds"] = time.monotonic() - started
            write_json(attempt / "attempt.json", metadata)
        report(output, config)
        print(f"Saved game {slot['index'] + 1}; margin {row['a_margin']:+}; "
              f"{metadata['elapsed_seconds']:.0f}s", flush=True)


def run(output: Path, config: dict[str, Any]) -> None:
    """One serial match: Qdab owns a fixed port, with a separate deadline per RPC."""
    checked_files(config["code_sha256"])
    checked_files({config["checkpoint"]: config["checkpoint_sha256"]})
    if runtime() != config["runtime"]:
        raise ValueError("Runtime changed; resume requires the recorded software/hardware")
    release = install_release(Path(config["release"]))
    if config["qdab_artifacts"] != ARTIFACTS or config["qdab_commit"] != COMMIT:
        raise ValueError("Qdab release changed")
    print(f"Verified Qdab: {release}", flush=True)
    if len(completed(output, config)) == len(config["schedule"]):
        report(output, config)
        return
    device = pick_device(config["device"])
    model = load_boxes_net(config["checkpoint"], device)
    if (model.rows, model.cols) != (5, 5):
        raise ValueError("Expected a 5x5 checkpoint")
    engine = PlaneBatchedEngine(model, device, batch_size=64)
    agent = BoxesMCTSAgent(BoxesEngineEvaluator(engine, checkpoint_path=config["checkpoint"],
                                              **config["evaluator"]), **config["search"])
    engine.start()
    try:
        run_games(output, config, agent)
    finally:
        engine.stop()


def main() -> None:
    def terminate(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt("SIGTERM: preserving partial match and stopping Qdab")

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("work/qdab-benchmarks"))
    parser.add_argument("--release", type=Path, default=Path("work/qdab-release"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--empty-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--hard-timeout", type=float, default=30)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / "config.json").read_text())
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for a new run")
        output, config = initialize(args)
    print(f"Results: {output}\nResume: bash {output / 'resume.sh'}", flush=True)
    print("Full games against Qdab; no training. Complete fixed sample before interpreting Elo.",
          flush=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(output, config)
        finally:
            report(output, config)
            package_results(output)
    print(json.dumps(json.loads((output / "summary.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
