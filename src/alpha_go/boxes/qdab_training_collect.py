"""Fresh Qdab games and approximate expert corrections, separate from evaluation.

Full games are committed before optional teacher queries. Existing learner search
targets and Qdab recommendations remain separate; this collector does not train.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from alpha_go.agents.base import Agent
from alpha_go.boxes.arena import random_opening
from alpha_go.boxes.curriculum import checked_files, package_results, write_json
from alpha_go.boxes.inference import PlaneBatchedEngine
from alpha_go.boxes.nn_agent import (
    BoxesEngineEvaluator,
    BoxesMCTSAgent,
    load_boxes_net,
    pick_device,
)
from alpha_go.boxes.qdab import (
    ARTIFACTS,
    COMMIT,
    decode_result,
    install_release,
    query,
    replay_prefix,
    sha256,
)
from alpha_go.boxes.qdab_benchmark import (
    EVALUATOR,
    SEARCH,
    OpeningQdabAgent,
    runtime,
    schedule,
    verify_game,
)
from alpha_go.boxes.rules import BoxesBoard
from alpha_go.boxes.symmetry import edge_permutation, transforms
from alpha_go.gameplay import play_game
from alpha_go.games import get_game


def opening_family(opening: list[list[int]]) -> str:
    """Group symmetry/transposition-equivalent four-edge starting boards together."""
    board = BoxesBoard(5)
    edges = [board.edge_index(*move) for move in opening]
    canonical = min(bytes(sorted(int(edge_permutation(board.geo, k)[e]) for e in edges))
                    for k in transforms(5, 5))
    return "qdab-opening-" + hashlib.sha256(canonical).hexdigest()


def fresh_schedule(games: int, seed: int) -> list[dict[str, Any]]:
    """Freeze independent training families and a family split before outcomes exist."""
    if games < 1:
        raise ValueError("Require at least one game")
    excluded = {opening_family(slot["opening"])
                for count, benchmark_seed in ((100, 20261112), (8, 20261116))
                for slot in schedule(count, 0, benchmark_seed)}
    rng = np.random.default_rng(seed)
    slots: list[dict[str, Any]] = []
    while len(slots) < games:
        opening = [list(move) for move in random_opening(5, 5, 4, rng)]
        family = opening_family(opening)
        if family in excluded:
            continue
        excluded.add(family)
        held_out = int(hashlib.sha256(("qdab-split-v1:" + family).encode()).hexdigest()[:8], 16)
        index = len(slots)
        slots.append(dict(index=index, panel="training", a_first=index % 2 == 0,
                          seed=seed + index, opening=opening, source_family=family,
                          split="validation" if held_out < 0.2 * 2**32 else "training"))
    return slots


def play_training_slot(agent: Agent, opponent: OpeningQdabAgent,
                       slot: dict[str, Any]) -> dict[str, Any]:
    """Record existing learner searches using the shared full-game implementation."""
    first, second = (agent, opponent) if slot["a_first"] else (opponent, agent)
    record = play_game(first, second, board_size=5, board_cols=5, max_moves=60,
                       seed=slot["seed"], collect_boards=False, collect_metrics=True, komi=0,
                       game=get_game("boxes"), start_moves=[tuple(m) for m in slot["opening"]],
                       render_debug_on_error=False, stop_when_solved=False,
                       black_agent_name="autoboxes" if slot["a_first"] else "Qdab-quctann",
                       white_agent_name="Qdab-quctann" if slot["a_first"] else "autoboxes")
    board = BoxesBoard(5)
    prefix = []
    for move in record.start_moves:
        edge = board.edge_index(*move)
        assert board.play_edge(edge)
        prefix.append(edge)
    decisions = []
    learner_player = 0 if slot["a_first"] else 1
    for move, metric in zip(record.moves, record.move_metrics, strict=True):
        edge = board.edge_index(*move)
        visits = metric.visit_counts
        if board.player() == learner_player and visits is not None and visits.sum() > 0:
            decisions.append(dict(move_count=len(prefix), prefix=list(prefix), rows=5, cols=5,
                                  scores=list(board.boxes), to_play=board.to_play(),
                                  undrawn=60 - len(prefix), learner_edge=edge,
                                  policy_target=(visits / visits.sum()).tolist(),
                                  visits=visits.tolist(),
                                  raw_visits=(metric.raw_visit_counts.tolist()
                                              if metric.raw_visit_counts is not None else None)))
        assert board.play_edge(edge)
        prefix.append(edge)
    first_margin = int(board.score())
    times = [(record.black_move_seconds, record.black_move_count),
             (record.white_move_seconds, record.white_move_count)]
    (a_sec, a_moves), (b_sec, b_moves) = times if slot["a_first"] else times[::-1]
    row = dict(slot=slot, moves=[list(m) for m in [*record.start_moves, *record.moves]],
               a_margin=first_margin * (1 if slot["a_first"] else -1), result=record.result,
               termination=record.termination, a_seconds=a_sec, a_moves=a_moves,
               b_seconds=b_sec, b_moves=b_moves, qdab_decisions=opponent.decisions,
               learner_decisions=decisions, source_family=slot["source_family"],
               split=slot["split"], outcome_kind="observed_complete_game_not_exact_root_value")
    verify_training_game(row, slot)
    return row


def verify_training_game(row: dict[str, Any], slot: dict[str, Any]) -> None:
    """Reject illegal history, changed split, invalid target mass or root identity."""
    verify_game(row, slot)
    if row["source_family"] != opening_family(slot["opening"]) or row["split"] != slot["split"]:
        raise ValueError("Training family or split changed")
    board = BoxesBoard(5)
    edges = [board.edge_index(*move) for move in row["moves"]]
    for decision in row["learner_decisions"]:
        prefix = edges[:decision["move_count"]]
        ref, _ = replay_prefix(prefix)
        target = np.asarray(decision["policy_target"])
        legal = np.zeros(60, dtype=bool)
        legal[ref.get_legal_moves_flat()] = True
        if (decision["prefix"] != prefix or decision["scores"] != ref.boxes or
                decision["to_play"] != ref.to_play() or
                decision["undrawn"] != 60 - len(prefix) or
                decision["learner_edge"] != edges[len(prefix)] or
                ref.player() != (0 if slot["a_first"] else 1)):
            raise ValueError("Learner decision disagrees with independent replay")
        if (target.shape != (60,) or not np.isfinite(target).all() or
                np.any(target < 0) or not np.isclose(target.sum(), 1) or
                np.any(target[~legal] != 0)):
            raise ValueError("Learner policy must be normalized over legal edges")
        visits = np.asarray(decision["visits"])
        if (visits.shape != (60,) or visits.dtype.kind not in "iu" or
                np.any(visits < 0) or visits.sum() <= 0 or np.any(visits[~legal] != 0) or
                not np.allclose(target, visits / visits.sum())):
            raise ValueError("Learner policy does not match legal integer visit counts")
        if decision["raw_visits"] is not None:
            raw = np.asarray(decision["raw_visits"])
            if (raw.shape != (60,) or raw.dtype.kind not in "iu" or np.any(raw < visits) or
                    np.any(raw[~legal] != 0)):
                raise ValueError("Raw exploration visits disagree with filtered visit counts")


def teacher_roots(row: dict[str, Any]) -> list[dict[str, Any]]:
    """At most one searched learner decision from each predeclared early band."""
    selected = []
    for lower, upper in ((48, 50), (45, 47), (42, 44), (38, 41)):
        eligible = [root for root in row["learner_decisions"] if lower <= root["undrawn"] <= upper]
        if eligible:
            selected.append(eligible[0])
    return selected


def collect_teachers(output: Path, config: dict[str, Any], row: dict[str, Any]) -> None:
    """Save resumable approximate Qdab advice without replacing learner targets."""
    for root in teacher_roots(row):
        token = f"{row['slot']['index']:04d}-{root['move_count']:02d}"
        path = output / "teacher" / f"{token}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            verify_teacher(saved, root, row, output, config)
            continue
        failures = list((output / "teacher" / "failures").glob(f"{token}-*.json"))
        if len(failures) >= 3:
            raise RuntimeError("Three teacher attempts failed; inspect before retrying")
        try:
            reply = query(Path(config["release"]), root["prefix"], config["seconds"],
                          config["hard_timeout"], output / "teacher" / f"{token}.log")
        except BaseException as error:
            write_json(output / "teacher" / "failures" / f"{token}-{len(failures) + 1}.json",
                       dict(root=root, error_type=type(error).__name__, error=str(error),
                            traceback=traceback.format_exc()))
            raise
        saved = dict(schema=1, root=root, source_family=row["source_family"],
                     split=row["split"], source_game=row["slot"]["index"],
                     teacher_kind="approximate_qdab_action", exact_policy=False,
                     teacher_edge=reply["edges"][0], teacher_reply=reply,
                     config_sha256=sha256(output / "config.json"),
                     checkpoint_sha256=config["checkpoint_sha256"])
        verify_teacher(saved, root, row, output, config)
        write_json(path, saved)
        print(f"  saved teacher {token}: learner {root['learner_edge']}, "
              f"Qdab {reply['edges']}; {root['undrawn']} undrawn", flush=True)


def verify_teacher(saved: dict[str, Any], root: dict[str, Any], row: dict[str, Any],
                   output: Path, config: dict[str, Any]) -> None:
    """Replay approximate recommendations and reject changed roots or capture bundles."""
    if (saved["root"] != root or saved["config_sha256"] != sha256(output / "config.json") or
            saved["checkpoint_sha256"] != config["checkpoint_sha256"] or
            saved["source_family"] != row["source_family"] or saved["split"] != row["split"] or
            saved["source_game"] != row["slot"]["index"] or saved["exact_policy"] is not False or
            saved["teacher_kind"] != "approximate_qdab_action"):
        raise ValueError("Saved teacher query changed its root, family, recipe or target semantics")
    ref, state = replay_prefix(root["prefix"])
    reply = saved["teacher_reply"]
    decoded = decode_result(reply["rpc_response"]["result"], ref)
    if (decoded != reply["edges"] or saved["teacher_edge"] != decoded[0] or
            reply["rpc_request"]["params"][0]["Board"] != state):
        raise ValueError("Teacher reply disagrees with replayed board or legal capture bundle")


def report(output: Path, config: dict[str, Any]) -> None:
    """Report data coverage only; this training collection is not a rating sample."""
    rows = [json.loads(p.read_text()) for p in sorted((output / "games").glob("*.json"))]
    teachers = [json.loads(p.read_text()) for p in (output / "teacher").glob("*.json")]
    write_json(output / "summary.json", dict(
        games_completed=len(rows), games_planned=len(config["schedule"]),
        families=len({r["source_family"] for r in rows}),
        split_games={split: sum(r["split"] == split for r in rows)
                     for split in ("training", "validation")},
        learner_searched_roots=sum(len(r["learner_decisions"]) for r in rows),
        teacher_queries=len(teachers),
        teacher_disagreements=sum(t["teacher_edge"] != t["root"]["learner_edge"] for t in teachers),
        training_performed=False, weights_changed=False,
        scope="Fresh training/development data; no rating inference; Qdab advice is approximate",
        exclusions="Known benchmark canonical four-edge opening families; later-state overlap "
                   "must be checked against the completed GPU benchmark before training"))


def run(output: Path, config: dict[str, Any]) -> None:
    """Finish complete games within a bounded session; each completed game is durable."""
    checked_files(config["code_sha256"])
    checked_files({config["checkpoint"]: config["checkpoint_sha256"]})
    if runtime() != config["runtime"]:
        raise ValueError("Runtime changed; resume requires the recorded software/hardware")
    install_release(Path(config["release"]))
    model = load_boxes_net(config["checkpoint"], pick_device(config["device"]))
    engine = PlaneBatchedEngine(model, pick_device(config["device"]), batch_size=64)
    agent = BoxesMCTSAgent(BoxesEngineEvaluator(engine, checkpoint_path=config["checkpoint"],
                                              **config["evaluator"]), **config["search"])
    deadline = time.monotonic() + config["hours"] * 3600
    engine.start()
    try:
        for slot in config["schedule"]:
            path = output / "games" / f"{slot['index']:04d}.json"
            if not path.exists() and time.monotonic() >= deadline:
                print("Session budget reached; completed games and teacher queries preserved.",
                      flush=True)
                break
            if path.exists():
                row = json.loads(path.read_text())
                if row["config_sha256"] != sha256(output / "config.json"):
                    raise ValueError("Saved game belongs to another collection recipe")
                verify_training_game(row, slot)
            else:
                previous = list((output / "attempts").glob(f"game-{slot['index']:04d}-*"))
                if len(previous) >= 3:
                    raise RuntimeError("Three game attempts failed; inspect before retrying")
                attempt = Path(tempfile.mkdtemp(prefix=f"game-{slot['index']:04d}-",
                                                dir=output / "attempts"))
                opponent = OpeningQdabAgent(Path(config["release"]), config["seconds"],
                                           config["hard_timeout"], attempt / "qdab.log",
                                           opening=slot["opening"])
                print(f"Training game {slot['index'] + 1}/{len(config['schedule'])}; "
                      f"{slot['split']}; learner {'first' if slot['a_first'] else 'second'}",
                      flush=True)
                started = time.monotonic()
                try:
                    row = play_training_slot(agent, opponent, slot)
                except BaseException as error:
                    write_json(attempt / "failure.json", dict(error_type=type(error).__name__,
                               error=str(error), traceback=traceback.format_exc(),
                               prefix=getattr(opponent, "prefix", []),
                               qdab_decisions=opponent.decisions))
                    raise
                row.update(config_sha256=sha256(output / "config.json"),
                           checkpoint_sha256=config["checkpoint_sha256"],
                           elapsed_seconds=time.monotonic() - started, attempt=attempt.name)
                write_json(path, row)
                print(f"Saved full game {slot['index'] + 1}; "
                      f"{len(row['learner_decisions'])} learner targets; "
                      f"margin {row['a_margin']:+}; {row['elapsed_seconds']:.0f}s", flush=True)
            collect_teachers(output, config, row)
            report(output, config)
    finally:
        engine.stop()


def initialize(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Freeze weights, split, recipes and source hashes before any data collection."""
    if not all(math.isfinite(x) and x > 0 for x in (args.hours, args.seconds, args.hard_timeout)):
        raise ValueError("Budgets must be finite and positive")
    if args.seconds > args.hard_timeout:
        raise ValueError("Qdab hard timeout must cover requested time")
    digest = sha256(args.checkpoint)
    if args.checkpoint_sha256 and digest != args.checkpoint_sha256:
        raise ValueError("Unexpected checkpoint")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="qdab-training-", dir=args.output_root)).resolve()
    for directory in ("inputs", "games", "teacher", "attempts"):
        (output / directory).mkdir()
    frozen = output / "inputs" / "checkpoint.pt"
    shutil.copy2(args.checkpoint, frozen)
    if sha256(frozen) != digest:
        raise ValueError("Checkpoint changed while copying")
    config = dict(schema=1, checkpoint=str(frozen), checkpoint_sha256=digest,
                  schedule=fresh_schedule(args.games, args.seed), seed=args.seed,
                  release=str(args.release.resolve()), seconds=args.seconds,
                  hard_timeout=args.hard_timeout, hours=args.hours,
                  search=SEARCH, evaluator=EVALUATOR, device="cpu", runtime=runtime(),
                  qdab_commit=COMMIT, qdab_artifacts=ARTIFACTS,
                  teacher_bands=[[48, 50], [45, 47], [42, 44], [38, 41]],
                  code_sha256={str(p): sha256(p)
                               for p in sorted(Path("src/alpha_go").rglob("*.py"))},
                  code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       text=True).strip(),
                  git_status=subprocess.check_output(["git", "status", "--short"], text=True),
                  qdab_seed_control="Wall-clock search seed not exposed by upstream",
                  scope="Fresh training/development collection, separate from benchmark")
    write_json(output / "config.json", config)
    command = shlex.join([sys.executable, "-u", "-m", "alpha_go.boxes.qdab_training_collect",
                          "--resume", str(output)])
    (output / "resume.sh").write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\ncd {shlex.quote(str(Path.cwd()))}\n"
        f"export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n{command}\n")
    return output, config


def main() -> None:
    """Run a bounded CPU collection or resume saved games and teacher queries."""
    def terminate(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt("SIGTERM: preserving completed training games")

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("work/qdab-training"))
    parser.add_argument("--release", type=Path, default=Path("work/qdab-release"))
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--hard-timeout", type=float, default=30)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / "config.json").read_text())
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for new collection")
        output, config = initialize(args)
    print(f"Training data: {output}\nResume: bash {output / 'resume.sh'}", flush=True)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(output, config)
        finally:
            report(output, config)
            package_results(output)


if __name__ == "__main__":
    main()
