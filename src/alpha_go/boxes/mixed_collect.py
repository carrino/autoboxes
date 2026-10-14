"""Optional training collection against self, frozen checkpoints and Boxes baselines.

Games retain their complete replay. ``train_mask`` selects only the learner's turns
with recorded search visits (both seats in self-play), so a baseline's played moves
never become policy targets. Feed this directory to BoxesDataset or EXTRA_DATA.
Opening pairs, opponent quotas and seeds are deterministic; full trajectories are
not bit-reproducible because C++ MCTS seeds its own RNG and classic is time-limited.

Run sequentially on one GPU, retaining at most the learner and one opponent net.
This is an optional collector; the existing self-play loop is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import torch
from numpy.typing import NDArray

from alpha_go.agents import Agent, get_agent
from alpha_go.boxes.arena import random_opening
from alpha_go.boxes.nn_agent import (
    BoxesLeafEvaluator,
    BoxesMCTSAgent,
    add_search_flags,
    load_boxes_net,
    pick_device,
    search_flags,
)
from alpha_go.boxes.rules import geometry
from alpha_go.boxes.symmetry import edge_permutation, transforms
from alpha_go.gameplay import GameRecord, play_game, save_game_data
from alpha_go.games import get_game


@dataclass(frozen=True)
class OpeningPair:
    """Two games with the same opponent and opening, swapping the learner's seat."""

    index: int
    opponent: int
    seed: int
    opening: list[tuple[int, int]]
    split_group: str


def pair_schedule(rows: int, cols: int, num_games: int, weights: list[float],
                  opening_moves: int, seed: int) -> list[OpeningPair]:
    """Allocate whole pairs by largest remainder, then shuffle their opponents.

    Nonempty opening families are grouped across duplicates and board symmetries.
    Empty-board pairs use their seed: grouping every empty game together would make
    replay validation unusable. These splits remain diagnostics, not external tests.
    """
    assert num_games > 0 and num_games % 2 == 0, "num-games must be positive and even"
    assert weights and all(math.isfinite(w) and w >= 0 for w in weights)
    assert sum(weights) > 0, "at least one opponent weight must be positive"
    geo = geometry(rows, cols)
    assert 0 <= opening_moves < geo.num_edges, "opening must leave moves to play"
    assert seed >= 0
    pairs = num_games // 2
    quota = np.asarray(weights, dtype=float) / sum(weights) * pairs
    counts = np.floor(quota).astype(int)
    for i in np.argsort(-(quota - counts), kind="stable")[:pairs - int(counts.sum())]:
        counts[i] += 1
    rng = np.random.default_rng(seed)
    opponents = rng.permutation(np.repeat(np.arange(len(weights)), counts))
    plans = []
    for i, opponent in enumerate(opponents):
        pair_seed = seed * 1_000_003 + 2 * i
        opening = random_opening(rows, cols, opening_moves, rng)
        if opening:
            edges = np.asarray([geo.lattice_edge[r, c] for r, c in opening])
            family = min(edge_permutation(geo, k)[edges].astype("<i4").tobytes()
                         for k in transforms(rows, cols))
        else:
            family = f"empty-pair:{pair_seed}".encode()
        group = hashlib.sha256(f"mixed:{rows}x{cols}:".encode() + family).hexdigest()
        plans.append(OpeningPair(i, int(opponent), pair_seed, opening, group))
    return plans


def learner_train_mask(record: GameRecord, learner_color: int | None) -> NDArray[np.bool_]:
    """Keep searched learner turns, using recorded movers rather than ply parity.

    Solver-optimal visit rows count as valid targets; unsearched forced moves do not.
    ``None`` selects both seats, and is reserved for current-checkpoint self-play.
    """
    assert len(record.move_metrics) == len(record.to_play) == len(record.moves)
    return np.asarray([
        (learner_color is None or mover == learner_color)
        and metric.visit_counts is not None and metric.visit_counts.sum() > 0
        for mover, metric in zip(record.to_play, record.move_metrics)
    ], dtype=np.bool_)


def file_sha256(path: Path) -> str:
    """Hash file contents without loading checkpoint-sized files into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_training_game(record: GameRecord, output: Path, index: int,
                       learner_color: int | None, metadata: dict[str, Any]) -> Path:
    """Save the standard full game plus an explicit training mask and provenance."""
    path = save_game_data(record, output, index, "mixed")
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays.update(metadata, train_mask=learner_train_mask(record, learner_color),
                  learner_color=-1 if learner_color is None else learner_color,
                  collection_source="mixed-opponent-v1")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as target:
        np.savez_compressed(target, **arrays)
    temporary.replace(path)
    return path


def parser() -> argparse.ArgumentParser:
    """CLI for a fresh, explicit output directory; no resume or overwrite behavior."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--opponent", action="append", default=None,
                   help="Repeat: self, boxes-classic, boxes-greedy, boxes-random, or ckpt:PATH")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Relative pair quotas in --opponent order (default: equal)")
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--num-games", type=int, default=100)
    p.add_argument("--opening-moves", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--solver", type=int, default=0)
    p.add_argument("--solver-budget", type=int, default=20_000)
    p.add_argument("--merge-equivalent", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-cutoff", type=int, default=8)
    p.add_argument("--noise-weight", type=float, default=0.25)
    p.add_argument("--noise-alpha", type=float, default=0.5)
    p.add_argument("--stop-when-solved", action="store_true")
    p.add_argument("--cpu", action="store_true")
    add_search_flags(p)
    return p


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the declared mixture with complete replay and learner-only targets."""
    cols = args.cols or args.rows
    opponents: list[str] = args.opponent or ["self", "boxes-classic"]
    weights = args.weights or [1.0] * len(opponents)
    assert len(weights) == len(opponents), "one weight per opponent is required"
    assert len(set(opponents)) == len(opponents), "duplicate opponents obscure mixture quotas"
    assert args.sims >= 2 and args.solver_budget > 0 and args.solver >= 0
    assert args.temperature >= 0 and 0 <= args.noise_weight <= 1 and args.noise_alpha > 0
    assert not args.output.exists(), f"output already exists: {args.output}; choose a fresh path"
    plans = pair_schedule(args.rows, cols, args.num_games, weights, args.opening_moves, args.seed)
    paths = {"learner": args.checkpoint.resolve()}
    for opponent in opponents:
        if opponent.startswith("ckpt:"):
            paths[opponent] = Path(opponent[len("ckpt:"):]).resolve()
        else:
            assert opponent in ("self", "boxes-classic", "boxes-greedy", "boxes-random"), opponent
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    device = pick_device("cpu" if args.cpu else None)
    assert args.cpu or device.type == "cuda", "CUDA not available; pass --cpu for a tiny smoke run"
    mcts_flags, evaluator_flags = search_flags(args)
    model = load_boxes_net(paths["learner"], device)
    assert (model.rows, model.cols) == (args.rows, cols), "learner board shape mismatch"
    evaluator = BoxesLeafEvaluator(model, device, checkpoint_path=str(paths["learner"]),
                                   **evaluator_flags)
    mcts_kwargs = dict(num_simulations=args.sims, temperature=args.temperature,
                       temperature_cutoff=args.temperature_cutoff,
                       add_noise=args.noise_weight > 0, noise_alpha=args.noise_alpha,
                       noise_weight=args.noise_weight, solver_max_undrawn=args.solver,
                       solver_node_budget=args.solver_budget,
                       merge_equivalent=args.merge_equivalent, **mcts_flags)
    source = Path(__file__).resolve().parent
    source_paths = [*source.glob("*.py"), source.parent / "gameplay.py"]
    source_hashes = {str(p.relative_to(source.parent)): file_sha256(p) for p in source_paths}
    source_hashes["alpha_go_cpp"] = file_sha256(Path(alpha_go_cpp.__file__))
    provenance: dict[str, Any] = {
        "schema": "mixed-opponent-v1", "status": "running", "arguments": vars(args),
        "opponents": opponents, "weights": weights, "checkpoint_sha256": hashes,
        "checkpoint_paths": paths, "source_sha256": source_hashes,
        "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "num_workers": 1, "pairs": [asdict(p) for p in plans], "games": [],
        "root_value_semantics": ("shaped_utility" if args.margin_utility_lambda
                                 else "win_probability"),
        "reproducibility": "Fixed quotas/openings/seeds; MCTS RNG and classic timing may vary",
    }
    args.output.mkdir(parents=True)
    manifest = args.output / "collection.json"
    manifest.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    parameter_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"Mixed collection: {args.num_games} games on {device}, one worker, "
          f"learner parameters {parameter_mb:.1f} MB; at most two resident networks", flush=True)
    start = time.monotonic()
    learner = BoxesMCTSAgent(evaluator, **mcts_kwargs)
    # Execute opponent blocks to avoid repeatedly loading frozen checkpoints. Pair indices
    # and per-game seeds stay tied to the saved plan, independent of this execution order.
    for opponent_index, spec in enumerate(opponents):
        selected = [pair for pair in plans if pair.opponent == opponent_index]
        if not selected:
            continue
        opponent_agent: Agent
        if spec == "self":
            opponent_agent = BoxesMCTSAgent(evaluator, **mcts_kwargs)
        elif spec.startswith("ckpt:"):
            other = load_boxes_net(paths[spec], device)
            assert (other.rows, other.cols) == (args.rows, cols), "opponent board shape mismatch"
            opponent_agent = BoxesMCTSAgent(BoxesLeafEvaluator(
                other, device, checkpoint_path=str(paths[spec]), **evaluator_flags), **mcts_kwargs)
            del other
        else:
            opponent_agent = get_agent(spec)
        for pair in selected:
            for color in (0, 1):
                first, second = ((learner, opponent_agent) if color == 0
                                 else (opponent_agent, learner))
                record = play_game(
                    first, second, board_size=args.rows, board_cols=cols, komi=0.0,
                    max_moves=geometry(args.rows, cols).num_edges, seed=pair.seed + color,
                    game=get_game("boxes"), start_moves=pair.opening,
                    collect_boards=True, collect_metrics=True, render_debug_on_error=False,
                    black_agent_name="learner" if color == 0 else spec,
                    white_agent_name=spec if color == 0 else "learner",
                    black_is_teacher=color == 0 or spec == "self",
                    white_is_teacher=color == 1 or spec == "self",
                    stop_when_solved=args.stop_when_solved,
                )
                mask_color = None if spec == "self" else color
                path = save_training_game(record, args.output, 2 * pair.index + color, mask_color, {
                    "split_group": pair.split_group, "pair_index": pair.index,
                    "game_seed": pair.seed + color, "opponent_spec": spec,
                    "learner_checkpoint_sha256": hashes["learner"],
                    "opponent_checkpoint_sha256": hashes["learner"] if spec == "self"
                    else hashes.get(spec, ""),
                    "root_value_semantics": provenance["root_value_semantics"],
                })
                kept = int(learner_train_mask(record, mask_color).sum())
                provenance["games"].append({"file": path.name, "sha256": file_sha256(path),
                                             "positions": len(record.moves), "training_rows": kept,
                                             "pair": pair.index, "learner_color": color,
                                             "opponent": spec, "result": record.result})
                print(f"  {len(provenance['games'])}/{args.num_games}: {spec}, "
                      f"learner color {color}, {kept} training rows, "
                      f"{time.monotonic() - start:.0f}s", flush=True)
                manifest.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
        del first, second, opponent_agent
    provenance["status"] = "complete"
    provenance["elapsed_seconds"] = time.monotonic() - start
    manifest.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    return provenance


def main() -> None:
    """Collect optional mixed-opponent replay without changing the default loop."""
    collect(parser().parse_args())


if __name__ == "__main__":
    main()
