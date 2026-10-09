"""Head-to-head arena for Boxes agents.

Plays `num_games` games between two registered agents, alternating who moves first, through
`alpha_go.gameplay.play_game`, and reports win rate (with a 95% Wilson interval) and mean
final margin from agent A's point of view.

    uv run -m alpha_go.boxes.arena --rows 3 --a boxes-ab-d4 --b boxes-greedy --num_games 20
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from alpha_go.agents import Agent, get_agent
from alpha_go.boxes import agents as _boxes_agents  # noqa: F401  (registers boxes-* agents)
from alpha_go.gameplay import play_game
from alpha_go.games import get_game

_thread_local = threading.local()


@dataclass
class MatchResult:
    a: str
    b: str
    rows: int
    cols: int
    games: int
    a_wins: int
    b_wins: int
    draws: int
    margins: list[int]  # final margin for A, one per game
    a_seconds_per_move: float
    b_seconds_per_move: float

    @property
    def a_win_rate(self) -> float:
        return (self.a_wins + 0.5 * self.draws) / self.games

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        """95% interval for A's win rate (draws count half)."""
        n, p = self.games, self.a_win_rate
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return centre - half, centre + half

    def summary(self) -> dict[str, float | int | str]:
        lo, hi = self.wilson()
        return {
            "a": self.a, "b": self.b, "board": f"{self.rows}x{self.cols}", "games": self.games,
            "a_wins": self.a_wins, "b_wins": self.b_wins, "draws": self.draws,
            "a_win_rate": round(self.a_win_rate, 4),
            "a_win_rate_ci95": f"[{lo:.3f}, {hi:.3f}]",
            "a_mean_margin": round(float(np.mean(self.margins)), 3),
            "a_margin_std": round(float(np.std(self.margins)), 3),
            "a_sec_per_move": round(self.a_seconds_per_move, 4),
            "b_sec_per_move": round(self.b_seconds_per_move, 4),
        }


def _agents(a: str, b: str) -> tuple[Agent, Agent]:
    """One agent pair per thread (search agents carry tables that are not thread-safe)."""
    if not hasattr(_thread_local, "pair"):
        _thread_local.pair = {}
    pairs: dict[tuple[str, str], tuple[Agent, Agent]] = _thread_local.pair
    if (a, b) not in pairs:
        pairs[(a, b)] = (get_agent(a), get_agent(b))
    return pairs[(a, b)]


def _one_game(
    a: str, b: str, rows: int, cols: int, index: int, seed: int
) -> tuple[int, float, int, float, int]:
    """Play game `index`; A moves first on even indices.

    Returns (margin for A, A seconds, A moves, B seconds, B moves).
    """
    agent_a, agent_b = _agents(a, b)
    a_first = index % 2 == 0
    first, second = (agent_a, agent_b) if a_first else (agent_b, agent_a)
    record = play_game(
        first, second, board_size=rows, board_cols=cols, seed=seed + index,
        max_moves=get_game("boxes").default_max_moves(rows, cols), collect_boards=False,
        game=get_game("boxes"), render_debug_on_error=False,
    )
    first_margin = int(round(float(record.result[2:]))) if record.result != "Draw" else 0
    if record.result.startswith("W+"):
        first_margin = -first_margin
    a_margin = first_margin if a_first else -first_margin
    black = (record.black_move_seconds, record.black_move_count)
    white = (record.white_move_seconds, record.white_move_count)
    (a_sec, a_moves), (b_sec, b_moves) = (black, white) if a_first else (white, black)
    return a_margin, a_sec, a_moves, b_sec, b_moves


def play_match(
    a: str, b: str, rows: int, cols: int | None = None, num_games: int = 20,
    seed: int = 0, num_workers: int = 1,
) -> MatchResult:
    """Play `num_games` games between registered agents `a` and `b`, alternating first move."""
    cols = cols or rows
    start = time.time()
    every = max(1, num_games // 10)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = []
        for r in pool.map(lambda i: _one_game(a, b, rows, cols, i, seed), range(num_games)):
            results.append(r)
            if len(results) % every == 0:
                wins = sum(x[0] > 0 for x in results)
                print(f"  {a} vs {b}: {len(results)}/{num_games} games, {wins} A wins, "
                      f"{time.time() - start:.0f}s", flush=True)
    margins = [r[0] for r in results]
    a_sec = sum(r[1] for r in results) / max(1, sum(r[2] for r in results))
    b_sec = sum(r[3] for r in results) / max(1, sum(r[4] for r in results))
    return MatchResult(
        a=a, b=b, rows=rows, cols=cols, games=num_games,
        a_wins=sum(m > 0 for m in margins), b_wins=sum(m < 0 for m in margins),
        draws=sum(m == 0 for m in margins), margins=margins,
        a_seconds_per_move=a_sec, b_seconds_per_move=b_sec,
    )


def register_checkpoint(name: str, args: argparse.Namespace) -> None:
    """`ckpt:<path>` names become MCTS agents on that checkpoint, one shared GPU engine
    per checkpoint across the game threads, with the search flags of the CLI."""
    from alpha_go.boxes.inference import PlaneBatchedEngine
    from alpha_go.boxes.nn_agent import (
        load_boxes_net,
        pick_device,
        register_boxes_mcts_agent,
        search_flags,
    )
    device = pick_device("cpu" if args.cpu else None)
    path = name[len("ckpt:"):]
    engine = PlaneBatchedEngine(load_boxes_net(path, device), device, batch_size=64)
    engine.start()
    mcts_flags, evaluator_flags = search_flags(args)
    register_boxes_mcts_agent(name, path, args.rows, args.cols, engine=engine,
                              num_simulations=args.sims, temperature=1.0,
                              temperature_cutoff=args.opening_moves,
                              solver_max_undrawn=args.solver,
                              solver_node_budget=args.solver_budget,
                              merge_equivalent=bool(args.merge_eq),
                              evaluator_kwargs=evaluator_flags, **mcts_flags)


def main() -> None:
    parser = argparse.ArgumentParser(description="Boxes arena: agent A vs agent B")
    parser.add_argument("--a", required=True,
                        help="Registered agent name, or ckpt:<path> for an MCTS agent on it")
    parser.add_argument("--b", required=True, help="Registered agent name or ckpt:<path>")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--num_games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--sims", type=int, default=400, help="ckpt agents: simulations per move")
    parser.add_argument("--opening_moves", type=int, default=4,
                        help="ckpt agents: moves sampled at temperature 1 so games differ")
    parser.add_argument("--solver", type=int, default=0,
                        help="ckpt agents: exact endgame solver at <= N undrawn edges")
    parser.add_argument("--solver_budget", type=int, default=20_000)
    parser.add_argument("--merge_eq", type=int, default=0,
                        help="ckpt agents: equivalent-edge merge")
    parser.add_argument("--cpu", action="store_true")
    from alpha_go.boxes.nn_agent import add_search_flags
    add_search_flags(parser)
    args = parser.parse_args()
    for name in (args.a, args.b):
        if name.startswith("ckpt:"):
            register_checkpoint(name, args)
    result = play_match(
        args.a, args.b, args.rows, args.cols, args.num_games, args.seed, args.num_workers
    )
    summary = result.summary()
    print(
        f"{args.a} vs {args.b} on {summary['board']}: A wins {result.a_wins}, "
        f"B wins {result.b_wins}, draws {result.draws}; A win rate {summary['a_win_rate']} "
        f"{summary['a_win_rate_ci95']}; A mean margin {summary['a_mean_margin']} "
        f"± {summary['a_margin_std']}"
    )
    print("===RESULT===")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
