"""Head-to-head arena for Boxes agents.

Plays `num_games` games between two registered agents through `alpha_go.gameplay.play_game`
and reports win rate (with a 95% Wilson interval) and mean final margin from agent A's point
of view. Games come in pairs: both games of a pair start from the same random opening of
`opening_moves` edges, with A moving first in one and second in the other, so the side-to-move
advantage of an opening cancels inside the pair. The pair score (share of pairs A wins on the
summed margin) is the strength signal; the per-colour win rates show how much colour decides.

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
    margins: list[int]  # final margin for A, one per game; A moves first in the even games
    a_seconds_per_move: float
    b_seconds_per_move: float

    @property
    def a_win_rate(self) -> float:
        return (self.a_wins + 0.5 * self.draws) / self.games

    @property
    def pair_margins(self) -> list[int]:
        """A's summed margin over each pair of games (same opening, colours swapped)."""
        return [self.margins[i] + self.margins[i + 1] for i in range(0, self.games - 1, 2)]

    @property
    def pair_score(self) -> float:
        """Share of pairs A wins on the summed margin, an equal sum counting half."""
        pairs = self.pair_margins
        return sum((m > 0) + 0.5 * (m == 0) for m in pairs) / len(pairs)

    def colour_win_rates(self) -> tuple[float, float]:
        """A's win rate moving first and moving second (draws count half)."""
        rates = []
        for parity in (0, 1):
            ms = self.margins[parity::2]
            rates.append(sum((m > 0) + 0.5 * (m == 0) for m in ms) / len(ms))
        return rates[0], rates[1]

    @staticmethod
    def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return centre - half, centre + half

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        """95% interval for A's win rate (draws count half)."""
        return self.wilson_interval(self.a_win_rate, self.games, z)

    def pair_wilson(self, z: float = 1.96) -> tuple[float, float]:
        """95% interval for A's pair score."""
        return self.wilson_interval(self.pair_score, len(self.pair_margins), z)

    def summary(self) -> dict[str, float | int | str]:
        lo, hi = self.wilson()
        plo, phi = self.pair_wilson()
        first, second = self.colour_win_rates()
        return {
            "a": self.a, "b": self.b, "board": f"{self.rows}x{self.cols}", "games": self.games,
            "a_wins": self.a_wins, "b_wins": self.b_wins, "draws": self.draws,
            "a_win_rate": round(self.a_win_rate, 4),
            "a_win_rate_ci95": f"[{lo:.3f}, {hi:.3f}]",
            "a_first_win_rate": round(first, 4), "a_second_win_rate": round(second, 4),
            "a_pair_score": round(self.pair_score, 4),
            "a_pair_score_ci95": f"[{plo:.3f}, {phi:.3f}]",
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


def random_opening(
    rows: int, cols: int, num_moves: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """`num_moves` uniformly random legal edges from the empty board, as lattice moves."""
    board = get_game("boxes").new_board(rows, 0.0, cols)
    moves: list[tuple[int, int]] = []
    for _ in range(num_moves):
        row, col = board.row_col(int(rng.choice(board.get_legal_moves_flat())))
        moves.append((int(row), int(col)))
        board.play(*moves[-1])
    return moves


def _one_game(
    a: str, b: str, rows: int, cols: int, index: int, seed: int,
    start_moves: list[tuple[int, int]] | None = None,
) -> tuple[int, float, int, float, int]:
    """Play game `index` from `start_moves`; A moves first on even indices.

    Returns (margin for A, A seconds, A moves, B seconds, B moves).
    """
    agent_a, agent_b = _agents(a, b)
    a_first = index % 2 == 0
    first, second = (agent_a, agent_b) if a_first else (agent_b, agent_a)
    record = play_game(
        first, second, board_size=rows, board_cols=cols, seed=seed + index,
        max_moves=get_game("boxes").default_max_moves(rows, cols), collect_boards=False,
        game=get_game("boxes"), render_debug_on_error=False, start_moves=start_moves,
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
    seed: int = 0, num_workers: int = 1, opening_moves: int = 0,
) -> MatchResult:
    """Play `num_games` games (an even number) between registered agents `a` and `b`.

    Games 2p and 2p+1 start from the same random opening of `opening_moves` edges, drawn
    from `seed`, with A moving first in the even game and second in the odd one.
    """
    assert num_games % 2 == 0, "games come in colour-swapped pairs"
    cols = cols or rows
    rng = np.random.default_rng(seed)
    openings = [random_opening(rows, cols, opening_moves, rng) for _ in range(num_games // 2)]
    start = time.time()
    every = max(1, num_games // 10)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = []
        for r in pool.map(lambda i: _one_game(a, b, rows, cols, i, seed, openings[i // 2]),
                          range(num_games)):
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
    # Greedy from the first searched move: the random opening of the pair gives the variety.
    register_boxes_mcts_agent(name, path, args.rows, args.cols, engine=engine,
                              num_simulations=args.sims, temperature=1.0,
                              temperature_cutoff=0,
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
                        help="random edges both games of a pair start from, so games differ")
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
        args.a, args.b, args.rows, args.cols, args.num_games, args.seed, args.num_workers,
        opening_moves=args.opening_moves,
    )
    summary = result.summary()
    print(
        f"{args.a} vs {args.b} on {summary['board']}: A wins {result.a_wins}, "
        f"B wins {result.b_wins}, draws {result.draws}; A win rate {summary['a_win_rate']} "
        f"{summary['a_win_rate_ci95']} (first {summary['a_first_win_rate']}, "
        f"second {summary['a_second_win_rate']}); pair score {summary['a_pair_score']} "
        f"{summary['a_pair_score_ci95']}; A mean margin {summary['a_mean_margin']} "
        f"± {summary['a_margin_std']}"
    )
    print("===RESULT===")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
