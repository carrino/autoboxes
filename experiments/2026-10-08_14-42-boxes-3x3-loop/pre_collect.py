"""Bootstrap games without search for iter0 training.

Writes greedy-vs-random and random-vs-random 3x3 games under
$GAME_DATA_DIR/experiments/<EXP>/bootstrap-it0/. No MCTS data, so the trainer uses the
played move (label-smoothed) as the policy target and the final margin as the value target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_NAME = Path(__file__).resolve().parent.name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--num_games", type=int, default=400, help="per matchup")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from alpha_go.self_play import main as self_play_main
    save_name = f"experiments/{EXP_NAME}/bootstrap-it0"
    for offset, (black, white) in enumerate([("boxes-greedy", "boxes-random"),
                                             ("boxes-random", "boxes-greedy"),
                                             ("boxes-random", "boxes-random")]):
        sys.argv = [
            "self_play", "--game", "boxes", "--board_size", str(args.rows),
            *(["--board-cols", str(args.cols)] if args.cols else []),
            "--black", black, "--white", white, "--num_games", str(args.num_games),
            "--num_workers", str(args.num_workers), "--save-name", save_name,
            "--seed", str(args.seed), "--game_index_offset", str(offset * args.num_games),
        ]
        print(f"=== bootstrap {black} vs {white}: {args.num_games} games -> {save_name} ===", flush=True)
        self_play_main()


if __name__ == "__main__":
    main()
