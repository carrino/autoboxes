"""Arena promotion for one iteration.

Plays the iter N checkpoint against the current champion (alternating first player),
promotes it on a win rate >= --threshold, and also scores it against the fixed baselines
boxes-greedy and boxes-ab-d4 so progress is visible on an absolute scale. State lives in
league_state.json next to this file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpha_go.boxes.arena import play_match
from alpha_go.boxes.nn_agent import register_boxes_mcts_agent

EXP_DIR = Path(__file__).resolve().parent
STATE = EXP_DIR / "league_state.json"


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"champion": None, "history": []}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--num_games", type=int, default=100)
    p.add_argument("--baseline_games", type=int, default=40)
    p.add_argument("--num_simulations", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = "cpu" if args.cpu else None
    mcts = dict(num_simulations=args.num_simulations, c_puct=1.5, temperature=0.0,
                leaf_batch_size=8)
    candidate_ckpt = EXP_DIR / "checkpoints" / f"iter{args.iteration}.pt"
    candidate = register_boxes_mcts_agent(f"cand-it{args.iteration}", candidate_ckpt, args.rows,
                                          args.cols, device=device, **mcts)
    state = load_state()
    entry: dict = {"iteration": args.iteration}
    if state["champion"] is None:
        promoted, vs_champion = True, None
    else:
        champion = register_boxes_mcts_agent(f"champ-it{state['champion']}",
                                             EXP_DIR / "checkpoints" / f"iter{state['champion']}.pt",
                                             args.rows, args.cols, device=device, **mcts)
        match = play_match(candidate, champion, args.rows, args.cols, args.num_games, seed=1000 + args.iteration,
                           num_workers=args.num_workers)
        vs_champion = match.summary()
        promoted = match.a_win_rate >= args.threshold
        print(f"iter{args.iteration} vs champion iter{state['champion']}: {vs_champion}")
    for baseline in ("boxes-greedy", "boxes-ab-d4"):
        result = play_match(candidate, baseline, args.rows, args.cols, args.baseline_games,
                            seed=2000 + args.iteration, num_workers=args.num_workers)
        entry[f"vs_{baseline}"] = result.summary()
        print(f"iter{args.iteration} vs {baseline}: {entry[f'vs_{baseline}']}")
    entry.update({"vs_champion": vs_champion, "promoted": promoted,
                  "champion_before": state["champion"]})
    if promoted:
        state["champion"] = args.iteration
    entry["champion_after"] = state["champion"]
    state["history"].append(entry)
    STATE.write_text(json.dumps(state, indent=2))
    print(f"league: champion=iter{state['champion']} promoted={promoted} -> {STATE.name}")
    print("===RESULT===")
    print(json.dumps(entry))


if __name__ == "__main__":
    main()
