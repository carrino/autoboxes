"""Arena promotion for one iteration.

Plays the iter N checkpoint against the current champion (alternating first player, the
first --opening_moves moves sampled at temperature 1 so the games differ),
promotes it on a win rate >= --threshold, and also scores it against the fixed baselines
boxes-greedy and boxes-ab-d4 so progress is visible on an absolute scale. Checkpoints are
read from checkpoints/<tag>/ and state lives in league_state-<tag>.json next to this file
(tag = <rows>x<cols>), so runs on different boards never collide.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpha_go.boxes.arena import play_match
from alpha_go.boxes.inference import PlaneBatchedEngine
from alpha_go.boxes.nn_agent import load_boxes_net, pick_device, register_boxes_mcts_agent

EXP_DIR = Path(__file__).resolve().parent


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {"champion": None, "history": []}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--tag", default=None,
                   help="checkpoint subdir and state-file suffix; default <rows>x<cols>")
    p.add_argument("--num_games", type=int, default=100)
    p.add_argument("--baseline_games", type=int, default=40)
    p.add_argument("--num_simulations", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--opening_moves", type=int, default=4,
                   help="moves sampled at temperature 1 before greedy play, for game variety")
    p.add_argument("--solver_max_undrawn", type=int, default=0,
                   help="exact endgame solver at <= N undrawn edges for the candidate and champion")
    p.add_argument("--solver_node_budget", type=int, default=20_000)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = pick_device("cpu" if args.cpu else None)
    assert args.cpu or device.type == "cuda", "CUDA not available; pass --cpu to run on CPU"
    engines: list[PlaneBatchedEngine] = []

    def shared_engine(checkpoint: Path) -> PlaneBatchedEngine:
        # One GPU engine per net, shared by every game thread (as in self-play), so leaves
        # from all threads batch into the same forwards.
        engine = PlaneBatchedEngine(load_boxes_net(checkpoint, device), device,
                                    batch_size=64, batch_timeout_ms=1.0)
        engine.start()
        engines.append(engine)
        return engine
    # Sample the first few moves from the visit distribution: at temperature 0 both nets are
    # deterministic and a 100-game match is the same two games played 50 times each.
    mcts = dict(num_simulations=args.num_simulations, c_puct=1.5, temperature=1.0,
                temperature_cutoff=args.opening_moves, leaf_batch_size=16,
                solver_max_undrawn=args.solver_max_undrawn,
                solver_node_budget=args.solver_node_budget)
    tag = args.tag or f"{args.rows}x{args.cols or args.rows}"
    ckpt_dir = EXP_DIR / "checkpoints" / tag
    state_file = EXP_DIR / f"league_state-{tag}.json"
    candidate_ckpt = ckpt_dir / f"iter{args.iteration}.pt"
    candidate = register_boxes_mcts_agent(f"cand-it{args.iteration}", candidate_ckpt, args.rows,
                                          args.cols, engine=shared_engine(candidate_ckpt), **mcts)
    state = load_state(state_file)
    entry: dict = {"iteration": args.iteration}
    if state["champion"] is None:
        promoted, vs_champion = True, None
    else:
        champion_ckpt = ckpt_dir / f"iter{state['champion']}.pt"
        champion = register_boxes_mcts_agent(f"champ-it{state['champion']}", champion_ckpt,
                                             args.rows, args.cols,
                                             engine=shared_engine(champion_ckpt), **mcts)
        match = play_match(candidate, champion, args.rows, args.cols, args.num_games,
                           seed=1000 + args.iteration, num_workers=args.num_workers)
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
    state_file.write_text(json.dumps(state, indent=2))
    print(f"league: champion=iter{state['champion']} promoted={promoted} -> {state_file.name}")
    print("===RESULT===")
    print(json.dumps(entry))
    for engine in engines:
        engine.stop()


if __name__ == "__main__":
    main()
