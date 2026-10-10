"""CPU probe of `--prove_terminals`: 3x3, uniform evaluator (no net), solver at 8, 300 random
positions at 11..14 undrawn edges (exact leaves 3..6 plies down), scored against the exact
solver. `uv run experiments/2026-10-08_14-42-boxes-3x3-loop/proof_probe.py` (about 5 s)."""
import random
import time

import alpha_go_cpp

from alpha_go.boxes.nn_agent import optimal_edges
from alpha_go.boxes.rules import geometry

solver = alpha_go_cpp.BoxesSolver(3, 3, 1 << 20)
geo = geometry(3, 3)
rng = random.Random(0)
positions = []
while len(positions) < 300:
    board = alpha_go_cpp.BoxesBoard(3, 3)
    target = rng.randint(11, 14)
    while board.num_edges() - board.move_count() > target:
        board.play_edge(rng.choice(board.get_legal_moves_flat()))
    state = alpha_go_cpp.BoxesSearchState(board, solver, 8, 1 << 20, False)
    if state.solved() or state.prefix() or state.has_decision():
        continue
    # Margin-optimal edges (the oracle's "optimal") and outcome-keeping edges (same exact
    # win / loss as the best move): the search maximises the latter.
    best = solver.final_margin(board)
    keeping = set()
    for e in board.get_legal_moves_flat():
        child = board.copy()
        child.play_edge(e)
        child_final = solver.final_margin(child)
        mine = child_final if child.to_play() == board.to_play() else -child_final
        if (mine > 0) == (best > 0):
            keeping.add(e)
    positions.append((board, state, set(optimal_edges(solver, board, geo)), keeping))


def uniform_batched(states):
    out = []
    for s in states:
        ms = s.get_legal_moves_flat()
        out.append(({m: 1.0 / len(ms) for m in ms}, 0.5))
    return out


n = len(positions)
random_optimal = sum(len(o) / len(b.get_legal_moves_flat()) for b, _, o, _ in positions) / n
random_keeps = sum(len(k) / len(b.get_legal_moves_flat()) for b, _, _, k in positions) / n
print(f"{n} positions; random legal move optimal: {random_optimal:.3f}; "
      f"random move keeps the outcome: {random_keeps:.3f}")
for sims in (100, 400):
    for prove in (False, True):
        cfg = alpha_go_cpp.MCTSConfig()
        cfg.c_puct = 1.5
        cfg.prove_terminals = prove
        hits = keeps = proven = 0
        t0 = time.time()
        for board, state, optimal, keeping in positions:
            tree = alpha_go_cpp.BoxesSearchMCTSTree(state, cfg)
            tree.run_simulations_batched(sims, 16, uniform_batched)
            move = tree.select_action(0.0)
            hits += move in optimal
            keeps += move in keeping
            proven += tree.is_root_proven()
        print(f"sims {sims:4d} prove {int(prove)}: margin-optimal {hits / n:.3f}  keeps outcome "
              f"{keeps / n:.3f}  roots proven {proven / n:.2f}  ({time.time() - t0:.0f}s)")
