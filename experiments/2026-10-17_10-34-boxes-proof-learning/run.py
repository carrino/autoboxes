"""Matched, single-thread 2x3 policy-target mechanism experiment.

All positions and searched decisions pass through shared play_game. Truncated
game outcomes are never used; the independent exhaustive oracle supplies value
targets and evaluation labels. Heldout canonical masks cannot occur in training.
"""
# ruff: noqa: N803, N806
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import alpha_go_cpp as cpp
import numpy as np
import torch

from alpha_go.boxes.agents import BoxesRandomAgent
from alpha_go.boxes.encode import encode
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import BoxesMCTSAgent
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.symmetry import apply, edge_permutation, transforms
from alpha_go.gameplay import play_game
from alpha_go.games import get_game

ROWS, COLS = 2, 3
GEO = geometry(ROWS, COLS)
SYMS = transforms(ROWS, COLS)
PERMS = [edge_permutation(GEO, k) for k in SYMS]


class UniformEvaluator:
    """Equivalent to an initial BoxesNet's zero policy and value readouts."""

    checkpoint_path = None
    margin_utility_lambda = 0.5
    margin_utility_k = 6.0

    def evaluate(self, board):
        edges = board.get_legal_moves_flat()
        return {e: 1 / len(edges) for e in edges}, 0.5

    def batch_evaluate(self, boards):
        return [self.evaluate(b) for b in boards]

    def close(self):
        pass


def canonical(mask):
    return min(sum(1 << int(p[e]) for e in range(GEO.num_edges) if mask >> e & 1)
               for p in PERMS)


def collect(args, folder):
    oracle = Oracle(ROWS, COLS)
    rows, seen = [], set()
    random_agent = BoxesRandomAgent()
    searched = BoxesMCTSAgent(UniformEvaluator(), num_simulations=args.simulations,
                             leaf_batch_size=8, c_puct=1.5, temperature=0,
                             forced_collapse=True, prove_terminals=True,
                             shape_terminals=True, solver_max_undrawn=args.solver,
                             solver_node_budget=50000, solver_table_entries=1 << 14)
    start = time.perf_counter()
    attempt = 0
    while len(rows) < args.train + args.heldout:
        # Separate feasibility pilot seeds from the frozen main collection.
        seed = args.data_seed + attempt
        attempt += 1
        opening = play_game(random_agent, random_agent, board_size=ROWS, board_cols=COLS,
                            max_moves=8, seed=seed, game=get_game("boxes"))
        board, py = cpp.BoxesBoard(ROWS, COLS), BoxesBoard(ROWS, COLS)
        prefix = [int(GEO.lattice_edge[rc]) for rc in opening.moves]
        for edge in prefix:
            assert board.play_edge(edge) and py.play_edge(edge)
        key = canonical(py.edges)
        if key in seen or cpp.BoxesSearchState(board).prefix():
            continue
        seen.add(key)
        record = play_game(searched, searched, board_size=ROWS, board_cols=COLS,
                           max_moves=1, seed=seed, start_moves=opening.moves,
                           game=get_game("boxes"), collect_metrics=True)
        metric = record.move_metrics[0]
        raw = np.asarray(metric.raw_visit_counts, dtype=np.int64)
        filtered = np.asarray(metric.visit_counts, dtype=np.int64)
        assert raw.sum() > 0 and filtered.sum() > 0
        legal = py.get_legal_moves_flat()
        children = {e: oracle.child_value(py.edges, e) for e in legal}
        best = oracle.value(py.edges)
        optimum = [e for e, value in children.items() if value == best]
        # Under shaped terminal proof semantics, retained targets should be exact
        # only when the full root is proved; do not assume all searches finish.
        proven = searched.last_search_result.tree.is_root_proven()
        if proven:
            assert all(e in optimum for e in np.flatnonzero(filtered))
        rows.append(dict(id=len(rows), family=key, prefix=prefix, seed=seed,
                         raw=raw.tolist(), filtered=filtered.tolist(),
                         optimum=optimum, children=children,
                         final_margin=oracle.final_margin(py), proven=proven,
                         selected=int(GEO.lattice_edge[record.moves[0]])))
        if len(rows) % 64 == 0:
            print(f"collected {len(rows)} roots in {time.perf_counter()-start:.1f}s", flush=True)
    # Assignment is fixed from the family hash, before any learning/readout.
    rows.sort(key=lambda r: hashlib.sha256(f"split-v1:{r['family']}".encode()).digest())
    for i, row in enumerate(rows):
        row["split"] = "train" if i < args.train else "heldout"
    (folder / "roots.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def tensors(rows, augment=False):
    planes, raw, filtered, margins, legal, optimum, regrets = [], [], [], [], [], [], []
    for row in rows:
        board = BoxesBoard(ROWS, COLS)
        for edge in row["prefix"]:
            assert board.play_edge(edge)
        x = encode(board, "chains")
        mask = np.zeros(GEO.num_edges, dtype=bool)
        mask[board.get_legal_moves_flat()] = True
        optimal = np.zeros(GEO.num_edges, dtype=bool)
        optimal[row["optimum"]] = True
        regret = np.zeros(GEO.num_edges, dtype=np.float32)
        best = max(row["children"].values())
        for edge, value in row["children"].items():
            regret[int(edge)] = best - value
        for k, p in zip(SYMS if augment else [0], PERMS if augment else [PERMS[0]]):
            planes.append(apply(x, k).copy())
            inv = np.argsort(p)
            for dest, counts in ((raw, row["raw"]), (filtered, row["filtered"])):
                v = np.asarray(counts, dtype=np.float32)
                dest.append((v / v.sum())[inv])
            margins.append(row["final_margin"])
            legal.append(mask[inv])
            optimum.append(optimal[inv])
            regrets.append(regret[inv])
    return {k: torch.from_numpy(np.asarray(v)) for k, v in dict(
        planes=planes, raw=raw, filtered=filtered, margins=margins,
        legal=legal, optimum=optimum, regrets=regrets).items()}


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits, value = model(data["planes"])
    probs = logits.masked_fill(~data["legal"], -torch.inf).softmax(-1)
    chosen = probs.argmax(-1)
    indices = torch.arange(len(chosen))
    per_root = dict(optimal_mass=(probs * data["optimum"]).sum(-1),
                    optimal_top=data["optimum"][indices, chosen].float(),
                    top_regret=data["regrets"][indices, chosen],
                    policy_expected_regret=(probs * data["regrets"]).sum(-1),
                    margin_abs_error=(model.expected_margin(value) - data["margins"]).abs())
    return ({k: float(v.mean()) for k, v in per_root.items()},
            {k: v.tolist() for k, v in per_root.items()})


def train(args, rows, folder):
    train_rows = [r for r in rows if r["split"] == "train"]
    heldout_rows = [r for r in rows if r["split"] == "heldout"]
    assert not {r["family"] for r in train_rows} & {r["family"] for r in heldout_rows}
    dataset = tensors(train_rows, augment=True)
    reads = dict(train=tensors(train_rows), heldout=tensors(heldout_rows))
    output = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        original = BoxesNet(ROWS, COLS, channels=24, n_blocks=2, value_hidden=32,
                            norm_type="gn", features="chains")
        batches = torch.randint(len(dataset["planes"]), (args.steps, args.batch),
                                generator=torch.Generator().manual_seed(seed + 1000))
        for arm in ("raw", "filtered"):
            model = copy.deepcopy(original)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            start = time.perf_counter()
            for step, indices in enumerate(batches):
                model.train()
                loss, _, _ = model.compute_loss(dataset["planes"][indices],
                                                dataset[arm][indices],
                                                dataset["margins"][indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if (step + 1) % 200 == 0:
                    print(f"seed {seed} {arm}: {step+1}/{args.steps} steps; "
                          f"{time.perf_counter()-start:.1f}s", flush=True)
            for split, data in reads.items():
                metrics, per_root = evaluate(model, data)
                result = dict(seed=seed, arm=arm, split=split, **metrics)
                output.append(result)
                (folder / f"per-root-{seed}-{arm}-{split}.json").write_text(
                    json.dumps(per_root, indent=2) + "\n")
                print(json.dumps(result), flush=True)
            (folder / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--heldout", type=int, default=128)
    parser.add_argument("--simulations", type=int, default=160)
    parser.add_argument("--solver", type=int, default=8)
    parser.add_argument("--data-seed", type=int, default=20261031)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {**vars(args), "output": str(args.output), "rows": ROWS, "cols": COLS,
                     "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                              text=True).strip(),
                     "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "extension_sha256": hashlib.sha256(
                         Path(cpp.__file__).read_bytes()).hexdigest()}
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    roots = args.output / "roots.jsonl"
    rows = ([json.loads(s) for s in roots.read_text().splitlines()] if roots.exists()
            else collect(args, args.output))
    assert len(rows) == args.train + args.heldout
    diagnostic = {}
    for split in ("train", "heldout"):
        group = [r for r in rows if r["split"] == split]
        removed = [sum(a for a, b in zip(r["raw"], r["filtered"]) if b == 0)
                   / sum(r["raw"]) for r in group]
        diagnostic[split] = dict(positions=len(group), proven=sum(r["proven"] for r in group),
                                 changed=sum(x > 0 for x in removed),
                                 mean_removed_mass=float(np.mean(removed)))
    (args.output / "target-diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    print(json.dumps(diagnostic), flush=True)
    if not args.collect_only:
        train(args, rows, args.output)


if __name__ == "__main__":
    main()
