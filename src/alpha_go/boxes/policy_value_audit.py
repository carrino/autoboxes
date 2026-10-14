"""Policy mass and value ranking of frozen Boxes checkpoints on exactly solved roots.

Inference only. The roots are a frozen position file (replay prefixes) and every legal
child's exact final margin for the root mover comes from a complete child table (the
`child-diagnostic-{positions.jsonl,roots.csv,children.csv}` files of the probe analysis).
Nothing is solved or trained here.

Policy: the search's prior is `policy_dict(logits, legal, T)`, a softmax over the legal
edges at temperature T, so per root and temperature this reports its entropy, top-move
probability, the mass on the exact-optimal set (children whose exact margin equals the
root's) and, on winning roots, the mass on the win-preserving set (children with a positive
exact margin), next to the uniform-prior baselines (set size over legal moves).

Value: a child is scored the way the search sees it. `BoxesSearchState(child)` plays the
forced captures (its prefix) and the net scores the collapsed position for its side to
move; the value is carried to the root mover by complementing P(win) and negating the
margin only when the mover changed (a capture keeps the turn). A collapsed child that is
game over is exact. `raw` mode scores the child before the collapse, which is what a
search without the collapse would see; `search` mode is the one that speaks for the search.
Per root, predictor (P(win), expected margin, the shaped utility at lambda and k) and mode:
the exact regret of the predictor's favourite child, pairwise ordering accuracy over the
children with distinct exact margins, Spearman correlation and the within-root spread.

  uv run python -m alpha_go.boxes.policy_value_audit \\
      --tables experiments/2026-10-14_09-58-boxes-probe-analysis/data \\
      --checkpoint iter0=$CK/iter0.pt --checkpoint iter22=$CK/iter22.pt --output DIR
"""
# ruff: noqa: N806  (dimension-suffixed array names)
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import torch
from numpy.typing import NDArray

from alpha_go.boxes.encode import encode_batch
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import load_boxes_net, policy_dict
from alpha_go.boxes.utility import shaped_value

BANDS: tuple[tuple[int, int], ...] = ((35, 36), (37, 38))
PREDICTORS = ("win", "margin", "utility")
MODES = ("search", "raw")
NAN = float("nan")


@dataclass(frozen=True)
class Root:
    """A frozen root with the exact final margin, for the root mover, of every legal edge."""

    position: int
    prefix: tuple[int, ...]
    family: str
    rows: int
    cols: int
    undrawn: int
    root_final: int
    exact: dict[int, int]

    def board(self) -> Any:
        board = alpha_go_cpp.BoxesBoard(self.rows, self.cols)
        for edge in self.prefix:
            assert board.play_edge(edge), f"illegal replay edge {edge} at position {self.position}"
        return board

    @property
    def winning(self) -> bool:
        return self.root_final > 0

    @property
    def optimal(self) -> set[int]:
        return {e for e, v in self.exact.items() if v == self.root_final}

    @property
    def win_preserving(self) -> set[int]:
        return {e for e, v in self.exact.items() if v > 0}


def band_label(undrawn: int) -> str:
    """The reporting band of an undrawn count, or its own number outside the bands."""
    return next((f"{lo}..{hi}" for lo, hi in BANDS if lo <= undrawn <= hi), str(undrawn))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tables(tables: Path) -> list[Root]:
    """The frozen roots with complete exact child tables; an incomplete root is an error."""
    positions = {int(p["position"]): p for p in (
        json.loads(line) for line in (tables / "child-diagnostic-positions.jsonl")
        .read_text().splitlines() if line.strip())}
    children: dict[int, dict[int, int]] = {}
    with (tables / "child-diagnostic-children.csv").open() as handle:
        for row in csv.DictReader(handle):
            children.setdefault(int(row["position"]), {})[int(row["edge"])] = int(
                row["final_margin_for_root_mover"])
    roots = []
    with (tables / "child-diagnostic-roots.csv").open() as handle:
        for row in csv.DictReader(handle):
            index = int(row["position"])
            assert row["complete_known"] == "True", f"position {index} is not completely solved"
            p = positions[index]
            root = Root(index, tuple(p["prefix"]), str(p["family"]), int(p["rows"]),
                        int(p["cols"]), int(row["undrawn"]), int(row["root_final"]),
                        children[index])
            board = root.board()
            assert sorted(root.exact) == sorted(board.get_legal_moves_flat()), index
            assert max(root.exact.values()) == root.root_final, index
            assert board.num_edges() - board.move_count() == root.undrawn, index
            assert not alpha_go_cpp.BoxesSearchState(board).prefix(), (
                f"position {index} has a forced capture at the root")
            roots.append(root)
    assert roots, f"no roots under {tables}"
    return roots


@torch.no_grad()
def predict(model: BoxesNet, boards: list[Any]
            ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Policy logits (N, E), P(win) and expected final margin for each board's side to move."""
    policy_NE, margin_NM = model(torch.from_numpy(encode_batch(boards, model.features)))
    return (policy_NE.float().numpy(), model.win_prob(margin_NM).float().numpy(),
            model.expected_margin(margin_NM).float().numpy())


def to_root(win: float, margin: float, mover_same: bool) -> tuple[float, float]:
    """A side-to-move estimate carried to the root mover: flipped only when the mover changed."""
    return (win, margin) if mover_same else (1.0 - win, -margin)


@dataclass(frozen=True)
class ChildEval:
    """One legal edge of a root: how the search would score it, and the exact answer."""

    edge: int
    exact: int
    prefix_len: int  # forced captures the search plays before the net sees the child
    mover_same: bool  # after the collapse, is the root mover still to move?
    undrawn_after: int
    terminal: bool  # the collapse ends the game: the search uses the exact outcome
    decision: bool  # the collapse stops at a take-all / keep-control choice
    win: float  # root mover's P(win), search-matched (exact when terminal)
    margin: float  # root mover's expected final margin, search-matched
    raw_win: float  # the net on the child before the collapse, root mover's perspective
    raw_margin: float


def evaluate_children(model: BoxesNet, root: Root, board: Any) -> list[ChildEval]:
    """Every legal child scored in the search's and the raw perspective (two forwards)."""
    edges = sorted(root.exact)
    raw, states = [], []
    for edge in edges:
        child = board.copy()
        assert child.play_edge(edge)
        raw.append(child)
        states.append(alpha_go_cpp.BoxesSearchState(child))
    _, raw_win, raw_margin = predict(model, raw)
    _, col_win, col_margin = predict(model, states)
    mover = int(board.player())
    out = []
    for i, (edge, child, state) in enumerate(zip(edges, raw, states)):
        collapsed = state.board()
        terminal = bool(collapsed.is_game_over())
        exact = root.exact[edge]
        win, margin = to_root(float(col_win[i]), float(col_margin[i]),
                              int(state.player()) == mover)
        if terminal:
            win, margin = (1.0 if exact > 0 else 0.5 if exact == 0 else 0.0), float(exact)
        raw_w, raw_m = to_root(float(raw_win[i]), float(raw_margin[i]),
                               int(child.player()) == mover)
        out.append(ChildEval(edge, exact, len(state.prefix()), int(state.player()) == mover,
                             collapsed.num_edges() - collapsed.move_count(), terminal,
                             bool(state.has_decision()), win, margin, raw_w, raw_m))
    return out


def ranks_desc(scores: dict[int, float]) -> dict[int, int]:
    """1 = the highest score; ties broken by edge index, as argmax breaks them."""
    order = sorted(scores, key=lambda e: (-scores[e], e))
    return {e: i + 1 for i, e in enumerate(order)}


def policy_metrics(policy: dict[int, float], root: Root) -> dict[str, Any]:
    """Entropy, top move, the mass on the exact-optimal and win-preserving sets, and how
    deep in the prior's ordering the best-ranked optimal / win-preserving move sits."""
    edges = sorted(policy)
    probs = np.array([policy[e] for e in edges])
    top = edges[int(np.argmax(probs))]
    optimal, preserving, legal = root.optimal, root.win_preserving, len(edges)
    rank = ranks_desc(policy)
    return dict(
        legal=legal, optimal_count=len(optimal), win_preserving_count=len(preserving),
        entropy=float(-(probs * np.log(np.maximum(probs, 1e-300))).sum()),
        max_entropy=math.log(legal), top_edge=top, top_prob=float(policy[top]),
        top_regret=root.root_final - root.exact[top], top_optimal=float(top in optimal),
        top_win_preserving=float(root.exact[top] > 0) if root.winning else NAN,
        optimal_mass=float(sum(policy[e] for e in optimal)), uniform_optimal=len(optimal) / legal,
        win_preserving_mass=(float(sum(policy[e] for e in preserving)) if root.winning else NAN),
        uniform_win_preserving=len(preserving) / legal if root.winning else NAN,
        optimal_best_rank=min(rank[e] for e in optimal),
        win_preserving_best_rank=(min(rank[e] for e in preserving) if root.winning else NAN),
    )


def average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """1-based ranks with ties sharing their mean rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def ranking_metrics(scores: dict[int, float], root: Root) -> dict[str, Any]:
    """How well a per-child score orders the children against their exact margins."""
    edges = sorted(scores)
    s = np.array([scores[e] for e in edges], dtype=np.float64)
    x = np.array([root.exact[e] for e in edges], dtype=np.float64)
    best = edges[int(np.argmax(s))]
    ds, dx = s[:, None] - s[None, :], x[:, None] - x[None, :]
    distinct = np.triu(dx != 0, 1)
    agree = np.where(ds * dx > 0, 1.0, np.where(ds == 0, 0.5, 0.0))[distinct]
    rs, rx = average_ranks(s), average_ranks(x)
    spearman = (NAN if rs.std() == 0 or rx.std() == 0
                else float(np.corrcoef(rs, rx)[0, 1]))
    return dict(
        argmax_edge=best, argmax_regret=root.root_final - root.exact[best],
        argmax_optimal=float(best in root.optimal),
        argmax_win_preserving=float(root.exact[best] > 0) if root.winning else NAN,
        pairwise_accuracy=float(agree.mean()) if agree.size else NAN,
        distinct_pairs=int(distinct.sum()), spearman=spearman,
        spread_std=float(s.std()), spread_range=float(s.max() - s.min()),
        optimal_best_rank=min(ranks_desc(scores)[e] for e in root.optimal),
    )


def child_scores(evals: list[ChildEval], mode: str, predictor: str, lam: float, k: float
                 ) -> dict[int, float]:
    """The per-child score of one predictor in one mode, in the root mover's perspective."""
    pairs = {c.edge: ((c.win, c.margin) if mode == "search" else (c.raw_win, c.raw_margin))
             for c in evals}
    pick: dict[str, Callable[[float, float], float]] = {
        "win": lambda w, m: w, "margin": lambda w, m: m,
        "utility": lambda w, m: shaped_value(w, m, lam, k)}
    return {e: float(pick[predictor](w, m)) for e, (w, m) in pairs.items()}


def root_metrics(root: Root, evals: list[ChildEval], root_win: float, root_margin: float,
                 solver_undrawn: int) -> dict[str, Any]:
    """Root-level calibration and how the collapse changed what the search evaluates."""
    decided = [c for c in evals if c.exact != 0]
    return dict(
        root_final=root.root_final, root_win=root_win, root_margin=root_margin,
        root_sign_correct=float((root_win > 0.5) == root.winning),
        root_margin_error=abs(root_margin - root.root_final),
        children=len(evals), collapsed=sum(c.prefix_len > 0 for c in evals),
        mover_changes=sum(not c.mover_same for c in evals),
        terminal=sum(c.terminal for c in evals), decision=sum(c.decision for c in evals),
        solver_zone=sum(c.undrawn_after <= solver_undrawn and not c.terminal for c in evals),
        child_sign_accuracy=float(np.mean([(c.win > 0.5) == (c.exact > 0) for c in decided])),
        child_margin_mae=float(np.mean([abs(c.margin - c.exact) for c in evals])),
        raw_child_sign_accuracy=float(np.mean([(c.raw_win > 0.5) == (c.exact > 0)
                                               for c in decided])),
        raw_child_margin_mae=float(np.mean([abs(c.raw_margin - c.exact) for c in evals])),
    )


def audit(model: BoxesNet, name: str, roots: list[Root], temperatures: list[float],
          lam: float, k: float, solver_undrawn: int
          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
                     list[dict[str, Any]]]:
    """Per-position rows of one checkpoint: children, policy, ranking and root tables."""
    children_rows, policy_rows, ranking_rows, root_rows = [], [], [], []
    for root in roots:
        board = root.board()
        logits, win, margin = predict(model, [board])
        key = dict(position=root.position, undrawn=root.undrawn, band=band_label(root.undrawn),
                   family=root.family, checkpoint=name)
        evals = evaluate_children(model, root, board)
        prior = policy_dict(logits[0], board.get_legal_moves_flat(), 1.0)
        prior_rank = ranks_desc(prior)
        children_rows += [dict(key, **asdict(c), optimal=float(c.exact == root.root_final),
                               win_preserving=float(c.exact > 0) if root.winning else NAN,
                               prior=prior[c.edge], prior_rank=prior_rank[c.edge])
                          for c in evals]
        for temperature in temperatures:
            policy = policy_dict(logits[0], board.get_legal_moves_flat(), temperature)
            policy_rows.append(dict(key, temperature=temperature, root_winning=float(root.winning),
                                    **policy_metrics(policy, root)))
        for mode in MODES:
            for predictor in PREDICTORS:
                ranking_rows.append(dict(key, mode=mode, predictor=predictor,
                                         root_winning=float(root.winning),
                                         **ranking_metrics(child_scores(evals, mode, predictor,
                                                                        lam, k), root)))
        root_rows.append(dict(key, **root_metrics(root, evals, float(win[0]), float(margin[0]),
                                                  solver_undrawn)))
    return children_rows, policy_rows, ranking_rows, root_rows


def mean_of(rows: list[dict[str, Any]], metric: str) -> float:
    values = np.array([r[metric] for r in rows], dtype=np.float64)
    values = values[~np.isnan(values)]
    return float(values.mean()) if values.size else NAN


def group_means(rows: list[dict[str, Any]], keys: tuple[str, ...], metrics: tuple[str, ...]
                ) -> list[dict[str, Any]]:
    """Mean of each metric (NaN-skipping) per key combination, plus the band 'all'."""
    out = []
    for band in [*(f"{lo}..{hi}" for lo, hi in BANDS), "all"]:
        in_band = [r for r in rows if band == "all" or r["band"] == band]
        for combo in sorted({tuple(r[k] for k in keys) for r in in_band}):
            group = [r for r in in_band if tuple(r[k] for k in keys) == combo]
            out.append(dict(zip(keys, combo), band=band, positions=len(group),
                            **{m: mean_of(group, m) for m in metrics}))
    return out


def paired_bootstrap(diffs: NDArray[np.float64], families: list[str], resamples: int,
                     seed: int) -> dict[str, float]:
    """Mean paired difference with a percentile interval from resampling source families."""
    keep = ~np.isnan(diffs)
    names = sorted(set(families))
    index = np.array([names.index(f) for f in families], dtype=np.int64)
    sums = np.bincount(index[keep], weights=diffs[keep], minlength=len(names))
    counts = np.bincount(index[keep], minlength=len(names)).astype(np.float64)
    if counts.sum() == 0:
        return dict(mean=NAN, ci95_low=NAN, ci95_high=NAN, positions=0, families=0)
    draws = np.random.default_rng(seed).integers(0, len(names), size=(resamples, len(names)))
    means = sums[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1)
    return dict(mean=float(sums.sum() / counts.sum()),
                ci95_low=float(np.quantile(means, 0.025)),
                ci95_high=float(np.quantile(means, 0.975)),
                positions=int(counts.sum()), families=int((counts > 0).sum()))


def paired_differences(rows: list[dict[str, Any]], keys: tuple[str, ...],
                       metrics: tuple[str, ...], reference: str, candidate: str,
                       resamples: int, seed: int) -> list[dict[str, Any]]:
    """candidate minus reference per position, summarised per key combination and band."""
    out = []
    by = {(r["checkpoint"], r["position"], *(r[k] for k in keys)): r for r in rows}
    combos = sorted({tuple(r[k] for k in keys) for r in rows})
    for band in [*(f"{lo}..{hi}" for lo, hi in BANDS), "all"]:
        for combo in combos:
            refs = [r for r in rows if r["checkpoint"] == reference
                    and tuple(r[k] for k in keys) == combo
                    and (band == "all" or r["band"] == band)]
            cands = [by[(candidate, r["position"], *combo)] for r in refs]
            for metric in metrics:
                diffs = np.array([c[metric] - r[metric] for c, r in zip(cands, refs)],
                                 dtype=np.float64)
                result = paired_bootstrap(diffs, [r["family"] for r in refs], resamples, seed)
                if result["positions"]:
                    out.append(dict(zip(keys, combo), band=band, metric=metric, **result))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}" if isinstance(value, float) else str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def summary_markdown(policy: list[dict[str, Any]], ranking: list[dict[str, Any]],
                     roots: list[dict[str, Any]], diffs: list[dict[str, Any]],
                     names: list[str]) -> str:
    """The report: policy mass, value ranking, root calibration and paired differences."""
    parts = ["## Policy at the root (means over roots; win-preserving on winning roots)", "",
             markdown_table(policy, ["checkpoint", "temperature", "band", "positions", "entropy",
                                     "max_entropy", "top_prob", "optimal_mass",
                                     "uniform_optimal", "win_preserving_mass",
                                     "uniform_win_preserving", "top_regret", "top_optimal",
                                     "optimal_best_rank"]),
             "", "## Value ranking of the legal children (search = collapsed, raw = before "
             "the collapse)", "",
             markdown_table(ranking, ["checkpoint", "mode", "predictor", "band", "positions",
                                      "argmax_regret", "argmax_optimal", "argmax_win_preserving",
                                      "pairwise_accuracy", "spearman", "spread_std",
                                      "spread_range", "optimal_best_rank"]),
             "", "## Root calibration and the collapse", "",
             markdown_table(roots, ["checkpoint", "band", "positions", "root_sign_correct",
                                    "root_margin_error", "child_sign_accuracy",
                                    "child_margin_mae", "raw_child_sign_accuracy",
                                    "raw_child_margin_mae", "collapsed", "mover_changes",
                                    "terminal", "decision", "solver_zone"])]
    if len(names) > 1:
        parts += ["", f"## Paired differences, {names[-1]} minus {names[0]} "
                  "(source-family bootstrap, 95% percentile interval)", "",
                  markdown_table(diffs, ["temperature", "mode", "predictor", "band", "metric",
                                         "mean", "ci95_low", "ci95_high", "positions",
                                         "families"])]
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True,
                        help="directory with the child-diagnostic-* files")
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="NAME=PATH; the first is the reference of the paired differences")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--margin-utility-lambda", type=float, default=0.5)
    parser.add_argument("--margin-utility-k", type=float, default=6.0)
    parser.add_argument("--solver-undrawn", type=int, default=30,
                        help="the search's solver reach: collapsed children at or below it "
                             "would be settled exactly by the search, not by the net")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args()
    assert not args.output.exists(), f"output must be fresh: {args.output}"
    assert all(t > 0 for t in args.temperatures) and args.margin_utility_k > 0
    torch.set_num_threads(max(1, torch.get_num_threads()))
    checkpoints = dict(spec.split("=", 1) for spec in args.checkpoint)
    roots = load_tables(args.tables)
    print(f"{len(roots)} roots, {sum(len(r.exact) for r in roots)} exact children, "
          f"{sum(r.winning for r in roots)} winning roots, "
          f"{len({r.family for r in roots})} source families", flush=True)
    children, policy, ranking, root_rows = [], [], [], []
    for name, path in checkpoints.items():
        model = load_boxes_net(path, torch.device("cpu"))
        c, p, r, t = audit(model, name, roots, args.temperatures, args.margin_utility_lambda,
                           args.margin_utility_k, args.solver_undrawn)
        children += c
        policy += p
        ranking += r
        root_rows += t
        print(f"{name}: {path} ({model.features} planes, value target {model.value_target})",
              flush=True)
    names = list(checkpoints)
    policy_metric_names = ("entropy", "max_entropy", "top_prob", "optimal_mass", "uniform_optimal",
                           "win_preserving_mass", "uniform_win_preserving", "top_regret",
                           "top_optimal", "top_win_preserving", "optimal_best_rank",
                           "win_preserving_best_rank")
    ranking_metric_names = ("argmax_regret", "argmax_optimal", "argmax_win_preserving",
                            "pairwise_accuracy", "spearman", "spread_std", "spread_range",
                            "optimal_best_rank")
    root_metric_names = ("root_sign_correct", "root_margin_error", "child_sign_accuracy",
                         "child_margin_mae", "raw_child_sign_accuracy", "raw_child_margin_mae",
                         "collapsed", "mover_changes", "terminal", "decision", "solver_zone")
    policy_summary = group_means(policy, ("checkpoint", "temperature"), policy_metric_names)
    ranking_summary = group_means(ranking, ("checkpoint", "mode", "predictor"),
                                  ranking_metric_names)
    root_summary = group_means(root_rows, ("checkpoint",), root_metric_names)
    diffs: list[dict[str, Any]] = []
    if len(names) > 1:
        diffs = [dict(mode="", predictor="", **d) for d in paired_differences(
            policy, ("temperature",), ("entropy", "top_prob", "optimal_mass",
                                       "win_preserving_mass", "top_regret"),
            names[0], names[-1], args.resamples, args.seed)]
        diffs += [dict(temperature="", **d) for d in paired_differences(
            ranking, ("mode", "predictor"), ("argmax_regret", "pairwise_accuracy", "spearman"),
            names[0], names[-1], args.resamples, args.seed)]
    args.output.mkdir(parents=True)
    write_csv(args.output / "children.csv", children)
    write_csv(args.output / "policy.csv", policy)
    write_csv(args.output / "ranking.csv", ranking)
    write_csv(args.output / "roots.csv", root_rows)
    git = shutil.which("git")
    commit = (subprocess.run([git, "rev-parse", "HEAD"], capture_output=True, text=True)
              .stdout.strip() if git else "unknown")
    summary = dict(
        scope="Development diagnostic on training-derived positions; inference only",
        arguments={k: str(v) for k, v in vars(args).items()}, code_commit=commit,
        checkpoint_sha256={n: sha256(Path(p)) for n, p in checkpoints.items()},
        table_sha256={f: sha256(args.tables / f) for f in (
            "child-diagnostic-positions.jsonl", "child-diagnostic-roots.csv",
            "child-diagnostic-children.csv")},
        roots=len(roots), winning_roots=sum(r.winning for r in roots),
        families=len({r.family for r in roots}),
        policy=policy_summary, ranking=ranking_summary, root=root_summary,
        paired_differences=diffs,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = summary_markdown(policy_summary, ranking_summary, root_summary, diffs, names)
    (args.output / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
