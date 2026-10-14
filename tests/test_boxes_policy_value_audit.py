"""The policy/value audit scores children the way the search does (forced-capture collapse,
root-mover perspective) and its metrics carry the right baselines: checked on 2x3 against the
exhaustive oracle with untrained nets."""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import pytest
import torch

from alpha_go.boxes import policy_value_audit as pva
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import save_boxes_net
from alpha_go.boxes.oracle import Oracle

ROWS, COLS = 2, 3


def tiny_net(seed: int) -> BoxesNet:
    torch.manual_seed(seed)
    return BoxesNet(ROWS, COLS, channels=8, n_blocks=1, value_hidden=8).eval()


def random_root(rng: random.Random, undrawn: int) -> Any | None:
    """A random 2x3 position with `undrawn` edges left and no forced capture at the root."""
    board = alpha_go_cpp.BoxesBoard(ROWS, COLS)
    while board.num_edges() - board.move_count() > undrawn:
        board.play_edge(rng.choice(board.get_legal_moves_flat()))
    ok = not board.is_game_over() and not alpha_go_cpp.BoxesSearchState(board).prefix()
    return board if ok else None


def exact_table(oracle: Oracle, board: Any) -> tuple[int, dict[int, int]]:
    """Root value and every child's exact final margin for the root mover, from the oracle."""
    mask, margin = int(board.edges()), int(board.margin())
    return margin + oracle.value(mask), {
        e: margin + oracle.child_value(mask, e) for e in board.get_legal_moves_flat()}


def roots_with_prefix(oracle: Oracle, rng: random.Random, count: int, undrawn: int
                      ) -> list[tuple[pva.Root, Any]]:
    """Roots built by replaying a stored prefix, as the frozen position files do."""
    out: list[tuple[pva.Root, Any]] = []
    while len(out) < count:
        prefix: list[int] = []
        board = alpha_go_cpp.BoxesBoard(ROWS, COLS)
        while board.num_edges() - board.move_count() > undrawn:
            edge = rng.choice(board.get_legal_moves_flat())
            prefix.append(edge)
            board.play_edge(edge)
        if board.is_game_over() or alpha_go_cpp.BoxesSearchState(board).prefix():
            continue
        root_final, exact = exact_table(oracle, board)
        out.append((pva.Root(len(out), tuple(prefix), f"family{len(out) % 3}", ROWS, COLS,
                             undrawn, root_final, exact), board))
    return out


class TestPerspectiveAndCollapse:
    def test_children_match_search_state_and_flip_only_on_mover_change(self) -> None:
        oracle, rng, model = Oracle(ROWS, COLS), random.Random(0), tiny_net(0)
        seen = {"normal": 0, "retained_capture": 0, "collapsed": 0, "terminal": 0}
        for undrawn in (9, 7, 5, 3):
            for root, board in roots_with_prefix(oracle, rng, 12, undrawn):
                evals = pva.evaluate_children(model, root, board)
                assert [c.edge for c in evals] == sorted(root.exact)
                for c in evals:
                    child = board.copy()
                    assert child.play_edge(c.edge)
                    gained = (int(child.margin()) if child.player() == board.player()
                              else -int(child.margin())) - int(board.margin())
                    state = alpha_go_cpp.BoxesSearchState(child)
                    same = int(state.player()) == int(board.player())
                    assert c.prefix_len == len(state.prefix()) and c.mover_same == same
                    seen["collapsed"] += c.prefix_len > 0
                    seen["normal"] += gained == 0 and child.player() != board.player()
                    seen["retained_capture"] += gained == 1
                    _, win, margin = pva.predict(model, [state])
                    expected = pva.to_root(float(win[0]), float(margin[0]), same)
                    if c.terminal:
                        seen["terminal"] += 1
                        assert c.win == (1.0 if c.exact > 0 else 0.5 if c.exact == 0 else 0.0)
                        assert c.margin == c.exact
                    else:
                        assert (c.win, c.margin) == pytest.approx(expected)
                    _, raw_win, raw_margin = pva.predict(model, [child])
                    assert (c.raw_win, c.raw_margin) == pytest.approx(pva.to_root(
                        float(raw_win[0]), float(raw_margin[0]),
                        int(child.player()) == int(board.player())))
                    assert c.exact == root.exact[c.edge]
        assert all(seen.values()), seen

    def test_double_capture_keeps_the_mover(self) -> None:
        """2x2: boxes (0,0) and (0,1) three-sided around their shared edge 7; drawing it
        captures both and keeps the turn, so the child's value is not flipped."""
        oracle, model = Oracle(2, 2), BoxesNet(2, 2, channels=8, n_blocks=1, value_hidden=8).eval()
        board = alpha_go_cpp.BoxesBoard(2, 2)
        for edge in (0, 2, 6, 1, 3, 8):
            assert board.play_edge(edge)
        mask, margin = int(board.edges()), int(board.margin())
        exact = {e: margin + oracle.child_value(mask, e) for e in board.get_legal_moves_flat()}
        root = pva.Root(0, (0, 2, 6, 1, 3, 8), "f", 2, 2, 6, margin + oracle.value(mask), exact)
        child = board.copy()
        assert child.play_edge(7) and child.player() == board.player()
        assert int(child.margin()) - margin == 2
        c = next(c for c in pva.evaluate_children(model, root, board) if c.edge == 7)
        state = alpha_go_cpp.BoxesSearchState(child)
        _, win, m = pva.predict(model, [state])
        same = int(state.player()) == int(board.player())
        assert c.mover_same == same and c.prefix_len == len(state.prefix())
        assert (c.win, c.margin) == pytest.approx(pva.to_root(float(win[0]), float(m[0]), same))
        _, raw_win, raw_m = pva.predict(model, [child])
        assert (c.raw_win, c.raw_margin) == pytest.approx((float(raw_win[0]), float(raw_m[0])))

    def test_to_root_complements_and_negates(self) -> None:
        assert pva.to_root(0.8, 3.0, True) == (0.8, 3.0)
        assert pva.to_root(0.8, 3.0, False) == pytest.approx((0.2, -3.0))


class TestMetrics:
    def test_policy_metrics_baselines(self) -> None:
        oracle, rng = Oracle(ROWS, COLS), random.Random(1)
        (root, board), = roots_with_prefix(oracle, rng, 1, 8)
        legal = sorted(root.exact)
        uniform = {e: 1 / len(legal) for e in legal}
        m = pva.policy_metrics(uniform, root)
        assert m["optimal_mass"] == pytest.approx(m["uniform_optimal"])
        assert m["entropy"] == pytest.approx(math.log(len(legal))) == m["max_entropy"]
        if root.winning:
            assert m["win_preserving_mass"] == pytest.approx(m["uniform_win_preserving"])
        else:
            assert math.isnan(m["win_preserving_mass"])
        best = min(root.optimal)
        one_hot = {e: float(e == best) for e in legal}
        m = pva.policy_metrics(one_hot, root)
        assert m["optimal_mass"] == 1.0 and m["top_regret"] == 0 and m["top_optimal"] == 1.0
        assert m["optimal_best_rank"] == 1
        assert pva.ranks_desc({3: 0.2, 5: 0.5, 7: 0.2}) == {5: 1, 3: 2, 7: 3}
        logits = np.random.default_rng(0).normal(size=max(legal) + 1).astype(np.float32)
        from alpha_go.boxes.nn_agent import policy_dict
        t1 = pva.policy_metrics(policy_dict(logits, legal, 1.0), root)
        t2 = pva.policy_metrics(policy_dict(logits, legal, 2.0), root)
        assert t2["entropy"] >= t1["entropy"] and t2["top_edge"] == t1["top_edge"]
        assert t1["top_regret"] >= 0 and 0 <= t1["optimal_mass"] <= 1

    def test_ranking_metrics_on_exact_reversed_and_constant_scores(self) -> None:
        oracle, rng = Oracle(ROWS, COLS), random.Random(2)
        (root, _), = roots_with_prefix(oracle, rng, 1, 9)
        exact = {e: float(v) for e, v in root.exact.items()}
        assert len(set(exact.values())) > 1
        m = pva.ranking_metrics(exact, root)
        assert m["pairwise_accuracy"] == 1.0 and m["spearman"] == pytest.approx(1.0)
        assert m["argmax_regret"] == 0 and m["argmax_optimal"] == 1.0
        assert m["optimal_best_rank"] == 1
        m = pva.ranking_metrics({e: -v for e, v in exact.items()}, root)
        assert m["pairwise_accuracy"] == 0.0 and m["spearman"] == pytest.approx(-1.0)
        m = pva.ranking_metrics({e: 0.5 for e in exact}, root)
        assert m["pairwise_accuracy"] == 0.5 and math.isnan(m["spearman"])
        assert m["spread_std"] == 0.0 and m["spread_range"] == 0.0

    def test_average_ranks_and_paired_bootstrap(self) -> None:
        assert pva.average_ranks(np.array([3.0, 1.0, 3.0])).tolist() == [2.5, 1.0, 2.5]
        diffs = np.array([1.0, 1.0, float("nan"), 1.0])
        b = pva.paired_bootstrap(diffs, ["a", "b", "b", "c"], 200, 0)
        assert b["mean"] == 1.0 and b["ci95_low"] == 1.0 and b["ci95_high"] == 1.0
        assert b["positions"] == 3 and b["families"] == 3
        empty = pva.paired_bootstrap(np.array([float("nan")]), ["a"], 10, 0)
        assert math.isnan(empty["mean"]) and empty["positions"] == 0


def write_tables(directory: Path, roots: list[tuple[pva.Root, Any]]) -> None:
    """The three child-diagnostic files in the probe analysis' layout."""
    directory.mkdir(parents=True)
    with (directory / "child-diagnostic-positions.jsonl").open("w") as f:
        for root, _ in roots:
            f.write(json.dumps(dict(position=root.position, rows=ROWS, cols=COLS,
                                    prefix=list(root.prefix), source="test",
                                    source_sha256="0", family=root.family)) + "\n")
    with (directory / "child-diagnostic-roots.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "undrawn", "root_final", "root_source", "complete_known",
                    "legal_children", "solved_children", "optimal_legal_fraction",
                    "uniform_margin_regret"])
        for root, _ in roots:
            w.writerow([root.position, root.undrawn, root.root_final, "test", True,
                        len(root.exact), len(root.exact), len(root.optimal) / len(root.exact),
                        float(np.mean([root.root_final - v for v in root.exact.values()]))])
    with (directory / "child-diagnostic-children.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "edge", "final_margin_for_root_mover"])
        for root, _ in roots:
            for edge in sorted(root.exact):
                w.writerow([root.position, edge, root.exact[edge]])


class TestEndToEnd:
    def test_tables_load_and_cli_writes_every_table(self, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        oracle, rng = Oracle(ROWS, COLS), random.Random(3)
        roots = roots_with_prefix(oracle, rng, 3, 9) + [
            (pva.Root(3 + i, r.prefix, r.family, ROWS, COLS, r.undrawn, r.root_final, r.exact),
             b) for i, (r, b) in enumerate(roots_with_prefix(oracle, rng, 3, 6))]
        write_tables(tmp_path / "tables", roots)
        loaded = pva.load_tables(tmp_path / "tables")
        assert [r.exact for r in loaded] == [r.exact for r, _ in roots]
        for i in range(2):
            save_boxes_net(tiny_net(i), tmp_path / f"net{i}.pt")
        monkeypatch.setattr(sys, "argv", [
            "audit", "--tables", str(tmp_path / "tables"), "--output", str(tmp_path / "out"),
            "--checkpoint", f"a={tmp_path / 'net0.pt'}", "--checkpoint",
            f"b={tmp_path / 'net1.pt'}", "--resamples", "50", "--solver-undrawn", "2"])
        pva.main()
        out = tmp_path / "out"
        rows = {name: list(csv.DictReader((out / f"{name}.csv").open()))
                for name in ("children", "policy", "ranking", "roots")}
        children = sum(len(r.exact) for r, _ in roots)
        assert len(rows["children"]) == 2 * children
        assert len(rows["policy"]) == 2 * 2 * len(roots)
        assert len(rows["ranking"]) == 2 * len(pva.MODES) * len(pva.PREDICTORS) * len(roots)
        assert len(rows["roots"]) == 2 * len(roots)
        assert all(int(r["exact"]) == loaded[int(r["position"])].exact[int(r["edge"])]
                   for r in rows["children"])
        summary = json.loads((out / "summary.json").read_text())
        assert summary["roots"] == len(roots) and len(summary["checkpoint_sha256"]) == 2
        assert {d["metric"] for d in summary["paired_differences"]} >= {
            "optimal_mass", "argmax_regret", "spearman"}
        assert (out / "summary.md").read_text().startswith("## Policy at the root")
