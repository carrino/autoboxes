"""Summarize the completed 200-game paired matches and retain all raw pair margins."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPPONENTS = ("boxes-ab-d6", "boxes-ab-d4-s28", "boxes-classic-fast", "boxes-classic-tiny")


def main() -> None:
    """Require full games, consistent build provenance, and correct summary aggregation."""
    lines = ["# A classical 5x5 yardstick", "",
             "Use **`boxes-classic`**: iterative-deepening alpha-beta with a one-second "
             "move deadline, chain/control evaluation, transposition table, exact forced-capture "
             "and equivalent-edge reductions, and budgeted exact-solver probes at ≤30 undrawn "
             "edges. `boxes-classic-fast` uses 100 ms and `boxes-classic-tiny` uses 20 ms.", "",
             "## Paired arena", "",
             "Each comparison has 200 games / 100 pairs, seed 0, four uniformly random opening "
             "edges shared within each color-swapped pair. Pair wins are decided by summed "
             "margin; ties count one half. Intervals are the repository arena's 95% Wilson "
             "intervals on paired scores. Seconds per move are weighted by actual move counts, "
             "including forced captures and fast exact finishes.", "",
             "| Opponent | Pair W/D/L | Pair score [95% CI] | Mean margin | Classic s/move "
             "| Opponent s/move |",
             "|---|---|---|---:|---:|---:|"]
    builds = set()
    summary_rows = []
    for opponent in OPPONENTS:
        base = HERE / "data" / opponent
        summary = json.loads(base.with_suffix(".summary.json").read_text())
        settings = json.loads(base.with_suffix(".settings.json").read_text())
        builds.add(settings["extension_sha256"])
        with base.with_suffix(".csv").open() as handle:
            games = list(csv.DictReader(handle))
        games.sort(key=lambda row: int(row["game"]))
        assert [int(row["game"]) for row in games] == list(range(200))
        assert summary["games"] == settings["games"] == 200
        assert summary["a"] == settings["a"] == "boxes-classic"
        assert summary["b"] == settings["b"] == opponent
        assert settings["seed"] == 0 and settings["opening_moves"] == 4
        margins = [int(row["margin"]) for row in games]
        pairs = [margins[i] + margins[i + 1] for i in range(0, 200, 2)]
        wins, draws, losses = (sum(m > 0 for m in pairs), sum(m == 0 for m in pairs),
                              sum(m < 0 for m in pairs))
        assert abs((wins + 0.5 * draws) / 100 - summary["a_pair_score"]) < 0.00006
        assert abs(statistics.mean(margins) - summary["a_mean_margin"]) < 0.0006
        lines.append(f"| {opponent} | {wins}/{draws}/{losses} | {summary['a_pair_score']:.3f} "
                     f"{summary['a_pair_score_ci95']} | {summary['a_mean_margin']:+.2f} | "
                     f"{summary['a_sec_per_move']:.4f} | {summary['b_sec_per_move']:.4f} |")
        summary_rows.append(summary)
    assert len(builds) == 1
    with (HERE / "data/summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    lines += ["", "A separate seed-42 smoke match (four games) finished 4–0 against depth 6, "
              "mean margin +7.5; it is not included in the 200-game results. Raw per-game "
              "timings, paired margins, settings, and summaries are retained in `data/`.", "",
              "## Evaluation and search details", "",
              "The score is the board's current margin plus an estimate of its remaining margin. "
              "The Python reference and C++ agree on randomly generated masks and symmetries. "
              "Simple chain/loop endings use the existing exact recurrence. For unresolved "
              "junctions, long-chain parity estimates who will control the ending, controlled "
              "value estimates the reward for that control, and the junction count reduces "
              "confidence. This is a heuristic, never an exact bound.", "",
              "The search uses safe/history ordering with a remembered best move, bounded "
              "transposition storage, and a steady-clock deadline. Each exact-solver probe is "
              "at most 1,024 nodes with checks on both sides of the call. An unfinished probe "
              "falls back to search/evaluation. On timeout, the agent returns the last completed "
              "iteration's move. Forced captures can return immediately.", "",
              "The engine is opt-in by registered name. Each agent is single-threaded and "
              "allocates a 4 MiB search table plus the default 16 MiB solver table. A new game "
              "starts fresh tables, and bindings release the GIL for independent games.", "",
              "## Validation and next measurement", "",
              "Python reference first: 8 tests, including controlled-value theorem checks against "
              "the exact recurrence and exhaustive small endgames. Reference plus C++: 26 tests, "
              "including exact tiny-board minimax with tiny collision-prone tables, "
              "move optimality, perspective, timeouts, and configuration checks. "
              "Full suite: **377 passed, 24 skipped**. "
              "New Python is ruff/strict-mypy clean. Build SHA256: "
              f"`{next(iter(builds))}`.", "",
              "On the GPU machine, use `--a ckpt:<checkpoint> --b boxes-classic --opening_moves 4 "
              "--num_games 200` with the usual net search settings. Record the net's solver reach "
              "and budget explicitly. This CPU arena does not establish strength against the nets "
              "or external Windows engines; that is the next comparison.", ""]
    (HERE / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
