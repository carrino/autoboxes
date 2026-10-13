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
    choice = json.loads((HERE / "data/probe-choice.json").read_text())
    assert (choice["threshold"], choice["budget"], choice["slice_nodes"]) == (26, 16384, 1024)
    lines = ["# A classical 5x5 yardstick", "",
             "Use **`boxes-classic`**: iterative-deepening alpha-beta with a one-second "
             "move deadline, chain/control evaluation, transposition table, exact forced-capture "
             "and equivalent-edge reductions, and budgeted exact-solver probes at ≤26 undrawn "
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
        assert settings["classic_settings"]["solver_max_undrawn"] == choice["threshold"]
        assert settings["classic_settings"]["probe_node_budget"] == choice["budget"]
        assert settings["classic_settings"]["probe_slice_nodes"] == choice["slice_nodes"]
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
    lines += ["", "The initial 30-edge/1,024-node build's seed-42 smoke match finished 4–0 "
              "against depth 6, mean margin +7.5. Its incomplete 116-game match is preserved "
              "under `data/preliminary-probe30/`. Both are excluded from the final matches, "
              "which restarted with the corrected build. Raw per-game timings, paired margins, "
              "settings, and summaries are retained in `data/`.", "",
              "## Probe calibration", "",
              "The original 1,024-node cap at 30 edges often left a heuristic leaf. Simply "
              "reducing the threshold to 26 did not resolve the low completion rate. The "
              "corrected engine uses a cumulative 16,384 nodes at ≤26 edges, retaining its "
              "solver table across 1,024-node slices and checking the clock between slices.", "",
              "Calibration sampled 1,400 development positions at 24–30 edges, seed 73. "
              "Every successful probe was checked against a fresh unrestricted exact answer. "
              "Selection preferred threshold 26 and the smallest tested cumulative budget "
              "(4,096 / 16,384 / 65,536) with ≥90% completion at each sampled count at or below "
              "the threshold. No held-out games or match outcomes selected this setting.", "",
              "| Undrawn | Positions | Original 1,024 solved | Sliced 16,384 solved "
              "| Sliced probe median / p90 ms |",
              "|---|---:|---:|---:|---:|"]
    with (HERE / "data/probe-calibration-summary.csv").open() as handle:
        single = {int(row["undrawn"]): row for row in csv.DictReader(handle)
                  if int(row["budget"]) == 1024}
    with (HERE / "data/probe-slices-summary.csv").open() as handle:
        sliced = {int(row["undrawn"]): row for row in csv.DictReader(handle)
                  if int(row["budget"]) == 16384}
    for undrawn in range(24, 31):
        old, new = single[undrawn], sliced[undrawn]
        if undrawn <= choice["threshold"]:
            assert float(new["solved_share"]) >= 0.9
        lines.append(f"| {undrawn} | {new['positions']} | {float(old['solved_share']):.1%} "
                     f"| {float(new['solved_share']):.1%} | {float(new['median_ms']):.2f} / "
                     f"{float(new['p90_ms']):.2f} |")
    lines += ["", "These are fresh-table probe completion rates on sampled game positions, "
              "not the engine's in-search hit rates. Warm tables and the remaining move "
              "deadline affect actual leaves. A timed-out or budget-exhausted probe remains "
              "heuristic; the engine never treats an unfinished solve as proof.", "",
              "## Evaluation and search details", "",
              "The score is the board's current margin plus an estimate of its remaining margin. "
              "The Python reference and C++ agree on randomly generated masks and symmetries. "
              "Simple chain/loop endings use the existing exact recurrence. For unresolved "
              "junctions, long-chain parity estimates who will control the ending, controlled "
              "value estimates the reward for that control, and the junction count reduces "
              "confidence. This is a heuristic, never an exact bound.", "",
              "The search uses safe/history ordering with a remembered best move, bounded "
              "transposition storage, and a steady-clock deadline. Each exact-solver probe is "
              "at most 16,384 cumulative nodes, with deadline checks on both sides of each "
              "1,024-node slice. An unfinished probe "
              "falls back to search/evaluation. On timeout, the agent returns the last completed "
              "iteration's move. Forced captures can return immediately.", "",
              "The engine is opt-in by registered name. Each agent is single-threaded and "
              "allocates a 4 MiB search table plus the default 16 MiB solver table. A new game "
              "starts fresh tables, and bindings release the GIL for independent games.", "",
              "## Validation and next measurement", "",
              "Python reference first: 8 tests, including controlled-value theorem checks against "
              "the exact recurrence and exhaustive small endgames. Reference plus C++: 29 tests, "
              "including exact tiny-board minimax with tiny collision-prone tables, "
              "move optimality, perspective, timeouts, configuration checks, and regression "
              "positions whose one-slice heuristic leaves now resolve exactly. "
              "Full suite: **380 passed, 24 skipped**. "
              "New Python is ruff/strict-mypy clean. Build SHA256: "
              f"`{next(iter(builds))}`.", "",
              "On the GPU machine, use `--a ckpt:<checkpoint> --b boxes-classic --opening_moves 4 "
              "--num_games 200` with the usual net search settings. Record the net's solver reach "
              "and budget explicitly. This CPU arena does not establish strength against the nets "
              "or external Windows engines; that is the next comparison.", ""]
    (HERE / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
