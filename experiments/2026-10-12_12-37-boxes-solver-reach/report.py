"""Generate the task report only after all frozen benchmark variants finish."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_rows(version: str, tag: str) -> list[dict[str, str]]:
    """Read a complete benchmark and check its input-position identities."""
    with (HERE / "data" / f"{version}-{tag}.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3000
    assert {int(row["index"]) for row in rows} == set(range(3000))
    return sorted(rows, key=lambda row: int(row["index"]))


def quantiles(rows: list[dict[str, str]], field: str) -> str:
    """Median, 90th percentile, and maximum, formatted with explicit common units."""
    values = np.quantile([float(row[field]) for row in rows], [0.5, 0.9, 1.0])
    return " / ".join(f"{float(value):,.2f}" if field.endswith("_ms") else f"{float(value):,.0f}"
                      for value in values)


def share(rows: list[dict[str, str]], budget: str) -> float:
    """Share with an exact answer from a fresh budgeted solver."""
    return statistics.mean(int(row[f"{budget}_solved"]) for row in rows)


def main() -> None:
    """Use only complete CSVs; the second candidate's decision was made on development."""
    decision = json.loads((HERE / "data/candidate2-decision.json").read_text())
    chosen = "candidate2" if decision["keep"] else "candidate1"
    with (HERE / "data/region_stats-5x5-mid-cost.csv").open() as handle:
        region_positions = sum(int(row["positions"]) for row in csv.DictReader(handle))
    data = {(version, tag): read_rows(version, tag)
            for version in ("baseline", "candidate1", "candidate2") for tag in ("chains", "mid")}
    for tag in ("chains", "mid"):
        for version in ("candidate1", "candidate2"):
            for old, new in zip(data["baseline", tag], data[version, tag], strict=True):
                assert (old["index"], old["mask"], old["value"]) == (
                    new["index"], new["mask"], new["value"])
                assert int(new["table_bytes"]) == 16 * 1024 * 1024
                if version == "candidate1":
                    assert new["exact_nodes"] == old["exact_nodes"]
    lines = ["# Exact solver reach: results", "",
             f"Final implementation: **{chosen}**. All 6,000 benchmark values match the baseline; "
             "both candidates were checked on both complete mask files.", "",
             "The Python API and one-thread-per-solver model are unchanged. The default table "
             "remains 1,048,576 entries / 16 MiB. Candidate 1 removes temporary heap allocations; "
             "candidate 2 uses existing entry padding to retain both alpha-beta bounds.", "",
             "## Attempts", "",
             "1. Profile: five development positions, about 1.27 million search nodes. Component "
             "walking, temporary vectors, and repeated chain analysis were prominent hotspots. "
             "The initial 100-position instrumented run was stopped; "
             "it supplied no reported timing.",
             "2. Allocation-free chain walking: same traversal order and values. The initial "
             "20-position development probe preserved every node count and had a provisional "
             "median paired speedup of 1.29×. Full results below supersede that small probe.",
             f"3. Two table bounds in 16 bytes: {'kept' if decision['keep'] else 'reverted'}. "
             f"Development geometric-mean speedup versus candidate 1: "
             f"{decision['geometric_mean_speedup_vs_candidate1']:.3f}×; total search nodes "
             f"{decision['total_nodes_candidate1']:,} → {decision['total_nodes_candidate2']:,}. "
             "The predeclared rule required ≥2% geometric-mean speedup and no increase in total "
             "nodes. Held-out timings did not enter this decision.", "",
             "## Before and after", "",
             "Times exclude solver construction. Nodes and milliseconds are shown as median / "
             "p90 / maximum. Every unrestricted and budgeted call starts with a fresh table. "
             "Raw per-position CSVs include all candidates, masks, exact values, and timings.", "",
             "| Dataset | Attempt | Geometric mean paired speedup | Total solve-time speedup "
             "| Total nodes |",
             "|---|---|---:|---:|---:|"]
    for tag in ("chains", "mid"):
        baseline = data["baseline", tag]
        for version in ("candidate1", "candidate2"):
            rows = data[version, tag]
            ratios = [float(old["exact_ms"]) / float(new["exact_ms"])
                      for old, new in zip(baseline, rows, strict=True)]
            geometric = math.exp(statistics.mean(math.log(ratio) for ratio in ratios))
            aggregate = (sum(float(row["exact_ms"]) for row in baseline)
                         / sum(float(row["exact_ms"]) for row in rows))
            nodes = sum(int(row["exact_nodes"]) for row in rows)
            lines.append(f"| 5x5-{tag} | {version} | {geometric:.3f}× | {aggregate:.3f}× "
                         f"| {nodes:,} |")
    lines += ["", "Candidate 1 preserves every per-position node count as well as every value. "
              "Timing measurements share the machine with other single-threaded CPU tasks; "
              "node counts and budget success rates do not depend on scheduling.", ""]
    first, second = data["candidate1", "mid"], data["candidate2", "mid"]
    heldout_ratios = [float(old["exact_ms"]) / float(new["exact_ms"])
                     for old, new in zip(first, second, strict=True)]
    heldout_geometric = math.exp(statistics.mean(math.log(r) for r in heldout_ratios))
    heldout_aggregate = (sum(float(row["exact_ms"]) for row in first)
                        / sum(float(row["exact_ms"]) for row in second))
    lines += [f"The second change is not a uniform timing improvement over candidate 1: "
              f"on held-out positions its geometric-mean paired speedup is "
              f"{heldout_geometric:.3f}×, while its total solve-time speedup is "
              f"{heldout_aggregate:.3f}×. The development-only selection rule was retained; "
              "the held-out result was not used to retune or change the candidate.", ""]
    eligible: dict[str, list[int]] = {}
    for tag in ("chains", "mid"):
        lines += [f"### 5x5-{tag}", "",
                  "| Version | Undrawn | Positions | Nodes med / p90 / max | ms med / p90 / max "
                  "| Solved ≤100k | Solved ≤1M |",
                  "|---|---:|---:|---|---|---:|---:|"]
        baseline_30 = statistics.median(float(row["exact_ms"]) for row in data["baseline", tag]
                                        if int(row["undrawn"]) == 30)
        eligible[tag] = []
        for version in ("baseline", chosen):
            for undrawn in range(30, 35):
                rows = [row for row in data[version, tag] if int(row["undrawn"]) == undrawn]
                lines.append(f"| {version} | {undrawn} | {len(rows)} | "
                             f"{quantiles(rows, 'exact_nodes')} | {quantiles(rows, 'exact_ms')} | "
                             f"{share(rows, '100k'):.1%} | {share(rows, '1m'):.1%} |")
                if (version == chosen and share(rows, "100k") >= 0.8
                        and statistics.median(float(row["exact_ms"]) for row in rows)
                        <= 1.1 * baseline_30):
                    eligible[tag].append(undrawn)
        lines.append("")
    common = sorted(set(eligible["chains"]) & set(eligible["mid"]))
    reach = str(max(common)) if common else "none in the tested 30–34 band"
    lines += ["## Practical reach", "",
              "For this report, ‘mostly’ was predeclared as at least 80% solved within 100k "
              "nodes, and ‘about today's cost of 30’ as median time no more than 110% of that "
              "dataset's baseline at 30. Both datasets must meet the condition.", "",
              f"Largest qualifying SOLVER_N: **{reach}**. Development qualifying counts: "
              f"{eligible['chains']}; held-out: {eligible['mid']}.", ""]
    if not common or max(common) < 32:
        lines += ["The requested 32-at-30-cost target was not established. Do not raise the "
                  "self-play threshold to 32 on these results alone. The tables show the speed "
                  "gain and the budget failures separately; faster nodes need not make more "
                  "positions fit a fixed node budget.", ""]
    lines += ["## Independent regions at fixed undrawn count", "",
              "This comparison uses exact baseline costs on the held-out mask file, comparing "
              "largest region ≤24 edges with an unsplit board at the same undrawn count.", "",
              "| Undrawn | Small-region n | Median nodes | Unsplit n | Median nodes | Ratio |",
              "|---|---:|---:|---:|---:|---:|"]
    ratios = []
    for undrawn in (30, 32):
        group = [row for row in data["baseline", "mid"] if int(row["undrawn"]) == undrawn]
        small = [int(row["exact_nodes"]) for row in group if int(row["largest_region"]) <= 24]
        unsplit = [int(row["exact_nodes"]) for row in group
                   if int(row["largest_region"]) == undrawn]
        ratio = statistics.median(unsplit) / statistics.median(small)
        ratios.append(ratio)
        lines.append(f"| {undrawn} | {len(small)} | {statistics.median(small):,.0f} | "
                     f"{len(unsplit)} | {statistics.median(unsplit):,.0f} | {ratio:.2f}× |")
    lines += ["", ("Yes, smaller largest regions correlate with lower cost at both counts."
                   if all(ratio > 1 for ratio in ratios) else
                   "The association is mixed at these two fixed undrawn counts."),
              "This is an observational comparison, not a claim that the regions can be solved "
              "independently and their margins summed. The required separate "
              f"`region_stats --solver-nodes 30 32` run requested 4,000 positions and sampled "
              f"{region_positions:,} distinct positions from the available games. "
              "It is recorded in `logs/region-cost.log`; "
              "its 2M-node cap censors the hardest tails.", "",
              "## Validation and next work", "",
              "See README for commands, machine, setup, and the baseline checks. Exactness uses "
              "the existing Python oracle, exhaustive tiny boards, full-window/MTD(f) comparisons, "
              "new tiny-table budget-restart tests, all Catch2 binaries, "
              "and the full 6,000-position "
              "corpus. Candidate CSVs are retained even for a reverted change.", ""]
    final_tests = re.search(r"\d+ passed, \d+ skipped", (HERE / "data/final-tests.txt").read_text())
    assert final_tests is not None
    lines += [f"Final solver suite: **{final_tests.group()}**; all three Catch2 binaries pass. "
              "Logs are in `data/final-tests.txt` and `data/final-catch2.txt`.", ""]
    integration_log = HERE / "data/integration-validation.txt"
    if integration_log.is_file():
        integration_tests = re.search(r"\d+ passed, \d+ skipped", integration_log.read_text())
        assert integration_tests is not None
        lines += [f"Combined with the corrected classical engine: "
                  f"**{integration_tests.group()}**, plus all three Catch2 binaries. "
                  "This separate integration build did not merge either branch. "
                  "See `data/integration-validation.txt`.", ""]
    lines += ["Next: attack the remaining tree size with a proved region-aware reduction. "
              "Keep using development positions for design and reserve 5x5-mid "
              "for final reports.", ""]
    (HERE / "report.md").write_text("\n".join(lines))
    (HERE / "data/reach.json").write_text(json.dumps({"chosen": chosen, "eligible": eligible,
                                                     "common": common}, indent=2) + "\n")


if __name__ == "__main__":
    main()
