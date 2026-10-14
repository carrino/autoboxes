"""Read a completed search probe and compare paired moves by source-family bootstrap.

No neural network or solver is loaded. Intervals describe this development sample;
they are exploratory, uncorrected for multiple comparisons, and never promote a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

LABELS = ("root_final", "policy_final", "search_final")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_probe(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Validate completeness, frozen metadata, and one record per case and position."""
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("complete") is not True:
        raise ValueError("Probe is incomplete; wait for summary.json complete=true")
    arguments = summary["arguments"]
    cases = [f"{mix}:s{sims}:lambda{weight}" for mix, sims, weight in itertools.product(
        dict.fromkeys(arguments["mixes"]), dict.fromkeys(arguments["simulations"]),
        dict.fromkeys(arguments["lambdas"]))]
    if (not cases or summary.get("completed_cases") != cases
            or set(summary.get("cases", {})) != set(cases)):
        raise ValueError("Completed cases do not match the requested cases")
    frozen_path = directory / "positions.jsonl"
    if _hash(frozen_path) != summary.get("positions_sha256"):
        raise ValueError("Frozen positions hash does not match summary.json")
    frozen = [json.loads(line) for line in frozen_path.read_text().splitlines() if line.strip()]
    if not frozen or summary.get("sampled") != len(frozen):
        raise ValueError("Sampled count does not match frozen positions")
    rows, cols = arguments["rows"], arguments["cols"]
    if type(rows) is not int or type(cols) is not int or min(rows, cols) < 1:
        raise ValueError("Invalid board dimensions")
    edges = (rows + 1) * cols + rows * (cols + 1)
    for position in frozen:
        prefix = position["prefix"]
        if ((position["rows"], position["cols"]) != (rows, cols)
                or not isinstance(position["family"], str) or not position["family"]
                or not isinstance(prefix, list)
                or any(type(edge) is not int or not 0 <= edge < edges for edge in prefix)
                or len(set(prefix)) != len(prefix)
                or not arguments["min_undrawn"] <= edges - len(prefix)
                <= arguments["max_undrawn"]):
            raise ValueError("Invalid frozen position metadata")
    if summary.get("source_families") != len({p["family"] for p in frozen}):
        raise ValueError("Source family count disagrees with frozen positions")
    with (directory / "positions.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"position", "case", "mix", "simulations", "margin_utility_lambda",
                    "undrawn", "source_family", "policy_edge", "search_edge",
                    "search_seconds", *LABELS}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing required positions.csv columns")
        raw_records = list(reader)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    roots: dict[int, int | None] = {}
    for raw in raw_records:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("Malformed CSV row")
        record: dict[str, Any] = dict(raw)
        index, case = int(raw["position"]), raw["case"]
        if not 0 <= index < len(frozen) or case not in cases or (case, index) in seen:
            raise ValueError("Unexpected or duplicate case/position row")
        seen.add((case, index))
        position = frozen[index]
        if (raw["source_family"] != position["family"]
                or int(raw["undrawn"]) != edges - len(position["prefix"])
                or case != f"{raw['mix']}:s{int(raw['simulations'])}"
                f":lambda{float(raw['margin_utility_lambda'])}"):
            raise ValueError("CSV metadata disagrees with its frozen position or case")
        record.update(position=index, undrawn=int(raw["undrawn"]))
        for key in LABELS:
            value = None if raw[key] == "" else int(raw[key])
            if value is not None and (abs(value) > rows * cols
                                      or (value - rows * cols) % 2):
                raise ValueError("Invalid exact final margin")
            record[key] = value
        root = record["root_final"]
        if index in roots and roots[index] != root:
            raise ValueError("Root values disagree between cases")
        roots[index] = root
        for choice in ("policy", "search"):
            edge, child = int(raw[f"{choice}_edge"]), record[f"{choice}_final"]
            if not 0 <= edge < edges or edge in position["prefix"]:
                raise ValueError("Selected edge is illegal in the frozen position")
            if root is not None and child is not None and child > root:
                raise ValueError("Exact child value exceeds exact minimax root")
        seconds = float(raw["search_seconds"])
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid search timing")
        record["search_seconds"] = seconds
        records.append(record)
    if len(seen) != len(cases) * len(frozen):
        raise ValueError("Missing case/position rows; probe results are incomplete")
    return summary, records, cases


def paired_metric(values: NDArray[np.float64], eligible: NDArray[np.bool_],
                  family_index: NDArray[np.int64], draws: NDArray[np.int64],
                  reference_index: int) -> list[dict[str, Any]]:
    """Bootstrap paired position-weighted means using shared family multiplicities."""
    groups = draws.shape[1]
    sizes = np.bincount(family_index[eligible], minlength=groups)
    totals = np.zeros((groups, values.shape[1]), dtype=np.float64)
    np.add.at(totals, family_index[eligible], values[eligible])
    count, families = int(sizes.sum()), int(np.count_nonzero(sizes))
    means = totals.sum(axis=0) / count if count else np.full(values.shape[1], np.nan)
    denominators = draws @ sizes
    valid = denominators > 0
    differences = totals - totals[:, reference_index:reference_index + 1]
    boot = (draws[valid] @ differences) / denominators[valid, None]
    result = []
    for index, value in enumerate(means):
        interval = (np.quantile(boot[:, index], [0.025, 0.975]).tolist()
                    if families >= 2 and np.any(valid) else None)
        reason = ("fewer than two eligible source families" if families < 2 else
                  "no bootstrap replicate has an eligible denominator")
        result.append(dict(
            mean=float(value) if count else None,
            delta_vs_reference=float(value - means[reference_index]) if count else None,
            delta_ci95=interval, positions=count, families=families,
            bootstrap_replicates=int(valid.sum()) if interval is not None else 0,
            ci_unavailable=None if interval is not None else reason))
    return result


def build_report(directory: Path, reference: str | None = None, seed: int = 20261014,
                 resamples: int = 10_000) -> dict[str, Any]:
    """Analyze every case on the same complete-known positions, overall and by band."""
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    summary, records, cases = load_probe(directory)
    references = [case for case in cases if case.startswith("reference:")]
    if reference is None and len(references) != 1:
        raise ValueError("Specify --reference-case when the probe has multiple/no reference cases")
    reference = reference or references[0]
    if reference not in cases:
        raise ValueError("Reference case is not in this probe")
    reference_index = cases.index(reference)
    by_case = {case: {r["position"]: r for r in records if r["case"] == case} for case in cases}
    first = by_case[cases[0]]
    common = {index for index in first if all(
        by_case[case][index][key] is not None for case in cases for key in LABELS)}
    result: dict[str, Any] = dict(
        schema=1, input=str(directory.resolve()),
        input_sha256={name: _hash(directory / name) for name in
                      ("summary.json", "positions.jsonl", "positions.csv")},
        reference_case=reference, case_order=cases,
        checkpoint_sha256=summary.get("checkpoint_sha256", {}),
        common_rule="Root, policy child and search child exact in every case; unknowns excluded",
        regret_units="final score-margin points, not boxes captured",
        delta_direction="negative regret delta and positive win-preservation delta are better",
        scope="Exploratory development comparison; no promotion or multiple-test correction. "
              "Coverage can favor easier solves; source positions may have been used in training.",
        bootstrap=dict(method="paired source-family cluster percentile", seed=seed,
                       resamples=resamples, confidence=0.95,
                       weighting="Resample families, retain all their positions; position-weighted "
                                 "means. Identical draws for every case and metric within a band."),
        bands={})
    rng = np.random.default_rng(seed)
    for band, minimum, maximum in (("all", 0, math.inf), ("35..36", 35, 36), ("37..38", 37, 38)):
        sampled = sorted(i for i, row in first.items() if minimum <= row["undrawn"] <= maximum)
        indices = [i for i in sampled if i in common]
        families = sorted({first[i]["source_family"] for i in indices})
        family_ids = {family: index for index, family in enumerate(families)}
        family_index = np.array([family_ids[first[i]["source_family"]] for i in indices],
                                dtype=np.int64)
        draws = (rng.multinomial(len(families), np.full(len(families), 1 / len(families)),
                                 size=resamples) if families else
                 np.zeros((resamples, 0), dtype=np.int64))
        band_result: dict[str, Any] = dict(
            sampled=len(sampled),
            sampled_families=len({first[i]["source_family"] for i in sampled}),
            common_known=len(indices), common_families=len(families),
            excluded=len(sampled) - len(indices), common_position_indices=indices, cases={})
        for case in cases:
            band_result["cases"][case] = dict(
                known={key: sum(by_case[case][i][key] is not None for i in sampled)
                       for key in LABELS},
                total_search_seconds=sum(by_case[case][i]["search_seconds"] for i in sampled))
        roots = np.array([first[i]["root_final"] for i in indices], dtype=np.float64)
        for choice in ("search", "policy"):
            children = np.array([[by_case[case][i][f"{choice}_final"] for case in cases]
                                 for i in indices], dtype=np.float64)
            children = children.reshape(len(indices), len(cases))
            for metric, values, eligible in (
                ("margin_regret", roots[:, None] - children, np.ones(len(indices), dtype=np.bool_)),
                ("win_preservation", (children > 0).astype(np.float64), roots > 0),
            ):
                estimates = paired_metric(values, eligible, family_index, draws, reference_index)
                for case, estimate in zip(cases, estimates):
                    band_result["cases"][case][f"{choice}_{metric}"] = estimate
        result["bands"][band] = band_result
    return result


def write_report(report: dict[str, Any], output: Path) -> None:
    """Write JSON and one row per band/case/metric into a fresh external directory."""
    source = Path(report["input"]).resolve()
    if output.exists() or output.resolve().is_relative_to(source):
        raise ValueError("--output must be a fresh directory outside the input directory")
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "comparisons.csv").open("w", newline="") as stream:
        fields = ["band", "case", "metric", "mean", "delta_vs_reference", "delta_ci95_low",
                  "delta_ci95_high", "positions", "families", "bootstrap_replicates",
                  "sampled", "sampled_families", "common_known", "common_families", "excluded"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for band, data in report["bands"].items():
            for case, metrics in data["cases"].items():
                for name, estimate in metrics.items():
                    if name in ("known", "total_search_seconds"):
                        continue
                    row = {key: estimate[key] for key in
                           ("mean", "delta_vs_reference", "positions", "families",
                            "bootstrap_replicates")}
                    interval = estimate["delta_ci95"] or [None, None]
                    row.update(band=band, case=case, metric=name, delta_ci95_low=interval[0],
                               delta_ci95_high=interval[1])
                    row.update({key: data[key] for key in fields[-5:]})
                    writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    """Print paired search comparisons and optionally save the complete report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="completed probe output directory")
    parser.add_argument("--output", type=Path, help="fresh report directory outside --input")
    parser.add_argument("--reference-case",
                        help="required for sweeps containing multiple references")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)
    report = build_report(args.input, args.reference_case, args.seed, args.resamples)
    print(report["scope"])
    print(f"Reference: {report['reference_case']}; regret deltas < 0 are better; "
          "win-preservation deltas > 0 are better.")
    for band, data in report["bands"].items():
        print(f"\n{band}: {data['common_known']}/{data['sampled']} complete-known positions, "
              f"{data['common_families']}/{data['sampled_families']} families")
        print("case | metric | mean | paired delta [95% CI] | positions / families")
        for case, metrics in data["cases"].items():
            for name in ("search_margin_regret", "search_win_preservation"):
                estimate = metrics[name]
                if estimate["mean"] is None:
                    print(f"{case} | {name} | unavailable (no eligible positions)")
                    continue
                interval = estimate["delta_ci95"]
                ci = (f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval is not None
                      else f"unavailable ({estimate['ci_unavailable']})")
                print(f"{case} | {name} | {estimate['mean']:.3f} | "
                      f"{estimate['delta_vs_reference']:+.3f} {ci} | "
                      f"{estimate['positions']} / {estimate['families']}")
    if args.output:
        write_report(report, args.output)
        print(f"Wrote {args.output / 'report.json'} and {args.output / 'comparisons.csv'}")


if __name__ == "__main__":
    main()
