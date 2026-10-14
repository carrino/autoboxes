"""Compare two completed temperature probes on identical, jointly solved positions.

This reads saved artifacts only. Source-family bootstrap intervals are exploratory
development diagnostics; they neither establish playing strength nor promote a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from alpha_go.boxes.probe_report import LABELS, load_probe, paired_metric
else:
    # Reuse the report implementation without alpha_go's eager Torch/C++ imports.
    _path = Path(__file__).resolve().parents[2] / "src/alpha_go/boxes/probe_report.py"
    _spec = importlib.util.spec_from_file_location("boxes_probe_report_readonly", _path)
    assert _spec is not None and _spec.loader is not None
    _report = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_report)
    LABELS, load_probe, paired_metric = _report.LABELS, _report.load_probe, _report.paired_metric

MODELS = ("reference", "candidate")
PATH_ARGUMENTS = {"reference", "candidate", "games", "positions_file", "output", "label_cache"}
SEARCH_ARGUMENTS = {
    "rows", "cols", "min_undrawn", "max_undrawn", "simulations", "lambdas", "mixes",
    "label_budget", "label_table_entries", "solver", "solver_budget", "solver_table_entries",
    "c_puct", "leaf_batch_size", "margin_utility_k", "prove_terminals", "shape_terminals",
    "seed", "cpu",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_hash(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _validate_pair(summaries: list[dict[str, Any]], cases: list[list[str]]) -> None:
    baseline, candidate = summaries
    if (baseline["positions_sha256"] != candidate["positions_sha256"]
            or baseline["source_families"] != candidate["source_families"]):
        raise ValueError("Frozen positions and source families must be identical")
    for summary, names in zip(summaries, cases):
        arguments = summary["arguments"]
        if not SEARCH_ARGUMENTS <= arguments.keys():
            raise ValueError("Missing search settings in summary.json")
        if len(names) != 2 or {name.split(":")[0] for name in names} != set(MODELS):
            raise ValueError("Each run must contain exactly one reference and one candidate case")
        temperature = arguments.get("policy_temperature")
        if (type(temperature) not in (float, int) or not math.isfinite(temperature)
                or temperature <= 0):
            raise ValueError("Policy temperatures must be finite and positive")
        hashes = summary.get("checkpoint_sha256", {})
        if set(hashes) != set(MODELS) or not all(_is_hash(v) for v in hashes.values()):
            raise ValueError("Missing or invalid checkpoint hashes")
        code = summary.get("code_sha256", {})
        if (not code or not all(_is_hash(v) for v in code.values())
                or not _is_hash(summary.get("cpp_sha256")) or not summary.get("device")):
            raise ValueError("Missing or invalid code/build/device provenance")
    ignored = PATH_ARGUMENTS | {"policy_temperature"}
    settings = [{key: value for key, value in s["arguments"].items() if key not in ignored}
                for s in summaries]
    if settings[0] != settings[1] or cases[0] != cases[1]:
        raise ValueError("Search settings must match except policy_temperature")
    for key in ("checkpoint_sha256", "code_sha256", "cpp_sha256", "device"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Mismatched {key}")
    if baseline["arguments"]["policy_temperature"] == candidate["arguments"]["policy_temperature"]:
        raise ValueError("Runs must have different policy temperatures")


def build_comparison(baseline: Path, candidate: Path, seed: int = 20261014,
                     resamples: int = 10_000) -> dict[str, Any]:
    """Return temperature effects and their paired interaction on a common denominator."""
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    loaded = [load_probe(path) for path in (baseline, candidate)]
    summaries, records, cases = [list(items) for items in zip(*loaded)]
    _validate_pair(summaries, cases)
    by_run = [{name.split(":")[0]: {r["position"]: r for r in rows if r["case"] == name}
               for name in names} for rows, names in zip(records, cases)]
    first = by_run[0]["reference"]
    roots: dict[int, int] = {}
    children: dict[tuple[int, int], int] = {}
    for models in by_run:
        for rows in models.values():
            for index, row in rows.items():
                if row["root_final"] is not None:
                    if index in roots and roots[index] != row["root_final"]:
                        raise ValueError("Exact root labels disagree across runs")
                    roots[index] = row["root_final"]
                for choice in ("policy", "search"):
                    value = row[f"{choice}_final"]
                    key = index, int(row[f"{choice}_edge"])
                    if value is not None:
                        if key in children and children[key] != value:
                            raise ValueError("Exact labels for the same child disagree")
                        children[key] = value
    common = {i for i in first if all(models[model][i][key] is not None
                                    for models in by_run for model in MODELS for key in LABELS)}
    result: dict[str, Any] = dict(
        schema=1, inputs={name: str(path.resolve()) for name, path in
                          (("baseline", baseline), ("candidate", candidate))},
        input_sha256={name: {file: _hash(path / file) for file in
                             ("summary.json", "positions.jsonl", "positions.csv")}
                      for name, path in (("baseline", baseline), ("candidate", candidate))},
        temperatures={name: summary["arguments"]["policy_temperature"] for name, summary in
                      zip(("baseline", "candidate"), summaries)},
        checkpoint_sha256=summaries[0]["checkpoint_sha256"],
        common_rule="Root, policy child and search child exact in every case in both runs",
        regret_units="final score-margin points, not boxes captured",
        delta_direction="candidate-temperature minus baseline-temperature; negative regret and "
                        "positive win-preservation deltas are better",
        interaction="candidate model temperature change minus reference model temperature change",
        scope="Exploratory development diagnostic; no promotion or multiple-test correction. "
              "Coverage can favor easier solves; positions may have been used in training. "
              "Intervals describe sampled families, not repeated-search variability.",
        bootstrap=dict(method="paired source-family cluster percentile", seed=seed,
                       resamples=resamples, confidence=0.95,
                       weighting="Whole families resampled; position-weighted means; shared draws "
                                 "for both temperatures, models and metrics within a band."),
        bands={})
    rng = np.random.default_rng(seed)
    for band, low, high in (("all", 0, math.inf), ("35..36", 35, 36), ("37..38", 37, 38)):
        sampled = sorted(i for i, row in first.items() if low <= row["undrawn"] <= high)
        indices = [i for i in sampled if i in common]
        families = sorted({first[i]["source_family"] for i in indices})
        family_ids = {family: i for i, family in enumerate(families)}
        family_index = np.array([family_ids[first[i]["source_family"]] for i in indices],
                                dtype=np.int64)
        draws = (rng.multinomial(len(families), np.full(len(families), 1 / len(families)),
                                 size=resamples) if families else
                 np.zeros((resamples, 0), dtype=np.int64))
        data: dict[str, Any] = dict(
            sampled=len(sampled),
            sampled_families=len({first[i]["source_family"] for i in sampled}),
            common_known=len(indices), common_families=len(families),
            excluded=len(sampled) - len(indices), common_position_indices=indices,
            known={run: {model: {key: sum(models[model][i][key] is not None for i in sampled)
                                 for key in LABELS} for model in MODELS}
                   for run, models in zip(("baseline", "candidate"), by_run)},
            models={model: {} for model in MODELS}, difference_of_changes={})
        root_values = np.array([first[i]["root_final"] for i in indices], dtype=np.float64)
        # Columns: original model at T1/T2, then newer model at T1/T2.
        child_values = np.array([[models[model][i]["search_final"]
                                  for model in MODELS for models in by_run] for i in indices],
                                dtype=np.float64).reshape(len(indices), 4)
        for metric, values, eligible in (
            ("search_margin_regret", root_values[:, None] - child_values,
             np.ones(len(indices), dtype=np.bool_)),
            ("search_win_preservation", (child_values > 0).astype(np.float64), root_values > 0),
        ):
            for offset, model in enumerate(MODELS):
                estimates = paired_metric(values[:, 2 * offset:2 * offset + 2], eligible,
                                          family_index, draws, 0)
                data["models"][model][metric] = dict(
                    baseline_mean=estimates[0]["mean"], candidate_mean=estimates[1]["mean"],
                    **{key: value for key, value in estimates[1].items() if key != "mean"})
            changes = values[:, [1, 3]] - values[:, [0, 2]]
            interaction = paired_metric(changes, eligible, family_index, draws, 0)
            data["difference_of_changes"][metric] = dict(
                reference_change=interaction[0]["mean"], candidate_change=interaction[1]["mean"],
                **{key: value for key, value in interaction[1].items() if key != "mean"})
        result["bands"][band] = data
    return result


def write_comparison(report: dict[str, Any], output: Path) -> None:
    """Write a new report outside both inputs, without modifying any input artifact."""
    if (output.exists() or any(output.resolve().is_relative_to(Path(source).resolve())
                               for source in report["inputs"].values())):
        raise ValueError("--output must be a fresh directory outside both input directories")
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "comparisons.csv").open("w", newline="") as stream:
        fields = ["band", "contrast", "metric", "baseline_mean", "candidate_mean", "delta",
                  "ci95_low", "ci95_high", "positions", "families", "bootstrap_replicates",
                  "sampled", "common_known", "excluded"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for band, data in report["bands"].items():
            for contrast, metrics in [*data["models"].items(),
                                      ("difference_of_changes", data["difference_of_changes"])]:
                for metric, estimate in metrics.items():
                    interval = estimate["delta_ci95"] or [None, None]
                    writer.writerow(dict(
                        band=band, contrast=contrast, metric=metric,
                        baseline_mean=estimate.get("baseline_mean",
                                                   estimate.get("reference_change")),
                        candidate_mean=estimate.get("candidate_mean",
                                                    estimate.get("candidate_change")),
                        delta=estimate["delta_vs_reference"], ci95_low=interval[0],
                        ci95_high=interval[1], **{key: estimate[key] for key in
                                               ("positions", "families", "bootstrap_replicates")},
                        **{key: data[key] for key in ("sampled", "common_known", "excluded")}))


def main(argv: list[str] | None = None) -> None:
    """Compare saved probes; no checkpoint, neural network or solver is loaded."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="completed T1 probe directory")
    parser.add_argument("--candidate", type=Path, required=True, help="completed T2 probe")
    parser.add_argument("--output", type=Path, required=True, help="fresh external directory")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)
    report = build_comparison(args.baseline, args.candidate, args.seed, args.resamples)
    write_comparison(report, args.output)
    print(report["scope"])
    print(f"Temperatures: {report['temperatures']}; all deltas are T2 minus T1.")
    for band, data in report["bands"].items():
        print(f"{band}: {data['common_known']}/{data['sampled']} jointly known positions; "
              f"{data['common_families']} families")
        for contrast, metrics in [*data["models"].items(),
                                  ("difference_of_changes", data["difference_of_changes"])]:
            for metric, estimate in metrics.items():
                print(f"  {contrast} {metric}: delta={estimate['delta_vs_reference']} "
                      f"CI95={estimate['delta_ci95']}")
    print(f"Wrote {args.output / 'report.json'} and {args.output / 'comparisons.csv'}")


if __name__ == "__main__":
    main()
