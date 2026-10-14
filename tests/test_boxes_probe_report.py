"""Pair alignment, missing labels and family-clustered uncertainty in probe reports."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alpha_go.boxes.probe_report import build_report, load_probe, main, paired_metric, write_report

REFERENCE = "reference:s400:lambda0.5"
CANDIDATE = "candidate:s400:lambda0.5"


def probe_files(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Write two cases with a known paired gain and two positions per family."""
    directory.mkdir()
    frozen = [dict(rows=5, cols=5, prefix=list(range(60 - undrawn)), source=f"game{i}.npz",
                   source_sha256="0" * 64, family=f"family{i // 2}")
              for i, undrawn in enumerate((35, 36, 37, 38))]
    data = "".join(json.dumps(p) + "\n" for p in frozen).encode()
    (directory / "positions.jsonl").write_bytes(data)
    summary = dict(complete=True, sampled=4, source_families=2,
                   positions_sha256=hashlib.sha256(data).hexdigest(),
                   arguments=dict(rows=5, cols=5, min_undrawn=35, max_undrawn=38,
                                  mixes=["reference", "candidate"], simulations=[400],
                                  lambdas=[0.5]),
                   cases={REFERENCE: {}, CANDIDATE: {}},
                   completed_cases=[REFERENCE, CANDIDATE])
    records: list[dict[str, Any]] = []
    for case, values in ((REFERENCE, (-1, 3, -3, -3)), (CANDIDATE, (1, 3, -1, -1))):
        for index, child in enumerate(values):
            records.append(dict(position=index, case=case, mix=case.split(":")[0],
                                simulations=400, margin_utility_lambda=0.5, undrawn=35 + index,
                                source_family=f"family{index // 2}",
                                root_final=(3, 3, -1, 1)[index], policy_edge=59, search_edge=58,
                                policy_final=child, search_final=child, search_seconds=0.25))
    save_inputs(directory, summary, records)
    return summary, records


def save_inputs(directory: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    (directory / "summary.json").write_text(json.dumps(summary))
    with (directory / "positions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


class TestPairedReports:
    def test_known_deltas_and_row_order_do_not_change_pairing(self, tmp_path: Path) -> None:
        directory = tmp_path / "probe"
        summary, records = probe_files(directory)
        report = build_report(directory, resamples=2000)
        random.Random(51).shuffle(records)
        save_inputs(directory, summary, records)
        shuffled = build_report(directory, resamples=2000)
        assert report["bands"] == shuffled["bands"]
        overall = report["bands"]["all"]
        assert overall["sampled"] == overall["common_known"] == 4
        assert overall["common_families"] == 2
        metrics = overall["cases"][CANDIDATE]
        assert metrics["search_margin_regret"]["mean"] == 1.0
        assert metrics["search_margin_regret"]["delta_vs_reference"] == -1.5
        assert metrics["search_margin_regret"]["delta_ci95"] == [-2.0, -1.0]
        assert metrics["search_win_preservation"]["mean"] == 2 / 3
        assert metrics["search_win_preservation"]["delta_vs_reference"] == pytest.approx(1 / 3)
        assert report["bands"]["35..36"]["cases"][CANDIDATE][
            "search_margin_regret"]["delta_ci95"] is None

    @pytest.mark.parametrize("missing", ["policy_final", "search_final"])
    def test_unknown_child_excludes_position_from_all_cases(self, tmp_path: Path,
                                                          missing: str) -> None:
        directory = tmp_path / "probe"
        summary, records = probe_files(directory)
        records[4][missing] = None
        save_inputs(directory, summary, records)
        report = build_report(directory, resamples=100)
        band = report["bands"]["all"]
        assert band["common_position_indices"] == [1, 2, 3]
        assert band["excluded"] == 1
        delta = band["cases"][CANDIDATE]["search_margin_regret"]["delta_vs_reference"]
        assert delta == pytest.approx(-4 / 3)
        assert band["cases"][REFERENCE]["search_margin_regret"]["positions"] == 3
        assert band["cases"][CANDIDATE]["known"][missing] == 3

    def test_unknown_roots_and_no_known_positions_are_not_zeros(self, tmp_path: Path) -> None:
        directory = tmp_path / "probe"
        summary, records = probe_files(directory)
        for record in records:
            record["root_final"] = None
        save_inputs(directory, summary, records)
        report = build_report(directory, resamples=100)
        overall = report["bands"]["all"]
        assert overall["common_known"] == 0
        assert overall["excluded"] == 4
        metric = overall["cases"][CANDIDATE]["search_margin_regret"]
        assert metric["mean"] is metric["delta_vs_reference"] is metric["delta_ci95"] is None
        write_report(report, tmp_path / "output")  # strict JSON rejects accidental NaN

    def test_zero_winning_roots_have_no_preservation_estimate(self, tmp_path: Path) -> None:
        directory = tmp_path / "probe"
        summary, records = probe_files(directory)
        for record in records:
            record.update(root_final=-1, policy_final=-1, search_final=-3)
        save_inputs(directory, summary, records)
        report = build_report(directory, resamples=100)
        metrics = report["bands"]["all"]["cases"][CANDIDATE]
        assert metrics["search_margin_regret"]["mean"] == 2.0
        assert metrics["search_win_preservation"]["mean"] is None
        assert metrics["search_win_preservation"]["delta_ci95"] is None

    def test_position_weighted_family_resampling_and_shared_draws(self) -> None:
        # The three positions from family 0 always move together. The point estimate
        # weights all four positions equally, not the two family means equally.
        values = np.array([[4, 0], [4, 0], [4, 0], [4, 8]], dtype=np.float64)
        family_index = np.array([0, 0, 0, 1], dtype=np.int64)
        draws = np.array([[2, 0], [1, 1], [0, 2]], dtype=np.int64)
        result = paired_metric(values, np.ones(4, dtype=np.bool_), family_index, draws, 0)
        assert result[1]["delta_vs_reference"] == -2
        assert result[1]["delta_ci95"] == pytest.approx([-3.9, 3.7])
        assert result[0]["delta_ci95"] == [0, 0]
        # A metric supported by only one family has no CI, even with four positions
        # in the complete sample and many requested bootstrap resamples.
        result = paired_metric(values, np.array([True, True, True, False]),
                               family_index, draws, 0)
        assert result[1]["positions"] == 3
        assert result[1]["families"] == 1
        assert result[1]["delta_ci95"] is None

    def test_bootstrap_replicates_without_winning_roots_are_excluded(self) -> None:
        result = paired_metric(np.array([[1, 0], [1, 1], [0, 0]], dtype=np.float64),
                               np.array([True, True, False]), np.array([0, 1, 2]),
                               np.array([[1, 1, 1], [0, 0, 3], [2, 1, 0]]), 0)
        assert result[1]["bootstrap_replicates"] == 2
        assert result[1]["delta_ci95"] is not None

    def test_no_valid_bootstrap_replicate_does_not_invent_interval(self) -> None:
        result = paired_metric(np.ones((3, 2)), np.array([True, True, False]),
                               np.array([0, 1, 2]), np.array([[0, 0, 3]]), 0)
        assert result[1]["mean"] == 1
        assert result[1]["families"] == 2
        assert result[1]["delta_ci95"] is None
        assert result[1]["bootstrap_replicates"] == 0
        assert "no bootstrap replicate" in result[1]["ci_unavailable"]


class TestValidationAndCli:
    @pytest.mark.parametrize("problem", ["duplicate", "missing", "root", "family", "undrawn",
                                         "child", "edge", "nan", "parity", "case"])
    def test_corrupt_rows_are_rejected(self, tmp_path: Path, problem: str) -> None:
        directory = tmp_path / "probe"
        summary, records = probe_files(directory)
        if problem == "duplicate":
            records.append(records[0].copy())
        elif problem == "missing":
            records.pop()
        else:
            key, value = {
                "root": ("root_final", 5), "family": ("source_family", "wrong"),
                "undrawn": ("undrawn", 34), "child": ("search_final", 5),
                "edge": ("search_edge", 0), "nan": ("search_seconds", "nan"),
                "parity": ("search_final", 0), "case": ("simulations", 800),
            }[problem]
            records[-1][key] = value
        save_inputs(directory, summary, records)
        with pytest.raises(ValueError):
            load_probe(directory)

    @pytest.mark.parametrize("problem", ["incomplete", "cases", "sampled", "families", "hash"])
    def test_bad_summary_is_rejected(self, tmp_path: Path, problem: str) -> None:
        directory = tmp_path / "probe"
        summary, records = probe_files(directory)
        key, value = {"incomplete": ("complete", False), "cases": ("completed_cases", [REFERENCE]),
                      "sampled": ("sampled", 5), "families": ("source_families", 4),
                      "hash": ("positions_sha256", "0" * 64)}[problem]
        summary[key] = value
        save_inputs(directory, summary, records)
        with pytest.raises(ValueError):
            load_probe(directory)

    def test_ambiguous_reference_requires_explicit_case(self, tmp_path: Path) -> None:
        directory = tmp_path / "probe"
        summary, records = probe_files(directory)
        other = "reference:s800:lambda0.5"
        for row in records[4:]:
            row.update(case=other, mix="reference", simulations=800)
        summary["arguments"].update(mixes=["reference"], simulations=[400, 800])
        summary["completed_cases"] = [REFERENCE, other]
        summary["cases"] = {REFERENCE: {}, other: {}}
        save_inputs(directory, summary, records)
        with pytest.raises(ValueError, match="reference-case"):
            build_report(directory)
        report = build_report(directory, reference=REFERENCE, resamples=10)
        assert report["reference_case"] == REFERENCE

    def test_cli_is_readonly_and_writes_fresh_output(self, tmp_path: Path,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
        directory, output = tmp_path / "probe", tmp_path / "report"
        probe_files(directory)
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        main(["--input", str(directory), "--output", str(output), "--resamples", "100"])
        report = json.loads((output / "report.json").read_text())
        assert report["bootstrap"]["seed"] == 20261014
        with (output / "comparisons.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 3 * 2 * 4
        assert "2/2 families" in capsys.readouterr().out
        assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
        with pytest.raises(ValueError, match="fresh"):
            write_report(report, output)
        with pytest.raises(ValueError, match="outside"):
            write_report(report, directory / "report")

    def test_standalone_script_needs_neither_torch_nor_cpp(self) -> None:
        # `python -m alpha_go...` imports the existing package's eager torch/C++
        # exports. This read-only script can instead run directly with just NumPy.
        script = Path(__file__).parents[1] / "src/alpha_go/boxes/probe_report.py"
        subprocess.run([sys.executable, "-c", "import runpy, sys; runpy.run_path(sys.argv[1]); "
                        "assert 'torch' not in sys.modules; "
                        "assert 'alpha_go_cpp' not in sys.modules", str(script)], check=True)
