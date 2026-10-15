"""Cross-temperature pairing, common labels, provenance and interaction uncertainty."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (Path(__file__).parents[1] / "experiments/2026-10-14_09-58-boxes-probe-analysis"
          / "compare_temperatures.py")
SPEC = importlib.util.spec_from_file_location("compare_temperatures", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON)


def write_inputs(directory: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    (directory / "summary.json").write_text(json.dumps(summary))
    with (directory / "positions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def make_run(directory: Path, temperature: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory.mkdir()
    frozen = [dict(rows=5, cols=5, prefix=list(range(60 - undrawn)), source=f"game{i}.npz",
                   source_sha256="0" * 64, family=f"family{i // 2}")
              for i, undrawn in enumerate((35, 36, 37, 38))]
    data = "".join(json.dumps(p) + "\n" for p in frozen).encode()
    (directory / "positions.jsonl").write_bytes(data)
    cases = [f"{model}:s400:lambda0.5" for model in ("reference", "candidate")]
    summary = dict(
        complete=True, sampled=4, source_families=2,
        positions_sha256=hashlib.sha256(data).hexdigest(),
        checkpoint_sha256={"reference": "a" * 64, "candidate": "b" * 64},
        code_sha256={"search_probe.py": "c" * 64}, cpp_sha256="d" * 64, device="cuda",
        arguments=dict(rows=5, cols=5, min_undrawn=35, max_undrawn=38,
                       mixes=["reference", "candidate"], simulations=[400], lambdas=[0.5],
                       positions=4, policy_temperature=temperature, label_budget=20_000_000,
                       label_table_entries=1 << 22, solver=30, solver_budget=50_000,
                       solver_table_entries=1 << 20, c_puct=1.5, leaf_batch_size=16,
                       margin_utility_k=6.0, prove_terminals=1, shape_terminals=1,
                       seed=20261014, cpu=False, output=str(directory), label_cache=None),
        cases=dict.fromkeys(cases, {}), completed_cases=cases)
    values = {1: ((-1, 3, -3, -3), (-3, 1, -5, -3)),
              2: ((1, 3, -1, -1), (1, 3, -1, 1))}
    records = []
    for model_index, (case, children) in enumerate(zip(cases, values[temperature])):
        for index, child in enumerate(children):
            records.append(dict(
                position=index, case=case, mix=case.split(":")[0], simulations=400,
                margin_utility_lambda=0.5, undrawn=35 + index, source_family=f"family{index // 2}",
                root_final=(3, 3, -1, 1)[index], policy_edge=59,
                search_edge=49 + temperature + 2 * model_index,
                policy_final=-5, search_final=child, search_seconds=0.25))
    write_inputs(directory, summary, records)
    return summary, records


class TestTemperatureStatistics:
    def test_known_paired_changes_and_interaction(self, tmp_path: Path) -> None:
        first, second = tmp_path / "t1", tmp_path / "t2"
        make_run(first, 1)
        summary, records = make_run(second, 2)
        write_inputs(second, summary, records[::-1])
        report = COMPARISON.build_comparison(first, second, resamples=2000)
        overall = report["bands"]["all"]
        assert overall["common_known"] == 4
        assert overall["common_families"] == 2
        reference = overall["models"]["reference"]["search_margin_regret"]
        candidate = overall["models"]["candidate"]["search_margin_regret"]
        assert reference["baseline_mean"] == 2.5
        assert reference["candidate_mean"] == 1
        assert reference["delta_vs_reference"] == -1.5
        assert reference["delta_ci95"] == [-2, -1]
        assert candidate["delta_vs_reference"] == -3.5
        interaction = overall["difference_of_changes"]["search_margin_regret"]
        assert interaction["delta_vs_reference"] == -2
        assert interaction["delta_ci95"] == [-2, -2]
        wins = overall["difference_of_changes"]["search_win_preservation"]
        assert wins["delta_vs_reference"] == pytest.approx(1 / 3)
        assert wins["positions"] == 3
        assert report["bands"]["35..36"]["difference_of_changes"][
            "search_margin_regret"]["delta_ci95"] is None
        assert report["temperatures"] == {"baseline": 1, "candidate": 2}

    @pytest.mark.parametrize("missing", ["root_final", "policy_final", "search_final"])
    def test_common_subset_uses_every_case_and_run(self, tmp_path: Path, missing: str) -> None:
        first, second = tmp_path / "t1", tmp_path / "t2"
        make_run(first, 1)
        summary, records = make_run(second, 2)
        records[4][missing] = None
        if missing == "root_final":
            records[0][missing] = None
        write_inputs(second, summary, records)
        report = COMPARISON.build_comparison(first, second, resamples=100)
        overall = report["bands"]["all"]
        assert overall["common_position_indices"] == [1, 2, 3]
        assert overall["excluded"] == 1
        for model in ("reference", "candidate"):
            assert overall["models"][model]["search_margin_regret"]["positions"] == 3
        assert overall["known"]["candidate"]["candidate"][missing] == 3

    def test_no_labels_produces_null_not_zero(self, tmp_path: Path) -> None:
        first, second = tmp_path / "t1", tmp_path / "t2"
        make_run(first, 1)
        summary, records = make_run(second, 2)
        for row in records:
            row["root_final"] = None
        write_inputs(second, summary, records)
        report = COMPARISON.build_comparison(first, second, resamples=100)
        metric = report["bands"]["all"]["difference_of_changes"]["search_margin_regret"]
        assert metric["delta_vs_reference"] is metric["delta_ci95"] is None
        COMPARISON.write_comparison(report, tmp_path / "report")

    def test_no_winning_roots_produces_null_preservation(self, tmp_path: Path) -> None:
        first, second = tmp_path / "t1", tmp_path / "t2"
        for path, temperature in ((first, 1), (second, 2)):
            summary, records = make_run(path, temperature)
            for row in records:
                row.update(root_final=-1, search_final=-3)
            write_inputs(path, summary, records)
        report = COMPARISON.build_comparison(first, second, resamples=100)
        metric = report["bands"]["all"]["difference_of_changes"]["search_win_preservation"]
        assert metric["delta_vs_reference"] is metric["delta_ci95"] is None


class TestTemperatureValidation:
    @pytest.mark.parametrize("key", ["checkpoint_sha256", "code_sha256", "cpp_sha256", "device"])
    def test_mismatched_provenance(self, tmp_path: Path, key: str) -> None:
        first, second = tmp_path / "t1", tmp_path / "t2"
        make_run(first, 1)
        summary, records = make_run(second, 2)
        if isinstance(summary[key], dict):
            summary[key][next(iter(summary[key]))] = "e" * 64
        else:
            summary[key] = "cpu" if key == "device" else "e" * 64
        write_inputs(second, summary, records)
        with pytest.raises(ValueError, match=key):
            COMPARISON.build_comparison(first, second)

    @pytest.mark.parametrize("key", ["solver_budget", "c_puct", "seed", "future_setting"])
    def test_changed_settings_rejected(self, tmp_path: Path, key: str) -> None:
        first, second = tmp_path / "t1", tmp_path / "t2"
        make_run(first, 1)
        summary, records = make_run(second, 2)
        summary["arguments"][key] = 99
        write_inputs(second, summary, records)
        with pytest.raises(ValueError, match="settings"):
            COMPARISON.build_comparison(first, second)

    @pytest.mark.parametrize("value", [0, -1, float("nan"), 1])
    def test_invalid_or_unchanged_temperature(self, tmp_path: Path, value: float) -> None:
        first, second = tmp_path / "t1", tmp_path / "t2"
        make_run(first, 1)
        summary, records = make_run(second, 2)
        summary["arguments"]["policy_temperature"] = value
        write_inputs(second, summary, records)
        with pytest.raises(ValueError, match="temperature"):
            COMPARISON.build_comparison(first, second)

    def test_different_frozen_family_rejected_even_with_updated_hash(self, tmp_path: Path) -> None:
        first, second = tmp_path / "t1", tmp_path / "t2"
        make_run(first, 1)
        summary, records = make_run(second, 2)
        frozen = second / "positions.jsonl"
        frozen.write_text(frozen.read_text().replace("family0", "different"))
        summary["positions_sha256"] = hashlib.sha256(frozen.read_bytes()).hexdigest()
        for row in records:
            if row["source_family"] == "family0":
                row["source_family"] = "different"
        write_inputs(second, summary, records)
        with pytest.raises(ValueError, match="Frozen positions"):
            COMPARISON.build_comparison(first, second)

    @pytest.mark.parametrize("kind", ["root", "child"])
    def test_conflicting_exact_values_across_runs(self, tmp_path: Path, kind: str) -> None:
        first, second = tmp_path / "t1", tmp_path / "t2"
        make_run(first, 1)
        summary, records = make_run(second, 2)
        if kind == "root":
            records[0]["root_final"] = records[4]["root_final"] = 5
        else:
            for row in records:
                row["policy_final"] = -3
        write_inputs(second, summary, records)
        with pytest.raises(ValueError, match="disagree"):
            COMPARISON.build_comparison(first, second)

    def test_cli_readonly_fresh_output_and_lightweight_import(self, tmp_path: Path) -> None:
        first, second, output = tmp_path / "t1", tmp_path / "t2", tmp_path / "report"
        make_run(first, 1)
        make_run(second, 2)
        before = {p: p.read_bytes() for directory in (first, second) for p in directory.iterdir()}
        COMPARISON.main(["--baseline", str(first), "--candidate", str(second),
                         "--output", str(output), "--resamples", "100"])
        report = json.loads((output / "report.json").read_text())
        assert before == {p: p.read_bytes() for p in before}
        with (output / "comparisons.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 18
        for forbidden in (output, first / "report", second / "report"):
            with pytest.raises(ValueError, match="fresh.*outside"):
                COMPARISON.write_comparison(report, forbidden)
        subprocess.run([sys.executable, "-c", "import runpy, sys; runpy.run_path(sys.argv[1]); "
                        "assert 'torch' not in sys.modules; "
                        "assert 'alpha_go_cpp' not in sys.modules", str(SCRIPT)], check=True)
