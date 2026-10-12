"""Continue a running pilot through the requested retry, sized run, verification and archive."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

from run_cpu_jobs import EXP, REPORT, record, run_one

STATE = REPORT / "data/pipeline-status.json"
with STATE.open("x") as stream:
    json.dump({"stage": "awaiting_pilot20"}, stream)


def stage(name: str, **fields) -> None:
    """Record the current pipeline stage atomically."""
    value = {"stage": name, **fields}
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(STATE)
    record({"event": "pipeline_stage", **value})
    print(json.dumps(value), flush=True)


def checked(command: list[str], name: str) -> None:
    """Stop this job on any failed subprocess."""
    code = run_one(command, name)
    if code:
        stage("failed", failed_command=name, exit_code=code)
        raise SystemExit(code)


pilot_path = REPORT / "data/pilot20-status.json"
while True:
    pilot = json.loads(pilot_path.read_text())
    if not pilot["running_shards"]:
        break
    time.sleep(5)
if pilot["failed"]:
    stage("failed", failed_command="pilot20")
    raise SystemExit(1)
assert len(pilot["results"]) == pilot["shards"]
sampled = sum(row["sampled"] for row in pilot["results"])
skipped = sum(row["skipped"] for row in pilot["results"])
retry = skipped * 5 > sampled
if retry:
    source = Path(pilot["output"])
    preserved = source.with_name(source.name + "-20m")
    assert not preserved.exists()
    source.rename(preserved)
    record({"event": "preserve_pilot", "from": str(source), "to": str(preserved)})
    pilot["output"] = str(preserved)
    pilot_path.write_text(json.dumps(pilot, indent=2) + "\n")
stage("verify_pilot20", sampled=sampled, skipped=skipped, retry_at_40m=retry)
checked([sys.executable, str(REPORT / "verify_labels.py"), str(pilot_path)], "verify-pilot20")
phase = "pilot20"
if retry:
    stage("pilot40")
    checked([sys.executable, str(REPORT / "run_cpu_jobs.py"), "label", "--phase", "pilot40",
             "--positions", str(pilot["requested"]), "--node-budget", "40000000", "--shards",
             str(pilot["shards"]), "--seed", "0", "--save-name", "5x5-oracle-31-36-pilot"],
            "pilot40-driver")
    stage("verify_pilot40")
    checked([sys.executable, str(REPORT / "verify_labels.py"),
             str(REPORT / "data/pilot40-status.json")], "verify-pilot40")
    phase = "pilot40"
checked([sys.executable, str(REPORT / "plan_full.py"), phase], "size-full")
plan = json.loads((REPORT / "data/full-plan.json").read_text())
stage("full", requested_positions=plan["requested_positions"], node_budget=plan["node_budget"])
checked([sys.executable, str(REPORT / "run_cpu_jobs.py"), "label", "--phase", "full",
         "--positions", str(plan["requested_positions"]), "--node-budget", str(plan["node_budget"]),
         "--shards", str(plan["shards"]), "--seed", "1", "--save-name", "5x5-oracle-31-36"],
        "full-driver")
stage("verify_full")
checked([sys.executable, str(REPORT / "verify_labels.py"),
         str(REPORT / "data/full-status.json")], "verify-full")
summary = json.loads((REPORT / "data/full-verification.json").read_text())
output = Path(summary["output"])
archive = output.with_name(output.name + ".tar.gz")
assert not archive.exists()
stage("archive", path=str(archive))
checked(["tar", "czf", str(archive), "-C", str(output.parent), output.name], "archive-full")
with (REPORT / "data/full-files.csv").open() as stream:
    expected = {row["file"]: row["sha256"] for row in csv.DictReader(stream)}
found = set()
with tarfile.open(archive, "r:gz") as tar:
    for member in tar:
        if member.isdir():
            assert member.name == output.name
            continue
        assert member.isfile()
        path = Path(member.name)
        assert path.parent == Path(output.name) and path.name in expected
        assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == expected[path.name]
        found.add(path.name)
assert found == set(expected)
archive_record = {"path": str(archive), "size_bytes": archive.stat().st_size,
                  "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "npz_files": len(found), "positions": summary["positions"],
                  "archive_verified_against_file_hashes": True}
(REPORT / "data/archive.json").write_text(json.dumps(archive_record, indent=2) + "\n")
inputs = json.loads((REPORT / "data/input-manifest.json").read_text())
data_root = Path(inputs["root"])
after = {}
for tag in ["5x5-chains", "5x5-mid"]:
    paths = sorted((data_root / EXP / tag).rglob("*.npz"))
    assert len(paths) == inputs["counts"][tag]
    digest = hashlib.sha256()
    for path in paths:
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update((str(path.relative_to(data_root)) + " " + checksum + "\n").encode())
    after[tag] = digest.hexdigest()
    assert after[tag] == inputs["aggregate_sha256"][tag], f"Input set changed: {tag}"
(REPORT / "data/input-integrity.json").write_text(json.dumps(
    {"verification_passed": True, "counts": inputs["counts"], "aggregate_sha256_after": after},
    indent=2) + "\n")
checked([sys.executable, str(REPORT / "write_report.py")], "write-final-report")
stage("complete", **archive_record)
