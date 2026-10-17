"""Build baseline and current native solvers, check exact parity, time interleaved runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="ed3fe2c")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    root = folder.parents[1]
    positions = root / "experiments/2026-10-17_10-17-boxes-search-target-audit/data/positions.jsonl"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in positions.read_text().splitlines()]
    masks = output / "masks.txt"
    masks.write_text("".join(str(sum(1 << e for e in p["prefix"])) + "\n" for p in records))
    baseline = output / "baseline"
    shutil.copytree(root / "src/alpha_go/cpp/boxes", baseline / "boxes", dirs_exist_ok=True)
    for name in ("boxes_solver.cpp", "boxes_solver.h"):
        text = subprocess.check_output(
            ["git", "show", f"{args.base}:src/alpha_go/cpp/boxes/{name}"], cwd=root
        )
        (baseline / "boxes" / name).write_bytes(text)
    current = root / "src/alpha_go/cpp"
    for name, src in (("baseline", baseline), ("candidate", current)):
        subprocess.run([
            "g++", "-O3", "-DNDEBUG", "-std=c++17", "-I" + str(src), str(folder / "bench.cpp"),
            *(str(src / "boxes" / f"boxes_{part}.cpp") for part in ("game", "chains", "solver")),
            "-o", str(output / name / "bench" if name == "baseline" else output / "candidate"),
        ], check=True)
    provenance = {
        "base": subprocess.check_output(
            ["git", "rev-parse", args.base], cwd=root, text=True).strip(),
        "source_positions_sha256": hashlib.sha256(positions.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "clock": "std::clock process CPU time, one runtime thread",
        "flags": "-O3 -DNDEBUG -std=c++17",
        "repeats": args.repeats,
        "candidate_solver_sha256": {
            name: hashlib.sha256((current / "boxes" / name).read_bytes()).hexdigest()
            for name in ("boxes_solver.cpp", "boxes_solver.h")},
        "bench_sha256": hashlib.sha256((folder / "bench.cpp").read_bytes()).hexdigest(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for mode in ("canonical", "construct", "solve"):
        expected = None
        for repeat in range(args.repeats):
            names = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
            for name in names:
                binary = output / "baseline/bench" if name == "baseline" else output / "candidate"
                result = subprocess.check_output([str(binary), str(masks), mode], text=True)
                (output / f"{mode}-{repeat}-{name}.jsonl").write_text(result)
                rows = [json.loads(line) for line in result.splitlines()]
                outcomes = [{k: v for k, v in row.items() if k != "cpu_seconds"} for row in rows]
                if expected is None:
                    expected = outcomes
                assert outcomes == expected, (mode, repeat, name)
                print(mode, repeat, name, len(rows),
                      sum(r["cpu_seconds"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
