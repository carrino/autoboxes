"""Build isolated baseline/candidate extensions, alternate full search traces and CPU timing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import torch


def build(cpp: Path, output: Path) -> Path:
    """Compile the same module with each solver implementation, one compiler process."""
    output.mkdir(parents=True, exist_ok=True)
    sources = [cpp / "bindings/bindings.cpp", cpp / "go/go_game.cpp", cpp / "mcts/mcts.cpp",
               *sorted((cpp / "boxes").glob("*.cpp"))]
    sources = [path for path in sources if not path.name.endswith("_test.cpp")]
    extension = output / ("alpha_go_cpp" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run([
        "g++", "-O3", "-DNDEBUG", "-std=c++17", "-shared", "-fPIC", "-pthread",
        "-I" + sysconfig.get_path("include"),
        "-I" + str(Path(torch.__file__).parent / "include"), "-I" + str(cpp),
        *map(str, sources), "-o", str(extension),
    ], check=True)
    return extension


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="ed3fe2c")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    root = folder.parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = output / "baseline-source"
    shutil.copytree(root / "src/alpha_go/cpp", baseline, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("build"))
    for name in ("boxes_solver.cpp", "boxes_solver.h"):
        data = subprocess.check_output(
            ["git", "show", f"{args.base}:src/alpha_go/cpp/boxes/{name}"], cwd=root)
        (baseline / "boxes" / name).write_bytes(data)
    extensions = {
        name: build(source, output / name)
        for name, source in (("baseline", baseline), ("candidate", root / "src/alpha_go/cpp"))
    }
    provenance = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  "base": args.base,
                  "candidate_solver_sha256": {
                      name: hashlib.sha256((root / "src/alpha_go/cpp/boxes" / name).read_bytes())
                      .hexdigest() for name in ("boxes_solver.cpp", "boxes_solver.h")},
                  "extension_sha256": {
                      name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for name, path in extensions.items()}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    expected = None
    for repeat in range(args.repeats):
        names = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for name in names:
            environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               PYTHONPATH=f"{root / 'src'}:{extensions[name].parent}")
            destination = output / f"mcts-{repeat}-{name}.jsonl"
            with (output / f"mcts-{repeat}-{name}.log").open("w") as log:
                subprocess.run([
                    sys.executable, "-u", str(folder / "mcts.py"),
                    "--checkpoint", str(args.checkpoint.resolve()), "--output", str(destination),
                ], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = [json.loads(line) for line in destination.read_text().splitlines()]
            outcomes = [{k: v for k, v in row.items() if k != "cpu_seconds"} for row in rows]
            if expected is None:
                expected = outcomes
            assert outcomes == expected, (repeat, name)
            print(repeat, name, sum(row["cpu_seconds"] for row in rows), flush=True)


if __name__ == "__main__":
    main()
