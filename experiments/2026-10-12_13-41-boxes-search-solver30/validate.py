"""Freeze development settings, then evaluate both checkpoints on untouched mid games."""
from __future__ import annotations

import importlib.util
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOOP = HERE.parent / "2026-10-08_14-42-boxes-3x3-loop"


def main() -> None:
    """Validate the frozen winner and original settings; never select using these results."""
    decision = json.loads((HERE / "data" / "round3.json").read_text())
    winner = json.loads((HERE / "data" / f"{decision['retained']}.json").read_text())
    baseline = json.loads((HERE / "data" / "dev-baseline.json").read_text())
    frozen = {"selected_run": winner["name"], "flags": winner["search_flags"],
              "development_correct": winner["correct"], "development_n": 300,
              "rule": "Fixed before reading or scoring 5x5-mid; no subsequent tuning."}
    path = HERE / "data" / "frozen.json"
    if path.exists():
        assert json.loads(path.read_text()) == frozen
    else:
        path.write_text(json.dumps(frozen, indent=2) + "\n")
    print("Frozen settings: " + json.dumps(frozen), flush=True)
    positions = HERE / "data" / "heldout.json"
    subprocess.run([sys.executable, str(HERE / "bench.py"), "sample", "--tag", "5x5-mid",
                    "--output", str(positions)], check=True)

    # Independently check the saved sampling/order against the task's original evaluator.
    spec = importlib.util.spec_from_file_location("task4_oracle", LOOP / "oracle_eval.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = Path(os.environ["GAME_DATA_DIR"]) / "experiments" / LOOP.name / "5x5-mid"
    boards = module.late_positions(source, 5, 5, 29, 34, 300, random.Random(0))
    assert [[b.edges(), b.to_play(), b.margin()] for b in boards] == json.loads(
        positions.read_text())["keys"]
    print("Held-out sample matches oracle_eval exactly (300/300, same order).", flush=True)

    def evaluate(item: tuple[str, int]) -> None:
        label, iteration = item
        flags = (winner if label == "best" else baseline)["search_flags"]
        name = f"test-{label}{iteration}"
        command = [sys.executable, str(HERE / "bench.py"), "evaluate", "--positions",
                   str(positions), "--name", name, "--iteration", str(iteration)]
        for key, value in flags.items():
            command.extend(["--" + key, str(int(value) if isinstance(value, bool) else value)])
        with (HERE / "logs" / f"{name}.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = json.loads((HERE / "data" / f"{name}.json").read_text())
        print(f"{name}: {result['correct']}/300 in {result['seconds']:.1f}s", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(evaluate, [(label, it) for it in (54, 15) for label in ("baseline", "best")]))


if __name__ == "__main__":
    main()
