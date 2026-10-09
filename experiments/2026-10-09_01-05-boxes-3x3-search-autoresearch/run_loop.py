"""The autoresearch keep / discard loop over eval_search.py's search hyperparameters.

Each candidate is the current best VAR plus one change (one variable at a time, then the
near-misses combined). A run that beats the running best on `search_optimal` is kept and
becomes the base for later candidates; anything else is discarded, which is the skill's
"revert" without touching git. Every run appends to results.tsv (metric, memory, status,
description) and runs.jsonl (the full VAR and ===RESULT=== line), so any run can be repeated
with `eval_search.py --var '<json>'`. Ends with analyze_runs.py.

  uv run run_loop.py            # ~1 min per run on CPU
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results.tsv"
RUNS = EXP_DIR / "runs.jsonl"
METRIC = "search_optimal"

# (description, overrides relative to the current best); the first is the baseline.
CANDIDATES: list[tuple[str, dict[str, object]]] = [
    ("baseline: sims=100 c_puct=1.5 leaf=16 T=1 lambda=0 solver=0 merge=off", {}),
    ("c_puct=1.0", {"c_puct": 1.0}),
    ("c_puct=2.5", {"c_puct": 2.5}),
    ("c_puct=0.75", {"c_puct": 0.75}),
    ("leaf_batch_size=8 (more sequential, less virtual loss)", {"leaf_batch_size": 8}),
    ("leaf_batch_size=4", {"leaf_batch_size": 4}),
    ("leaf_batch_size=32", {"leaf_batch_size": 32}),
    ("policy_temperature=0.7 (sharper priors)", {"policy_temperature": 0.7}),
    ("policy_temperature=1.5 (flatter priors)", {"policy_temperature": 1.5}),
    ("margin_utility_lambda=0.25 (k=6), PLAN 6.4", {"margin_utility_lambda": 0.25}),
    ("margin_utility_lambda=0.5 (k=6)", {"margin_utility_lambda": 0.5}),
    ("margin_utility_lambda=0.25 k=3", {"margin_utility_lambda": 0.25, "margin_utility_k": 3.0}),
    ("merge_equivalent=True (one action per chain/loop)", {"merge_equivalent": True}),
    ("solver_max_undrawn=6 (exact leaves below the sampled band)", {"solver_max_undrawn": 6}),
    ("solver_max_undrawn=8", {"solver_max_undrawn": 8}),
    ("solver_max_undrawn=10", {"solver_max_undrawn": 10}),
    ("sims=200 (double budget; cost in sec_per_move)", {"num_simulations": 200}),
    ("sims=50 (half budget)", {"num_simulations": 50}),
]


def run(var: dict[str, object]) -> dict[str, object] | None:
    cmd = [sys.executable, str(EXP_DIR / "eval_search.py"), "--var", json.dumps(var)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    lines = proc.stdout.splitlines()
    if "===RESULT===" not in lines:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        return None
    return dict(json.loads(lines[lines.index("===RESULT===") + 1]))


def main() -> None:
    if not RESULTS.exists():
        RESULTS.write_text(f"{METRIC}\tmemory_gb\tstatus\tdescription\n")
    best_var: dict[str, object] = {}
    best = -1.0
    for description, change in CANDIDATES:
        var = {**best_var, **change}
        t0 = time.time()
        result = run(var)
        if result is None:
            status, value = "crash", 0.0
        else:
            value = float(result[METRIC])
            status = "keep" if value > best else "discard"
        if status == "keep":
            best, best_var = value, var
        with RESULTS.open("a") as f:
            f.write(f"{value:.4f}\t0.0\t{status}\t{description}\n")
        with RUNS.open("a") as f:
            f.write(json.dumps({"description": description, "var": var, "status": status,
                                "result": result, "seconds": round(time.time() - t0)}) + "\n")
        extra = "" if result is None else (f" root_q_sign={result['root_q_sign']} "
                                           f"sec/move={result['sec_per_move']}")
        print(f"{status:8s} {value:.4f}  {description}{extra}", flush=True)
    print(f"best {METRIC} {best:.4f} with {json.dumps(best_var)}")
    subprocess.run([sys.executable, str(EXP_DIR / "analyze_runs.py")], check=False)


if __name__ == "__main__":
    main()
