"""The autoresearch keep / discard loop over eval_search.py's search hyperparameters.

Each candidate is the current best VAR plus one change (one variable at a time, then the
near-misses combined). A run that beats the running best on `search_optimal` is kept and
becomes the base for later candidates; a tie is kept only when it is cheaper (`sec_per_move`,
the skill's "simpler at equal metric wins"); anything else is discarded, which is the
skill's "revert" without touching git. Every run appends to results.tsv (metric, memory,
status, description) and runs.jsonl (the full VAR and ===RESULT=== line), so any run can be
repeated with `eval_search.py --var '<json>'`. Re-running resumes: candidates already in
runs.jsonl are skipped and the running best is restored from its last kept run, so later
rounds are appended to CANDIDATES. Ends with analyze_runs.py.

  uv run run_loop.py            # ~1-3 min per run on CPU
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
    # Round 2: push the winners further, and re-test the cheap near-ties on top of the best.
    ("margin_utility_lambda=0.75 (k=6)", {"margin_utility_lambda": 0.75}),
    ("margin_utility_lambda=1.0 (k=6)", {"margin_utility_lambda": 1.0}),
    ("margin_utility_k=10 at the best lambda", {"margin_utility_k": 10.0}),
    ("policy_temperature=0.5", {"policy_temperature": 0.5}),
    ("solver_max_undrawn=10 on top of the best (tied round 1 at a third of the time)",
     {"solver_max_undrawn": 10}),
    ("leaf_batch_size=16 on top of the best (is leaf=4's gain worth 2x the time?)",
     {"leaf_batch_size": 16}),
    ("c_puct=1.5 on top of the best (was c_puct=2.5's gain real?)", {"c_puct": 1.5}),
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
    past = [json.loads(line) for line in RUNS.read_text().splitlines()] if RUNS.exists() else []
    done = {r["description"] for r in past}
    kept = [r for r in past if r["status"] == "keep"]
    best_var: dict[str, object] = kept[-1]["var"] if kept else {}
    best = float(kept[-1]["result"][METRIC]) if kept else -1.0
    best_sec = float(kept[-1]["result"]["sec_per_move"]) if kept else float("inf")
    for description, change in CANDIDATES:
        if description in done:
            continue
        var = {**best_var, **change}
        t0 = time.time()
        result = run(var)
        if result is None:
            status, value, sec = "crash", 0.0, float("inf")
        else:
            value, sec = float(result[METRIC]), float(result["sec_per_move"])
            status = "keep" if value > best or (value == best and sec < best_sec) else "discard"
        if status == "keep":
            best, best_var, best_sec = value, var, sec
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
