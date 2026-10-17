# Retrying budget-limited solver probes

The 48 frozen 5x5 roots from the preceding search-target audit were rerun with
the same checkpoint, 400 simulations, solver34/50k, proof/shaping settings and
fresh solver table per root. Scratch instrumentation now records each bounded
call's edge mask, node count and whether it exhausted its budget. Search code
and budgets were not changed.

All 48 chosen moves, complete visit maps, selection policies, root proof flags,
bounded/unbounded node counts and abort counts matched the previous audit.
`data/summary.json` is reproducible with `analyze.py`; the per-call trace is
retained in `data/solver34.jsonl`.

| Measurement | Result |
|---|---:|
| Bounded calls | 7,377 |
| Nodes | 191,639,804 |
| Distinct masks, grouped separately within each root | 5,572 |
| Masks whose first call exhausted its budget | 2,223 |
| Retries of an as-yet-unsolved mask after budget exhaustion | 686 |
| Nodes in those retries | 28,453,588 (14.85% of all bounded nodes) |
| Previously timed-out masks later solved exactly | 203 |

Grouping by symmetry-canonical masks gives exactly the same counts here.
The transposition table retains partial bounds between calls. A timeout is
therefore not a permanent fact about a position: **203 masks that initially
failed were proved on later attempts** with the same 50k per-call budget.
This trace cannot attribute each recovery to a specific table entry or rule.

Permanently caching timeouts would prevent those later proofs. The trace does
not establish how many lost proofs would change the selected move or game
outcome, but it rules out treating timeout caching as a semantics-preserving
optimization. Also, retries before first success account for only 14.85% of
bounded nodes in this sample; they do not explain most solver work. A smarter
allocation or structural trigger would need its own strength/cost experiment.

Prefer the independent exact-equivalent canonicalization speed experiment
before changing which solver calls are attempted. No timeout-cache or budget
change is included here. These are equal-count development bands, not the
frequency distribution of all positions in the running self-play loop.

## Reproduce

From the repo with the reference checkpoint identified by the preceding
audit's `data/provenance.json`:

```bash
PREV=experiments/2026-10-17_10-17-boxes-search-target-audit
EXP=experiments/2026-10-17_10-33-boxes-probe-retries
uv run python "$PREV/instrument.py" --trace --output work/solver-trace-extension
PYTHONPATH=src:work/solver-trace-extension OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python "$PREV/bench.py" --positions "$PREV/data/positions.jsonl" \
    --checkpoint "$REFERENCE" --solver 34 --output work/probe-trace.jsonl
uv run python "$EXP/analyze.py" "$EXP/data/solver34.jsonl" \
  --baseline "$PREV/data/solver34.jsonl"
```

The trace is diagnostic instrumentation in a separate extension, not a runtime
logging feature. The new scripts pass Ruff. Full baseline parity is asserted
by the analysis script; wall times are deliberately excluded from parity.
