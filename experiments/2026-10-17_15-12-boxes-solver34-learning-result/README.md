# Solver34 overnight: iter15 improves over iter0

The final checkpoint from `5x5-solver34-8pt6ai3d` won **126–74** against its
frozen iter0 in the planned 200-game final arena. Both networks used solver34/50k,
400 simulations, identical search settings and paired four-edge openings. This
supports retaining **iter15 as the next starting incumbent** and comparing the
next final learner against a frozen copy of iter15.

| Final measure | Recomputed result |
| --- | ---: |
| Game wins | 126 / 200 (63%) |
| Opening pairs: sweeps / splits / losses | 40 / 46 / 14 |
| Paired game-score bootstrap 95% interval | 56.0–69.5% |
| Two-sided whole-pair swap p-value | 0.0005354 |
| Mean final score margin | +2.05 |
| Win rate as first / second player | 54% / 72% |
| Recorded seconds/move, iter15 / iter0 | 1.0409 / 2.1003 |

The 100 ordered openings are distinct and each is replayed with both player
assignments. The independent Python rules replay validates all 12,000 edges,
final scores and pair grouping. The bootstrap resamples whole opening pairs
(50,000 samples, seed 0). With no drawn games, the exact pair-swap test reduces
to a fair binomial test over the 40 sweeps and 14 losses; split pairs contribute
zero advantage. The saved legacy `a_pair_score=0.70` instead uses the sign of each
pair's summed score margin. **It is not the 63% game win rate.**

## Training and operational evidence

All 15 training logs continue from the preceding checkpoint, and all 15
collections use the latest learner. Checkpoint hashes agree across the state,
league and final command. The 6,100 optimizer updates therefore accumulate along
one ancestry. Each replay manifest contains only this run's appropriate rolling
eight-iteration window. All 1,500 fresh games and all labelling stages completed;
the logs report 1,026 fresh exact policy replacements in total.

Work consumed 29,252.49 seconds (8.126 hours), then the final arena took about
4,372 seconds. Timing files separate:

- Collection plus labelling: **23,842 seconds**.
- Training stage wall time: **976 seconds**.
- Development arenas: **4,428 seconds**.

The trainers themselves report **817 seconds** of elapsed training time in total.
That is a narrower timer than the shell's 976 seconds for the complete training
commands, which also include startup and preparation overhead. Neither is the
eight-hour overall work budget. Small differences from the summed stage timings
are controller and inter-stage overhead.

The saved league still says **champion0**: none of the small 20-game development
arenas passed the promotion rule. With `GATE=0`, that did not reset collection or
training. The separately planned, larger final match supports advancing iter15;
the league label is not evidence of another failed learning run.

This run used commit `3645000e194134d0a6750b1ee053eff7dffe54a0`. Every one of
the 60 recorded Python/shell source hashes matches that commit. The archive's
config and final-artifact hashes also validate.

## Limits and next comparison

The result measures improvement against iter0 under this search configuration
and opening distribution. It does not establish performance against Dabble,
PRsBoxes or BoxesZero, or from separately measured empty-board starts. The
faster recorded decisions are an observed consequence of this match, not an
isolated inference benchmark or a measurement of the new canonical speedup.

This overnight predates the proof-filter target correction and faster solver.
Its gain cannot be attributed to either change. Replay loss and exact-policy
diagnostics use changing populations and are not independent generalisation
tests. Full training NPZs, checkpoints and compiled-extension bytes are absent
from the uploaded archive, so their content cannot be independently checked
here; checkpoint identity must be enforced on the user's machine before reuse.

The incumbent checkpoint is the previous run's `iter15.pt`, SHA256:

```
3f17b19f3caa66905be1f33a960c038fec01fa499cda483ed00367b99e164dac
```

The next prepared experiment initializes from that checkpoint and freezes the
same weights as its final opponent. Preserve old iter0 for later ladder reads;
do not accidentally use it as the next final baseline and call that another
incremental improvement.

## Reproduce

`data/results.tgz` preserves the original 145,840-byte user upload unchanged.
SHA256:

```
166037e4520df66c84ce209c5813d00a433ee349c82762199826c1c371c84532
```

From the repository root, with the recorded commit available in local Git history:

```bash
EXP=experiments/2026-10-17_15-12-boxes-solver34-learning-result
mkdir -p work/solver34-learning-result
tar -xzf "$EXP/data/results.tgz" -C work/solver34-learning-result
PYTHONPATH="$PWD/src" uv run python "$EXP/analyze.py" \
  --results work/solver34-learning-result
```

The script checks the archive and extracted bytes, recorded source hashes,
configuration and final-artifact hashes, learner ancestry, replay manifests,
completed collection/label logs, and matched final settings. It replays with
the pure Python rules and recomputes statistics without using the arena's
statistics implementation. No GPU or checkpoint is required. The package import
expects the normal installed C++ extension, but replay itself uses only Python rules.
`--output DIR` chooses another output directory.

Outputs are `data/summary.json`, `data/iterations.csv`, parsed
`data/train-results.json`, and `data/verified-source-hashes.json`. The original
archive contains the full logs and evidence needed to regenerate these tables.
