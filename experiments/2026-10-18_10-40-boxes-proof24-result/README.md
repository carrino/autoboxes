# Proof-target overnight: a second verified learned gain

Run `5x5-proof34-a6gdsd3i` finished 24 consecutive learning iterations in
8.172 active hours. Its predeclared final checkpoint, **iter24**, beat its frozen
starting checkpoint **121–79 (60.5%)** in 200 games with identical search.
The starting checkpoint was the previous solver34 run's winning **iter15**.
Use this new iter24 as the incumbent for subsequent experiments; retain both
older checkpoints as regression references.

| Final-match measure | Result |
| --- | ---: |
| Game wins | 121–79 |
| Paired game-score 95% interval, archived seed/count | 54.0–66.5% |
| Independent bootstrap, seed 0 / 50,000 samples | 54.0–67.0% |
| Paired two-sided win p-value | 0.0030877 |
| Opening pairs swept / split / lost | 34 / 53 / 13 |
| Mean final score margin | +1.11 |
| Candidate wins playing first / second | 45% / 76% |
| Candidate margin playing first / second | −1.12 / +3.34 |
| Recorded seconds per move, candidate / reference | 0.7515 / 0.9408 |

There are 100 distinct ordered four-edge openings, each played with sides
swapped. Both agents use 400 simulations, solver34 with a 50,000-node budget,
the same policy temperature/utility/search settings, and four arena workers.
All 200 games and 12,000 recorded edges were independently replayed with Python
rules. Config and final-artifact hashes match; the archived confidence interval
reproduces with its exact seed and sample count. The different independent
bootstrap endpoint reflects Monte Carlo variation, not contradictory games.

The legacy `a_pair_score=0.645` is a score based on summed margins per opening
pair; it is not the 60.5% game win rate. The substantial seat asymmetry should
remain visible in later matches. Pairing accounts for sides, but this match does
not establish an improvement separately in each seat. Recorded move time was
about 20% lower for the candidate in this matchup, not a controlled inference
or equal-time strength benchmark.

This is the second successive matched-search gain:

| Run | Candidate | Frozen opponent | Final result |
| --- | --- | --- | --- |
| `5x5-solver34-8pt6ai3d` | iter15 | original iter0 | 126–74 |
| `5x5-proof34-a6gdsd3i` | iter24 | preceding run's iter15 | 121–79 |

These are different comparisons. Do not multiply the win rates, infer a direct
win rate against original iter0, or attribute the second gain solely to corrected
targets: this run also provides another eight hours of training. It establishes
overall learned strength under the tested settings, not mastery of the value
function through 36 or strength against Dabble, PRsBoxes or BoxesZero.

## Training continuity and proof-target audit

- **2,400 fresh self-play games**, **1,413 complete exact-policy replacements**,
  and **10,256 optimizer updates** across 24 iterations.
- Every collector used the current learner, and every trainer resumed the
  preceding checkpoint. All eight-iteration replay manifests were checked.
- The saved league champion finally advanced to **iter22**, after a 17/20
  development match. The learner still continued through iter24. The larger
  predeclared final test supports iter24 as the next incumbent; no historical
  league or checkpoint is rewritten to manufacture a promotion.
- Collection plus labelling used **23,437 seconds**; training-stage commands
  **1,649 seconds** (1,376 reported inside training); development arenas
  **4,317 seconds**. The final arena is additional and took about 40 minutes.
- All **62 source hashes** and **25 native build-input hashes** match declared
  commit `6901dbc641eb96c2de8ebcfa0bb65e6c42ba380e`. The saved fresh-build preflight
  reports 17 passing checks and consistent imported-extension identity.

The 24 saved target reports cover 100 source files each. They contain 67,144
stored positions, of which 65,321 were searched. Proof filtering changed
**501 searched policy targets (0.767%)**. Average removed visit mass was
0.564% across all searched rows, including unchanged ones.

| Undrawn band | Changed / searched rows | Mean raw mass removed over searched rows |
| --- | ---: | ---: |
| 31–34 | 419 / 3,698 | 8.668% |
| 35–36 | 82 / 4,275 | 1.124% |
| 37+ | 0 | 0% |

These are collection-stage proof corrections, counted **before** the separate
35–36 exact-policy overrides. The archive contains internally consistent reports
and source NPZ hashes, not the NPZ bytes needed to repeat every per-row audit.
Changing replay validation populations and selected-mistake training metrics
are diagnostics, not independent band mastery scores.

## Next step

Follow the staged plan in
`experiments/2026-10-17_15-27-boxes-band-progression/CURRICULUM_PLAN.md`.
First compare this run's frozen iter0 and winning iter24 on the already frozen
35–36 and 37–38 development positions. Keep the solver/search settings matched.
Read selected-move regret, preserved wins and paired source-family intervals
before choosing a branch-start training pilot or expanding exact labels.

The band launcher in this folder performs **evaluation only**. It checks the two
checkpoint hashes, validates the imported runtime against the completed run,
uses finite per-band time and solve budgets, preserves partial results, and
packages a result archive. It does not train, rebuild the extension, or overwrite
the completed overnight run. Its checkpoint comparisons still need the actual
weights on the user's 3070 machine; the uploaded archive contains only their
recorded hashes.

After updating the Windows mirror and pulling this branch into WSL, run from
the repository root:

```bash
bash experiments/2026-10-18_10-40-boxes-proof24-result/run_band_probe.sh
```

It uses 80 frozen roots per band, 400 simulations, solver34/50k and a
20-million-node budget per label solve. Each band has a 60-minute wall limit,
so allow at most about two hours plus shutdown/reporting. This is a limit, not
a runtime prediction. Progress appears every 30 seconds. The printed log folder
contains `resume.sh` and `results.tgz`; send back that archive even if a band
times out. Completed bands survive resume, and only exact successes enter the
reused label cache. Unknown roots are excluded on the same basis for both
models, with coverage reported because solvable roots can be easier positions.

The default inputs are this completed run's `iter0.pt`, `iter24.pt` and
`config.json`. The launcher verifies their expected identities and requires the
same imported native extension as the overnight run. Full inference has not
been run in the cloud, where these checkpoints are absent. Focused launcher and
paired-report tests, Ruff, strict type checks and shell syntax checks pass.

## Reproduce the result audit

```bash
EXP=experiments/2026-10-18_10-40-boxes-proof24-result
mkdir -p work/proof24-upload
tar -xzf "$EXP/data/results.tgz" -C work/proof24-upload
uv run python "$EXP/analyze.py" --results work/proof24-upload
```

`analyze.py` reuses the earlier independent replay and training-ancestry verifier,
then checks this run's build receipt, proof-target summaries and bootstrap seed.
It writes `data/summary.json`, `iterations.csv`, full training results and
verified source/build hash maps. The original uploaded archive is retained:

- Size: **333,065 bytes**, 277 members, 4,893,554 uncompressed bytes.
- Archive SHA256: `8eef8346a33d2864db4f2d38af7f9d1391fb6f33d6dc491fce53ca7d585316b6`.
- Reference iter0 = old solver34 iter15 SHA256:
  `3f17b19f3caa66905be1f33a960c038fec01fa499cda483ed00367b99e164dac`.
- Winning iter24 SHA256:
  `73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d`.

Weights and native-library bytes are not included. Their recorded hashes are
consistent across configuration, training ancestry and final evaluation, but
cannot be recomputed from this archive alone.
