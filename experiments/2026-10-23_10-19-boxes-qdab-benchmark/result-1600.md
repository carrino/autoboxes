# Qdab at 1,600 simulations: higher observed score, substantial gap remains

The unchanged learner won **24/100** paired-opening games at 1,600 simulations,
versus **17/100** at 400. The observed seven-point improvement has a matched
opening-family bootstrap 95% interval of **−5 to +19 percentage points**. These
runs suggest search headroom but do not establish a reliable strength gain.
Neither run trained or changed the weights.

| Metric | 400 simulations | 1,600 simulations |
|---|---:|---:|
| Paired wins | 17/100 | 24/100 |
| Game-score 95% interval | 9–26% | 15–34% |
| First-seat wins | 9/50 | 17/50 |
| Second-seat wins | 8/50 | 7/50 |
| Relative Elo against this Qdab setup | −275 | −200 |
| Empty-board wins, descriptive only | 1/20 | 7/20 |
| Mean margin, paired games | +0.76 | +2.44 |
| Autoboxes seconds per played edge | 0.516 | 1.777 |
| Qdab seconds per played edge | 2.126 | 2.230 |

Our average time rose **3.44×**. Qdab's average time is now about **1.26×** ours;
these averages do not make this an equal-time experiment. The new 120 successful
attempts took 13,831 seconds, about **3h51m**, with no retries or forfeits.
The new relative-Elo interval is **−301 to −115**, conditional on this opponent,
hardware and configuration. It is not a published absolute rating. The small
reported p-value, `1.2913485988974571e-05`, tests the imbalance against Qdab,
not whether 1,600 simulations improve on 400.

## Matched comparison

Both runs used identical weights, opening schedule, learner seeds, evaluator,
solver34/50k, proof/terminal shaping, native-extension hash, recorded runtime and
pinned Qdab artifacts/settings. Only `num_simulations` changed in search.
The checkout commit changed, but all 70 existing frozen Python source hashes
match; the only new source is the separate training collector. No existing
benchmark/search source changed.

Of the 100 matched game slots, **20 changed from loss to win**, **13 from win
to loss**, four remained wins and 63 remained losses. Among the 50 opening
pairs, 16 improved, ten worsened and 24 tied in game score. Qdab does not expose
a controlled seed and uses timed search; these flips combine search-budget effects
and run-to-run opponent variation. They are not 20 individually proven rescues.

The comparison resamples the 50 complete opening clusters, keeping both seats
and both runs together: 100,000 percentile-bootstrap draws, seed 20261024.
The seat differences are exploratory: first +16 points [0, +32], second −2
[−18, +14]. They do not establish a second-player bug. The repeated empty-board
panel remains separate and contributes neither to Elo nor to the primary delta.
See [data/search-comparison.json](data/search-comparison.json) and
[data/matched-opening-pairs.csv](data/matched-opening-pairs.csv).

## What the margins reveal

Despite mean margin +2.44, the learner lost 76 games. **63 of those losses were
by one box**, twelve by three and one by five. Wins averaged **+14.5 boxes**;
losses averaged **−1.37**. All 24 wins were by at least five boxes. A positive
average margin therefore remains a poor description of competitive strength.
This is consistent with Qdab securing narrow wins and surrendering large margins
after losing its chance to win. It does not prove our margin utility causes losses.

The pair outcomes were 31 lose-both, 14 split and five win-both. Since each
25-box post-opening state has a theoretically winning side and we occupied both
seats, at least 31 games lost an initially winning post-opening position. No
opening values or first-error locations were solved in this audit.

## Decision

Keep the current weights; the 1,600 setting is an option when more runtime is
available, not a learning milestone. The result does not justify expecting a
further simulation multiplier to remove the Qdab gap.

Prioritize the separately specified early-policy transfer pilot: identical
positions, outcomes, replay, initial weights and update count; original learner
search targets in the control versus approximate Qdab recommendations on the
explicit teacher rows. The 60 fresh teaching games are already collected, but
the matched bundle exporter is still required before that GPU experiment is ready.
These two benchmark archives remain evaluation-only. The collector excludes
their shared opening bank; full-state exclusions must include this new archive
before creating the training bundles.

The cloud's completed small structural-value fit did not improve heldout child
selection. Do not promote those heads based on lower value loss. Its exact
root/child labels remain useful for diagnosis and a separately declared test of
learning differences between moves. Change one training intervention at a time.

## Independent verification and reproduction

The standalone verifier reconstructed all **7,200 edges**, ownership, final
scores and extra turns, plus **2,869 Qdab RPC request states** and their returned
bundles comprising 3,259 Qdab edges. All 120 first-attempt receipts, configuration
hashes and the regenerated schedule check out. All 71 frozen source hashes, six
pinned Qdab artifact hashes and the separately uploaded checkpoint match.
The archive records GPU execution; this audit does not re-execute GPU inference.

New run: `qdab-match-wmsy03l1`. Checkpoint:
`c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f`.
Config hash: `d18f61b6d873717ff14eea5aa8a71c0670859edc23122ecdb8c3c6bfc3b61612`.
The verification record is [data/review-1600.json](data/review-1600.json).

From the matching repository root, with the archives extracted under `work/`:

```bash
EXP=experiments/2026-10-23_10-19-boxes-qdab-benchmark
python "$EXP/verify_benchmark.py" work/qdab-gpu-benchmark-1yafbn9k work/verify-400
python "$EXP/verify_benchmark.py" work/qdab-gpu-benchmark-wmsy03l1 work/verify-1600
python "$EXP/compare_search.py" work/qdab-gpu-benchmark-1yafbn9k \
  work/qdab-gpu-benchmark-wmsy03l1 work/compare-search
```

The verifier defaults to the independently uploaded checkpoint and pinned local
Qdab release; explicit paths can be supplied as its third and fourth arguments.
No solving, training or new games are involved.
