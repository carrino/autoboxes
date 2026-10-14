# CPU findings and pending GPU evidence

No new strength result is claimed. The user's 100-position policy/value probe is running
on the 3070; this work reads prior artifacts and prepares its analysis on four cloud CPUs.

## Concentration of earlier arena mistakes

The prior exact A/B arena contains 100 games per arm. In its already-solved replay, all
model moves in the 35..38 band had known regret. Selecting each game's worst model move
in hindsight captures about four fifths of that arm's total margin regret in this band.

| Arm | Known model moves | Total margin regret | Share from one worst move per game |
|---|---:|---:|---:|
| Reference | 171 | 538 | 81.4% |
| Outcome control | 184 | 626 | 81.8% |
| Exact-value candidate | 186 | 634 | 79.5% |

At most two positive-regret model moves per game account for all observed regret in this
short band. That is partly a consequence of alternating turns and the four-edge window;
it is not a broad compression claim. The result supports a small consequential-mistake
policy-label pilot on separate training games. These arena records remain evaluation only.
The input hash, counts, and reproducible script accompany the result.

## Verified training mechanics, not identified causes

`train.py` resumes weights and BatchNorm buffers, but creates fresh AdamW and a new
warmup/cosine schedule every invocation. With 131 batches per epoch, four epochs mean
524 updates and 52 warmup steps. Eight/twelve epochs have 1,048/1,572 updates and 100
warmup steps. At scheduler step 524 their learning rates are about 0.000582/0.000809,
while the four-epoch schedule has finished. An epoch sweep changes both exposure and
schedule; an optimizer-continuity test must hold batch order and learning-rate sequence
fixed while changing only whether moments survive a boundary.

Both reported training and validation diagnostics run in eval mode and FP32. The training
metric samples at most 40 shuffled batches; validation covers the whole validation set.
Training uses CUDA bf16, while search/oracle inference uses fp16. No train/eval mode bug
was demonstrated by the audit; small CPU checks found unchanged BatchNorm buffers during
evaluation and only floating-point-scale batch-composition differences.

## Analysis controls

The new report aligns cases by position ID, verifies frozen metadata, excludes the same
unknown positions from every case, and resamples source families together. Root winner
accuracy equal to the majority baseline is insufficient to conclude that predictions
are constant; Fable's sibling-ranking diagnostic is the independent next check. The active
probe and its game-family-aware report can identify which component deserves that work,
but cannot alone establish a stronger full-game player.

## Validation

The final CPU regression run passed: 566 passed, 29 skipped in 100.58 seconds, excluding
the known-broken upstream `test_gpu_lease.py`. All 26 focused report tests passed; they
cover paired alignment, shared family resampling, unknown labels, unavailable intervals,
input validation, and read-only output handling. Ruff and strict mypy passed on all three
new Python files. The CLI also read an existing completed two-position, four-case search
probe and wrote a report without loading a model or rerunning a solve. That smoke run
checks artifact compatibility only, not playing strength.
