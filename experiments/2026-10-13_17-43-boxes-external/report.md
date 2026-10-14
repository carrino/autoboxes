# Published BoxesZero source audit and adapter

## Provenance

[Qirui Liu's author homepage](https://neuqrui.github.io/) links the BoxesZero paper to
[`neuqrui/-Distributed-Computer-Games-Framework`](https://github.com/neuqrui/-Distributed-Computer-Games-Framework).
The paper is [BoxesZero, Entropy 27(3), 285 (2025)](https://www.mdpi.com/1099-4300/27/3/285).
Its reported results are the authors' claims, not a result of this adapter's validation.

- Pinned repository commit: `a62c5b213961f69cad9be1c2b716680cfeb98ebe`.
- Checkpoint: `ComputingCenter1/data/dotsboxes_6/model_best.pth.tar`.
- Checkpoint Git blob: `4d3e02fd365bd03fedf8c2a4f7097a9f95dc468a`.
- Repository metadata gives its size as 12,403,693 bytes. ControlCenter's copy is the
  same blob; TrainCenter's copy differs.
- No LICENSE file was found in the pinned tree. The integration downloads an external
  installation; it does not vendor the author's source or weights into autoboxes.
- The repository does not establish that these weights are the paper's final or
  competition checkpoint. Call this opponent the **BoxesZero author-repository snapshot**.

All 15 selected source files were retrieved and matched against their exact Git blob
identifiers. The binary checkpoint could not be downloaded in this cloud environment:
the required shell proxy was unreachable, while the connector exposes text content only.
Its identity/size were inspected through repository metadata. Actual weight loading and
strength remain unverified until the README's user-side smoke and matches complete.

## Compatibility and search choices

The 11x11 lattice represents 5x5 boxes. Host edges are 0/1 and owners are 0/1/2;
upstream edges are 0/+2 and owners are 0/+1/-1 from the side-to-move perspective.
The upstream network outputs 121 policy probabilities and a scalar win/loss value.
The adapter does not reinterpret its value as a box margin.

The repository has inconsistent center configurations. ComputingCenter1 contains the
chain/loop-pruned MCTS and endgame logic, but its own settings omit required fields and
contradict ControlCenter assertions in that code. The adapter explicitly combines
ComputingCenter1's game, network, search and endgame modules with ControlCenter's settings.
This is a documented compatibility configuration, not a recovered tournament launcher.

Runtime choices, printed with the verified provenance:

1. Infer input channels (2, 4 or 26) from the checkpoint's first convolution. Load with
   `weights_only=True` and strict state-dict checking **after** TrainNet construction,
   because that constructor reinitializes weights. Evaluation mode and selected device.
2. Set `CenterName` to the Control Center expected by search; enable chain/loop pruning
   and endgame evaluation. Use a fixed caller-selected search budget with no training
   curriculum or training-example extraction. Select the most-visited root child.
3. Preserve upstream multi-edge capture plans across host moves only while the full
   expected board and player still agree, including upstream's `fake_last_move` handling.
4. Replace removed NumPy `ndarray.tostring()` with its equivalent `tobytes()` cache key.
   Convert the network's one-element value array to a scalar before upstream backup.
   These are compatibility shims; source files stay unchanged and hash-verified.
5. After either player owns 13 boxes, complete our game using the first legal edge.
   Win/loss is fixed, but the final margin is not optimized. Evaluate game wins.

Each arena game gets an isolated process, with full-state requests, validated legal
responses, a response deadline, and cleanup on failures. Opening prefixes need no replay
inside upstream. Model startup occurs before move timing; IPC and selection are timed.
Upstream stdout goes to stderr so diagnostic messages cannot corrupt the protocol.

## Validation

Machine: CPU-only cloud environment, 5 logical CPUs, approximately 18 GB RAM. Tests use
one or two Torch threads. No GPU training or user replay data was required.

The unchanged baseline passed 422 tests with 24 skips (excluding the repository's known
broken `test_gpu_lease.py`). New checks cover config and artifact validation, subprocess
failure/cleanup, full-state synchronization, capture perspective and forced sequences.
Optional checks use the actual pinned upstream source with **synthetic weights**:
eight seeded full-game rules comparisons, strict weight preservation, all three input
encodings, full-game search, and JSON-lines startup/reset/close.

Final full suite, including the optional source tests: **475 passed, 24 skipped in
49.08 seconds**. Ruff and strict mypy pass for all three new modules; new tests are
ruff-clean. README shell blocks pass `bash -n`.

The bridge also completed two 60-edge games through the existing paired arena and shared
gameplay loop against boxes-greedy, swapping colors and using a four-edge opening, with
CPU/synthetic weights. This proves an integration path; those outcomes say nothing about
the published model's strength. Two more full arena games passed with a synthetic
26-channel network, eight simulations, and a seven-edge opening with a captured box.

## Next decision

First run the real-checkpoint CPU smoke. If it succeeds, measure the frozen autoboxes
reference against this snapshot on the 3070, then compare the exact-label candidate with
the same settings. Our current exact-label diagnostic improvements make further learning
plausible; they do not show that autoboxes beats BoxesZero. The first opponent matches
will supply the missing evidence and reveal whether failures concentrate in openings,
chain/control decisions, or search budget. Dabble and PRsBoxes remain separate adapter
tasks; neither is implemented or measured by this change.
