# Exact labels after fresh Qdab collection

The next artifact is a bounded, replay-verified training bundle from the **new**
`alpha_go.boxes.qdab_training_collect` run. Wait until its timed games and teacher
queries have stopped before running CPU solvers on the same machine. This plan
does not change weights, turn benchmark games into training data, or describe an
already implemented packing command.

## Input and boundary

The collector freezes `config.json`, its checkpoint, search settings, source
hashes, a schedule, and the family split before playing. Each `games/0000.json`
contains all 60 lattice moves including the four-edge opening, `slot.a_first`,
`a_margin`, `source_family`, `split`, and independently verified learner decisions.
Those decisions retain edge-ID prefixes, scores, mover, chosen edge, real visits,
raw visits when available, and normalized search policy. `teacher/*.json` stores
separate Qdab recommendations at selected 38–50-edge roots, explicitly marked
`approximate_qdab_action` and `exact_policy: false`. A completed game's observed
outcome is not its intermediate positions' perfect-play value.

Use only this fresh collection as a labeling source. The 120-game GPU benchmark,
the eight-game CPU pilot, and their diagnostic child tables remain evaluation or
development evidence. Their contents may be used for **exclusion checks**, never
as these training targets. Fork every derived label into a new output directory;
source games, learner search targets and teacher receipts remain unchanged.

The current collector excludes canonical opening families of the 100 paired GPU
benchmark games (seed 20261112) and eight CPU pilot games (seed 20261116). This is
not proof of later-state separation. Before training, compare against the actual
completed GPU benchmark, pilot and existing frozen development roots. Check D4
symmetry-equivalent state keys including edge mask, mover and score; report exact
coverage of the exclusion inputs. Do not claim all-state disjointness before those
actual histories are available. The empty board itself is a common start, not a
useful exclusion key; the checks concern sampled/training decision states and
opening families.

## Freeze and validate before solving

1. Hash `config.json`, every selected source game and teacher receipt, checkpoint,
   native extension and labeling implementation. Verify frozen source/checkpoint
   hashes and the collector's `verify_training_game` first.
2. Independently replay each full game through Python and native boards. At every
   candidate cut verify legal edges, owners/scores, extra turns, current margin,
   and the chosen next edge. Validate the final signed `a_margin` using
   `slot.a_first`; no sign flip is assumed merely because an edge was played.
3. Predeclare **all roots at 35, 36, 37 and 38 undrawn edges**, on both players'
   turns. Freeze their full prefixes, family, split, actor, played edge, board
   state, raw score/mover and forced-capture prefix before reading solver results
   or selecting mistakes. Record forced-capture roots too; they can supply value
   data even when no policy decision is required.
4. Preserve the collector's `source_family` and `training`/`validation` assignment.
   It canonicalizes the unordered four-edge starting mask under D4. Derived roots,
   children, capture collapses and branches inherit that family and split. Do not
   replace it with the older first-eight-move split. For the existing NPZ reader,
   encode this stable family as a hexadecimal `source_replay_group` scalar.
5. Deduplicate symmetric state transpositions across train/validation after
   generating raw and collapsed states. Validation wins collisions: remove the
   training instance, with a receipt. Never move validation rows into training.

## Bounded solve order

Use four workers, one private solver/table per active source game, 2^22 table
entries (about 67 MB per worker), and **20,000,000 nodes per root or child query**.
Persist every finished query or root group. Use a 45-minute outer label-session
limit, with a graceful shutdown and a hard termination fallback; completed work
must survive interruption. Resume verifies hashes and reuses saved results.

The priority is global, not driven by whether a model won or made a mistake:

| Pass | Population | Output | Initial session allocation |
|---|---|---|---|
| 1 | Every frozen 35–38 root and its actual played child | Exact root/selected-child values, or explicit unknown | Up to 10 minutes |
| 2 | Legal alternatives of those same roots, in a frozen round-robin game/band order | Complete child tables where affordable | Remaining 35 minutes |

If pass 1 uses its allocation, unfinished roots remain explicit and resume before
pass 2. Within a root, prioritize a saved Qdab recommendation only if the receipt
refers to that **identical** state; then use stable edge order. Teacher queries at
39–50 edges remain approximate advice outside this initial exact-label pass.
Do not spend the first session exhaustively solving the mostly unknown 41–42
region, or attempt to infer perfect labels from Qdab's action alone.

For each query retain budget, status (`exact`, `node_budget`, `deadline`, or
`not_attempted`), nodes/time if available, score/mover perspective, and source
identity. A budget failure is not a loss, draw or negative move label. A known root
with incomplete alternatives still has a usable exact value. Known children of
an incomplete group also have exact values; their unknown siblings are not
negative examples. When **all** legal children are exact, derive an unknown root
from the maximum child final margin in the root mover's perspective; retain the
original failed direct-root solve and mark the derivation separately. Check
agreement whenever the direct root was also known.

## Labels and trainer representation

The solver returns remaining margin for the state's mover. Add the state's known
score margin, then convert to fixed physical player one:

`final_p1 = current_score_p1 + remaining * (+1 if to_play == 1 else -1)`.

For comparing children, carry this final margin to the root mover's perspective.
Captures can retain the mover, so unconditional child-value negation is wrong.
The native/Python public `to_play()` is 1/2; existing NPZ `to_play` uses `player()`
indices 0/1. `exact_margin_p1` is the **final** player-one margin, not remaining
margin or the eventual observed game result.

- Complete all-child decision groups get a policy uniform over exact optimal
  legal moves and the exact root value. Report win-preserving mass separately;
  distinguish maximizing score margin from merely preserving a win.
- Independently known roots, raw children and their forced-capture-collapsed
  states may supply value-only rows. Deduplicate them and preserve lineage. Keep
  the raw and collapsed state mappings and verify collapse semantics; a terminal
  collapse has its exact terminal score.
- Value-only NPZ rows require `policy_mask=False`, a known `exact_margin_p1`, zero
  policy counts, and `train_mask=True`. Unknown states remain diagnostic records,
  never fabricated NPZ targets. Complete policy rows get an explicit exact-policy
  root mask; no invented MCTS visits are attributed to Qdab.
- Keep root-policy and child-value populations in separate portable manifests,
  together with per-family receipts and the frozen validation manifest. This
  prevents the much larger child population from silently dominating rehearsal
  and root-policy batches. Decide sampling weights in the subsequent GPU pilot;
  this collection does not justify another aggressive weight multiplier.

## Existing interfaces to reuse, and gaps

`qdab_training_collect.verify_training_game` validates the fresh source schema;
`qdab.replay_prefix` and Python/native boards provide score/mover replay checks.
`policy_ab_data.solve_children`, `ChildLabel`, `save_game` and `write_json` supply
bounded child solving, capture-collapse records and atomic artifact writes.
`curriculum_labels.label_arrays` demonstrates trainer-compatible complete-root
and value-only-child arrays. `dataset.BoxesDataset` enforces known exact values
for value-only rows. Reuse these semantics rather than creating another target
convention.

Do **not** invoke `all_policy_data` directly on these JSON games: it expects NPZ
histories with genuine MCTS counts and excludes unsearched/opponent roots. Do not
invoke the existing `curriculum_labels` JSON path unchanged either: it rederives
first-eight families, skips failed direct roots before child recovery, and only
emits complete groups. A small fresh-schema adapter/packer is still required to
preserve the collector's split, retain partial exact values and all unknown
records, check transpositions, and enforce the two-pass time budget. Its schema
should be finalized against a real completed collection game before implementation.

## Completion report and next GPU decision

Report attempted/known roots and children by band, actor and split; direct versus
derived roots; complete policies; forced/collapsed rows; source-family counts;
all exclusions; budget failures; and wall/solver time. Include a manifest of
artifact hashes and relative paths so the bundle can move from cloud to WSL.
Checkpoint weights remain unchanged.

The first GPU step is a short, matched training pilot on this bundle plus fixed
rehearsal, using the same starting checkpoint and update budget. Check held-out
fresh Qdab families for exact policy mass, value ordering and search regret, then
confirm any candidate in new full games. The existing GPU benchmark remains an
untouched opponent-strength read; do not pick training examples from its losses.

The evidence motivating this is specific: in the separate CPU pilot, one
37-edge root had exactly one winning action among 37. Its policy rank was 13 and
several losing children received higher value estimates. Two of four matched
opening pairs were already losing in both colors by 38. Earlier solver coverage
was poor (0/8 known at 41–42), so those observations do not identify most first
mistakes, prove a general value-head defect, or guarantee that this data will
improve full-game strength. The fresh-family pilot is how to test that hypothesis.
