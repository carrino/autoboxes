"""Compare uniform replay with a fixed exact-root quota on one frozen exact bundle.

Both arms use identical targets, family splits, update counts, augmentation and learning
rate. Sampling changes root policy/value exposure and batch-normalization composition.
The final fixed update is primary; intermediate reads never select a checkpoint.
"""
# ruff: noqa: N806
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import torch
from torch.utils.data import default_collate

from alpha_go.boxes.augment import augment, inverse_edge_perms
from alpha_go.boxes.curriculum_train import State, evaluate_roots, state_key, subset
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.nn_agent import load_boxes_net, pick_device, save_boxes_net
from alpha_go.boxes.policy_target_ab import (
    IndexedPositions,
    bundle_files,
    evaluate,
    load_arm,
    sha256,
)
from alpha_go.boxes.relabel import EXACT_UNKNOWN
from alpha_go.boxes.rules import BoxesGeometry
from alpha_go.boxes.search_probe import Position, position_key
from alpha_go.boxes.symmetry import apply, transforms
from alpha_go.boxes.validation import replay_group


def write_json(path: Path, value: Any) -> None:
    """Write a finite, readable diagnostic without replacing another stage's files."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def input_key(grid: np.ndarray[Any, Any], mover: int, geo: BoxesGeometry) -> bytes:
    """Canonicalize exactly the basic/chains model input, including box ownership.

    Input planes 0/1/2 recover drawn edges and mover-relative owners; all remaining
    planes are deterministic equivariant functions of those. Thus this compact key
    is equivalent to canonicalizing every encoded plane, without allocating 21 planes.
    Dots are ignored by the encoder. Absolute player names are intentionally absent.
    """
    assert mover in (0, 1)
    relative = np.zeros(geo.lattice_shape, np.uint8)
    relative[0::2, 1::2] = grid[0::2, 1::2] == 1
    relative[1::2, 0::2] = grid[1::2, 0::2] == 1
    boxes = grid[1::2, 1::2]
    relative[1::2, 1::2] = np.where(boxes == mover + 1, 1,
                                    np.where(boxes == 2 - mover, 2, 0))
    return min(apply(relative, k).tobytes() for k in transforms(geo.rows, geo.cols))


def root_mask(game: dict[str, Any]) -> np.ndarray[Any, Any]:
    """Return the declared complete exact-policy root mask, never infer it from values."""
    count = len(game["boards"])
    mask = np.asarray(game.get("policy_ab_root_mask", np.zeros(count, bool)))
    assert mask.dtype == np.bool_ and mask.shape == (count,), "invalid exact-root mask"
    return mask


def coverage(dataset: BoxesDataset) -> dict[str, Any]:
    """Count the actual eligible population and its source-family ownership."""
    roots = policy = exact = 0
    root_families: set[bytes] = set()
    families: set[bytes] = set()
    for game, indices in zip(dataset.games, dataset.position_indices):
        marked = root_mask(game)[indices]
        policy_mask = np.asarray(game.get("policy_mask", np.ones(len(game["boards"]), bool)))
        values = np.asarray(game.get("exact_margin_p1",
                                     np.full(len(game["boards"]), EXACT_UNKNOWN)))
        assert not np.any(marked & ~policy_mask[indices]), "root lacks policy supervision"
        assert not np.any(marked & (values[indices] == EXACT_UNKNOWN)), "root lacks exact value"
        roots += int(marked.sum())
        policy += int(policy_mask[indices].sum())
        exact += int((values[indices] != EXACT_UNKNOWN).sum())
        if len(indices):
            families.add(replay_group(game, dataset.geo))
        if marked.any():
            root_families.add(replay_group(game, dataset.geo))
    return dict(positions=len(dataset), exact_roots=roots, nonroots=len(dataset)-roots,
                policy_positions=policy, value_only_positions=len(dataset)-policy,
                exact_value_positions=exact, families=len(families),
                exact_root_families=len(root_families))


def clean_validation(train: BoxesDataset, validation: BoxesDataset
                     ) -> tuple[BoxesDataset, dict[str, Any]]:
    """Remove all D4 input overlaps, then deduplicate validation with roots first.

    Neither outcomes nor model predictions influence this filter. The training
    population and its naturally repeated states are left intact for both arms.
    """
    assert train.features in ("basic", "chains") and train.features == validation.features
    assert {replay_group(g, train.geo) for g in train.games}.isdisjoint(
        {replay_group(g, validation.geo) for g in validation.games}), "family leakage"
    keys = {input_key(g["boards"][i], int(g["to_play"][i]), train.geo)
            for g, indices in zip(train.games, train.position_indices) for i in indices}
    masks = [np.zeros(len(g["boards"]), bool) for g in validation.games]
    seen: set[bytes] = set()
    removed: Counter[str] = Counter()
    # Exact-policy roots take precedence over child rows describing the same input.
    for want_root in (True, False):
        for index, (game, indices) in enumerate(zip(validation.games,
                                                   validation.position_indices)):
            marked = root_mask(game)
            for row in indices:
                if bool(marked[row]) != want_root:
                    continue
                key = input_key(game["boards"][row], int(game["to_play"][row]), train.geo)
                reason = "training_overlap" if key in keys else "validation_duplicate"
                if key in keys or key in seen:
                    removed[reason] += 1
                    removed[reason + "_roots"] += int(want_root)
                    continue
                seen.add(key)
                masks[index][row] = True
    cleaned = subset(validation, masks)
    report = dict(before=coverage(validation), after=coverage(cleaned),
                  training_unique_inputs=len(keys),
                  **{name: removed[name] for name in ("training_overlap",
                     "training_overlap_roots", "validation_duplicate",
                     "validation_duplicate_roots")})
    return cleaned, report


def root_subset(dataset: BoxesDataset, undrawn: int | None = None) -> BoxesDataset:
    """Select declared exact roots, optionally within one undrawn-edge band."""
    masks = []
    for game, indices in zip(dataset.games, dataset.position_indices):
        mask = np.zeros(len(game["boards"]), bool)
        for row in indices:
            if root_mask(game)[row] and (undrawn is None or dataset.geo.num_edges -
                    state_key(game, int(row), dataset.geo)[0].bit_count() == undrawn):
                mask[row] = True
        masks.append(mask)
    return subset(dataset, masks)


class CyclingPool:
    """Independently shuffle a fixed pool, consuming each permutation before reshuffling."""

    def __init__(self, indices: np.ndarray[Any, Any], rng: np.random.Generator) -> None:
        assert len(indices), "empty sampling pool"
        self.indices = np.asarray(indices, np.int64)
        self.rng = rng
        self.order = np.empty(0, np.int64)
        self.offset = 0

    def take(self, count: int) -> np.ndarray[Any, Any]:
        """Draw exactly count indices; a batch may cross a permutation boundary."""
        chunks = []
        while count:
            if self.offset == len(self.order):
                self.order = self.rng.permutation(self.indices)
                self.offset = 0
            size = min(count, len(self.order) - self.offset)
            chunks.append(self.order[self.offset:self.offset + size])
            self.offset += size
            count -= size
        return np.concatenate(chunks) if chunks else np.empty(0, np.int64)


def batches(marked: np.ndarray[Any, Any], batch_size: int, steps: int,
            seed: int, root_count: int | None) -> Any:
    """Yield uniform batches or an exact quota; nonroot rows retain uniform frequency."""
    streams = np.random.SeedSequence(seed).spawn(3)
    if root_count is None:
        pool = CyclingPool(np.arange(len(marked)), np.random.default_rng(streams[0]))
        for _ in range(steps):
            yield pool.take(batch_size)
    else:
        assert 0 < root_count < batch_size
        roots = CyclingPool(np.flatnonzero(marked), np.random.default_rng(streams[0]))
        rest = CyclingPool(np.flatnonzero(~marked), np.random.default_rng(streams[1]))
        order = np.random.default_rng(streams[2])
        for _ in range(steps):
            yield order.permutation(np.concatenate([roots.take(root_count),
                                                    rest.take(batch_size-root_count)]))


def read_labels(prepared: Path, geo: BoxesGeometry
                ) -> tuple[dict[State, dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    """Join complete root/child CSVs and verify all raw legal children are represented."""
    recorded = json.loads((prepared / "preparation.json").read_text())["artifact_sha256"]
    paths = sorted(prepared.rglob("roots.csv"))
    for path in paths:
        for candidate in (path, path.with_name("children.csv")):
            assert str(candidate.relative_to(prepared)) in recorded, "unrecorded label CSV"
    source_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["complete"].lower() != "true":
                    continue
                key = row["source_sha256"], int(row["position"])
                assert key not in source_rows, "duplicate CSV source row"
                source_rows[key] = dict(row, child_margins={}, raw_children=[])
        children = path.with_name("children.csv")
        assert children.is_file(), "complete root table lacks children"
        with children.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["source_sha256"], int(row["position"])
                if key not in source_rows:
                    continue
                root = source_rows[key]
                sign = 1 if int(root["mover"]) == 0 else -1
                edge = int(row["edge"])
                assert edge not in root["child_margins"], "duplicate child edge"
                root["child_margins"][edge] = sign * int(row["exact_margin_p1"])
                root["raw_children"].append(tuple(int(row[k]) for k in
                    ("raw_edges", "raw_mover", "raw_score_p1", "exact_margin_p1")))
    labels: dict[State, dict[str, Any]] = {}
    for record in source_rows.values():
        state: State = int(record["edges"]), int(record["mover"]), int(record["score_p1"])
        sign = 1 if state[1] == 0 else -1
        record["root_margin"] = sign * int(record["exact_margin_p1"])
        legal = {edge for edge in range(geo.num_edges) if not state[0] >> edge & 1}
        assert set(record["child_margins"]) == legal, "incomplete raw child labels"
        assert max(record["child_margins"].values()) == record["root_margin"]
        if state in labels:
            assert labels[state]["child_margins"] == record["child_margins"], "label conflict"
        labels[state] = record
    assert labels, "no complete exact-policy roots"
    return labels, source_rows


def verify_root_targets(dataset: BoxesDataset, labels: dict[State, dict[str, Any]]) -> None:
    """Require NPZ root policies and values to agree with every exact child label."""
    for game, indices in zip(dataset.games, dataset.position_indices):
        for row in indices:
            if not root_mask(game)[row]:
                continue
            key = state_key(game, int(row), dataset.geo)
            label = labels[key]
            optimal = {edge for edge, value in label["child_margins"].items()
                       if value == label["root_margin"]}
            target = np.asarray(game["mcts_visits"][row])
            assert np.isfinite(target).all() and (target >= 0).all(), "invalid policy target"
            assert set(np.flatnonzero(target)) == optimal, "NPZ exact policy disagrees with CSV"
            assert np.all(target[list(optimal)] == target[list(optimal)][0])
            assert (1 if key[1] == 0 else -1) * int(game["exact_margin_p1"][row]) == (
                label["root_margin"])


def export_validation(dataset: BoxesDataset, source_rows: dict[tuple[str, int], dict[str, Any]],
                      output: Path) -> dict[str, int]:
    """Export clean held-out decision roots and a verified no-solve raw-child cache."""
    positions = []
    cache: dict[str, int] = {}
    seen: set[tuple[int, int, int]] = set()
    forced = duplicate = 0

    def remember(edges: int, mover: int, score: int, final: int) -> None:
        remaining = (final - score) * (1 if mover == 0 else -1)
        key = str(edges)
        assert key not in cache or cache[key] == remaining, "inconsistent exact cache collision"
        cache[key] = remaining

    for game, indices in zip(dataset.games, dataset.position_indices):
        for row in indices:
            if not root_mask(game)[row]:
                continue
            record = source_rows[(str(np.asarray(game["source_sha256"]).item()), int(row))]
            position = Position(dataset.rows, dataset.cols, json.loads(record["prefix"]),
                                record["source"], record["source_sha256"], record["family"])
            board = position.board()
            assert np.array_equal(board.to_numpy(), game["boards"][row]), "prefix board mismatch"
            assert int(board.to_play()) - 1 == int(game["to_play"][row]), "prefix mover mismatch"
            if alpha_go_cpp.BoxesSearchState(board).prefix():
                forced += 1
                continue
            key = position_key(board)
            if key in seen:
                duplicate += 1
                continue
            seen.add(key)
            positions.append(vars(position))
            remember(*(int(record[k]) for k in ("edges", "mover", "score_p1", "exact_margin_p1")))
            for child in record["raw_children"]:
                remember(*child)
    (output / "validation-positions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in positions))
    write_json(output / "validation-labels.json", dict(schema=1, rows=dataset.rows,
               cols=dataset.cols, remaining_values=cache))
    return dict(positions=len(positions), forced_prefix_excluded=forced,
                duplicate_search_state_excluded=duplicate, cached_states=len(cache))


def root_reads(model: Any, dataset: BoxesDataset, labels: dict[State, dict[str, Any]],
               batch_size: int, device: torch.device) -> dict[str, Any]:
    """Read pooled and individual-edge bands; retain each root for paired analysis."""
    was_training = model.training
    bands = sorted({dataset.geo.num_edges - state_key(g, int(i), dataset.geo)[0].bit_count()
                    for g, indices in zip(dataset.games, dataset.position_indices) for i in indices
                    if root_mask(g)[i]})
    result = {}
    for name, band in [("all", None), *[(str(band), band) for band in bands]]:
        roots = root_subset(dataset, band)
        metrics = evaluate_roots(model, roots, labels, batch_size, device)
        n = metrics["positions"]
        positive = metrics["winning_roots"]
        metrics["constant_majority_sign_accuracy"] = max(positive, n-positive) / n if n else None
        result[name] = metrics
    model.train(was_training)
    return result


def train_arm(name: str, reference: Path, train: BoxesDataset, validation: BoxesDataset,
              labels: dict[State, dict[str, Any]], output: Path, settings: dict[str, Any],
              device: torch.device) -> dict[str, Any]:
    """Train one fixed-update endpoint; preserve policy masks and log independent traces."""
    torch.manual_seed(settings["seed"])
    model = load_boxes_net(reference, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=1e-4)
    indexed = IndexedPositions(train)
    marked = np.concatenate([root_mask(g)[rows] for g, rows in
                             zip(train.games, train.position_indices)])
    quota = None if name == "control" else settings["roots_per_batch"]
    stream = batches(marked, settings["batch_size"], settings["steps"], settings["seed"], quota)
    rng = np.random.default_rng(settings["seed"] + 1)
    inverse = inverse_edge_perms(model.rows, model.cols, device)
    index_trace = hashlib.sha256()
    augment_trace = hashlib.sha256()
    indices_saved = []
    exposed: Counter[str] = Counter()
    unique: set[int] = set()
    reads = {}
    started = time.monotonic()
    read_steps = {min(256, settings["steps"]), min(1024, settings["steps"]), settings["steps"]}
    snapshot_every = settings.get("snapshot_every", 0)
    if snapshot_every:
        read_steps.update(range(snapshot_every, settings["steps"], snapshot_every))
    with (output / f"{name}-updates.jsonl").open("w") as log:
        for step, indices in enumerate(stream, 1):
            batch = default_collate([indexed[int(i)] for i in indices])
            indices_saved.append(indices.astype(np.int32))
            index_trace.update(indices.astype("<i8").tobytes())
            augmentation = dict(step=step, rng=rng.bit_generator.state, lr=settings["lr"],
                                batch_size=len(indices))
            augment_trace.update(json.dumps(augmentation, sort_keys=True).encode())
            nroot = int(batch["exact_root"].sum())
            npolicy = int(batch["policy_weight"].sum())
            exposed.update(rows=len(indices), exact_roots=nroot, policy_rows=npolicy,
                           value_only_rows=len(indices)-npolicy,
                           exact_value_rows=int(batch["exact"].sum()))
            unique.update(int(i) for i in indices)
            planes, policy = augment(batch["planes"].to(device), batch["policy"].to(device),
                                     model.rows, model.cols, inverse, rng)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, _, _ = model.compute_loss(planes, policy, batch["margin"].to(device),
                                                policy_weight_B=batch["policy_weight"].to(device))
            assert torch.isfinite(loss), "non-finite loss"
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            log.write(json.dumps(dict(augmentation, exact_roots=nroot, policy_rows=npolicy,
                      indices_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()))
                      + "\n")
            if step % 100 == 0 or step == settings["steps"]:
                print(f"{name}: step {step}/{settings['steps']}, loss {float(loss.detach()):.4f}, "
                      f"{time.monotonic()-started:.0f}s", flush=True)
                log.flush()
            if step in read_steps:
                reads[str(step)] = dict(training_roots=root_reads(
                    model, train, labels, settings["batch_size"], device),
                    validation_roots=root_reads(
                    model, validation, labels, settings["batch_size"], device))
                write_json(output / f"{name}-fit-{step}.json", reads[str(step)])
            if snapshot_every and step % snapshot_every == 0 and step < settings["steps"]:
                save_boxes_net(model, output / f"{name}-step{step}.pt", step=step,
                               seed=settings["seed"], diagnostic_only=True)
    checkpoint = output / f"{name}.pt"
    save_boxes_net(model, checkpoint, step=settings["steps"], seed=settings["seed"],
                   balanced_policy_arm=name, split_seed=0)
    np.savez_compressed(output / f"{name}-indices.npz", indices=np.stack(indices_saved))
    result = dict(status="complete", arm=name, steps=settings["steps"],
                  elapsed_seconds=time.monotonic()-started, checkpoint=checkpoint.name,
                  checkpoint_sha256=sha256(checkpoint), index_trace_sha256=index_trace.hexdigest(),
                  augmentation_schedule_sha256=augment_trace.hexdigest(), exposures=dict(exposed),
                  unique_rows=len(unique), unique_exact_roots=sum(marked[list(unique)]).item(),
                  validation=evaluate(model, validation, settings["batch_size"], device),
                  training=evaluate(model, train, settings["batch_size"], device),
                  final=reads[str(settings["steps"])],
                  diagnostic_steps=sorted(read_steps), primary_step=settings["steps"])
    write_json(output / f"{name}.json", result)
    return result


def run(reference: Path, prepared: Path, output: Path, *, steps: int = 2048,
        batch_size: int = 256, lr: float = 1e-4, seed: int = 20261020,
        root_fraction: float = 0.25, cpu: bool = False, min_undrawn: int = 24) -> dict[str, Any]:
    """Freeze inputs, filter validation, and compare uniform versus balanced exact replay."""
    assert steps > 0 and batch_size > 1 and seed >= 0 and min_undrawn >= 0
    assert math.isfinite(lr) and lr > 0 and 0 < root_fraction < 1
    roots_per_batch = round(root_fraction * batch_size)
    assert math.isclose(roots_per_batch, root_fraction * batch_size), "quota must be integral"
    assert 0 < roots_per_batch < batch_size
    assert not output.exists(), f"output must be fresh: {output}"
    inputs = bundle_files(prepared)
    reference_hash = sha256(reference)
    device = pick_device("cpu" if cpu else None)
    assert cpu or device.type == "cuda", "CUDA unavailable; --cpu is for small smoke tests"
    model = load_boxes_net(reference, device)
    dataset = load_arm(prepared, "exact", model.features, min_undrawn)
    assert (model.rows, model.cols) == (dataset.rows, dataset.cols)
    train, original_validation = dataset.split_stable(0.1, 0)
    validation, filtering = clean_validation(train, original_validation)
    train_coverage, val_coverage = coverage(train), coverage(validation)
    assert train_coverage["exact_roots"] and train_coverage["nonroots"], "empty training pool"
    assert val_coverage["exact_roots"], "no clean validation roots; collect more families"
    labels, source_rows = read_labels(prepared, dataset.geo)
    verify_root_targets(dataset, labels)
    output.mkdir(parents=True)
    shutil.copyfile(reference, output / "reference.pt")
    settings = dict(steps=steps, batch_size=batch_size, lr=lr, seed=seed,
                    root_fraction=root_fraction, roots_per_batch=roots_per_batch,
                    weight_decay=1e-4, gradient_clip=1.0, val_fraction=0.1, split_seed=0,
                    min_undrawn=min_undrawn, schedule="constant", optimizer="fresh AdamW")
    summary: dict[str, Any] = dict(
        schema=1, status="running", settings=settings, device=str(device),
        reference_sha256=reference_hash, prepared_sha256=inputs,
        source_sha256={name: sha256(Path(__file__).parent / name) for name in (
            "balanced_policy.py", "curriculum_train.py", "policy_target_ab.py", "dataset.py",
            "model.py", "augment.py", "encode.py", "symmetry.py", "validation.py")},
        training=train_coverage, validation=val_coverage, validation_filter=filtering,
        search_validation=export_validation(validation, source_rows, output), arms={},
        scope="Fixed-endpoint development experiment; no automatic promotion or strength claim",
        treatment="Root sampling changes policy/value exposure and batch-normalization composition",
        precision="CUDA bf16 training; CPU FP32 training; all evaluations FP32",
        reproducibility="Matching augmentation/schedule; different index streams; CUDA not bitwise",
        regret_units="final score-margin points")
    print(f"Training: {len(train)} rows / {train_coverage['exact_roots']} roots; "
          f"clean validation: {len(validation)} rows / {val_coverage['exact_roots']} roots",
          flush=True)
    summary["reference"] = dict(training=evaluate(model, train, batch_size, device),
        validation=evaluate(model, validation, batch_size, device),
        training_roots=root_reads(model, train, labels, batch_size, device),
        validation_roots=root_reads(model, validation, labels, batch_size, device))
    del model
    write_json(output / "summary.json", summary)
    for arm in ("control", "balanced"):
        write_json(output / f"{arm}-status.json", dict(status="running", steps=steps))
        result = train_arm(arm, output / "reference.pt", train, validation, labels,
                           output, settings, device)
        summary["arms"][arm] = result
        write_json(output / f"{arm}-status.json", dict(status="complete", steps=steps))
        write_json(output / "summary.json", summary)
    control, balanced = summary["arms"]["control"], summary["arms"]["balanced"]
    assert control["augmentation_schedule_sha256"] == balanced["augmentation_schedule_sha256"]
    assert balanced["exposures"]["exact_roots"] == steps * roots_per_batch
    assert bundle_files(prepared) == inputs, "prepared inputs changed during training"
    assert sha256(reference) == reference_hash == sha256(output / "reference.pt")
    summary["status"] = "complete"
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    """Run one isolated fixed-update sampling comparison against a frozen checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "prepared", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--root-fraction", type=float, default=0.25)
    parser.add_argument("--min-undrawn", type=int, default=24)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
