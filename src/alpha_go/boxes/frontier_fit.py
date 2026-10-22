"""Fixed-data fitting study on verified frontier replay, with a diagnostic canary.

Every arm starts from the same reference and family split. Control and balanced
retain full trajectories and exact children; the canary fits only a small set of
distinct training roots and is never a candidate for full-game promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from alpha_go.boxes.balanced_policy import (
    clean_validation,
    coverage,
    export_validation,
    input_key,
    read_labels,
    root_mask,
    root_reads,
    root_subset,
    train_arm,
    verify_root_targets,
    write_json,
)
from alpha_go.boxes.curriculum_train import state_key, subset
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.nn_agent import load_boxes_net, pick_device
from alpha_go.boxes.policy_target_ab import bundle_files, load_arm, sha256
from alpha_go.boxes.validation import replay_group


def select_canary(dataset: BoxesDataset, band: int, count: int, seed: int
                  ) -> tuple[BoxesDataset, dict[str, Any]]:
    """Select a deterministic D4-unique subset from eligible training roots only."""
    assert count > 0 and band > 0 and seed >= 0
    candidates = []
    seen: set[bytes] = set()
    for game_index, (game, indices) in enumerate(zip(dataset.games, dataset.position_indices)):
        marked = root_mask(game)
        for row in indices:
            state = state_key(game, int(row), dataset.geo)
            if not marked[row] or dataset.geo.num_edges - state[0].bit_count() != band:
                continue
            key = input_key(game["boards"][row], int(game["to_play"][row]), dataset.geo)
            if key in seen:
                continue
            seen.add(key)
            order = hashlib.sha256(str(seed).encode() + b":" + key).hexdigest()
            candidates.append((order, game_index, int(row), state, key))
    assert candidates, f"no eligible exact training roots at {band} edges"
    chosen = sorted(candidates)[:count]
    masks = [np.zeros(len(game["boards"]), bool) for game in dataset.games]
    records = []
    for _, game_index, row, state, key in chosen:
        masks[game_index][row] = True
        game = dataset.games[game_index]
        records.append(dict(game_index=game_index, position=row, state=list(state),
            family=replay_group(game, dataset.geo).hex(),
            source_sha256=str(np.asarray(game.get("source_sha256", "")).item()),
            input_sha256=hashlib.sha256(key).hexdigest()))
    selected = subset(dataset, masks)
    report = dict(band=band, requested=count, eligible_unique=len(candidates),
                  selected=len(selected), seed=seed, records=records,
                  scope="Training-only memorization diagnostic; never an arena candidate")
    report["selection_sha256"] = hashlib.sha256(
        json.dumps(records, sort_keys=True).encode()).hexdigest()
    return selected, report


def run(reference: Path, prepared: Path, output: Path, arm: str, *, steps: int = 8192,
        batch_size: int = 256, lr: float = 1e-4, seed: int = 20261105,
        cpu: bool = False, min_undrawn: int = 24, canary_band: int = 37,
        canary_roots: int = 128) -> dict[str, Any]:
    """Train one fixed endpoint from frozen replay; no monitor selects a checkpoint."""
    assert arm in ("control", "balanced", "canary")
    assert steps > 0 and batch_size >= 4 and batch_size % 4 == 0 and seed >= 0
    assert math.isfinite(lr) and lr > 0 and min_undrawn >= 0
    assert canary_band > 0 and canary_roots > 0
    assert not output.exists(), f"output must be fresh: {output}"
    inputs = bundle_files(prepared)
    reference_hash = sha256(reference)
    device = pick_device("cpu" if cpu else None)
    assert cpu or device.type == "cuda", "CUDA unavailable; --cpu is for smoke tests"
    model = load_boxes_net(reference, device)
    dataset = load_arm(prepared, "exact", model.features, min_undrawn)
    assert (model.rows, model.cols) == (dataset.rows, dataset.cols)
    train, original_validation = dataset.split_stable(.1, 0)
    validation, filtering = clean_validation(train, original_validation)
    full_coverage = coverage(train)
    assert full_coverage["exact_roots"] and full_coverage["nonroots"], "empty replay pool"
    assert coverage(validation)["exact_roots"], "no clean validation roots"
    labels, source_rows = read_labels(prepared, dataset.geo)
    verify_root_targets(dataset, labels)
    selection = None
    if arm == "canary":
        train, selection = select_canary(train, canary_band, canary_roots, seed)
    output.mkdir(parents=True)
    shutil.copyfile(reference, output / "reference.pt")
    if selection is not None:
        write_json(output / "canary-selection.json", selection)
    search_validation = export_validation(root_subset(validation, canary_band),
                                          source_rows, output)
    assert search_validation["positions"], f"no held-out decision roots at {canary_band} edges"
    settings = dict(steps=steps, batch_size=batch_size, lr=lr, seed=seed,
                    roots_per_batch=batch_size // 4, root_fraction=.25,
                    weight_decay=1e-4, gradient_clip=1.0, split_seed=0,
                    val_fraction=.1, min_undrawn=min_undrawn,
                    schedule="constant", optimizer="fresh AdamW")
    if arm == "canary":
        settings.update(roots_per_batch=batch_size, root_fraction=1.0)
    summary: dict[str, Any] = dict(
        schema=1, status="running", arm=arm, settings=settings, device=str(device),
        reference_sha256=reference_hash, prepared_sha256=inputs,
        source_sha256={name: sha256(Path(__file__).parent / name) for name in (
            "frontier_fit.py", "balanced_policy.py", "curriculum_train.py",
            "policy_target_ab.py", "dataset.py", "model.py", "augment.py", "encode.py",
            "symmetry.py", "validation.py")},
        full_training=full_coverage, training=coverage(train), validation=coverage(validation),
        validation_filter=filtering, search_validation=search_validation, canary=selection,
        arena_eligible=arm != "canary",
        scope="Fixed endpoint, family-held-out development read; no automatic promotion",
        treatment="Uniform replay versus 25% exact roots; sampling changes policy/value "
                  "exposure and batch-normalization composition. Canary is diagnostic only.",
        reference=dict(training_roots=root_reads(model, train, labels, batch_size, device),
                       validation_roots=root_reads(model, validation, labels, batch_size, device)))
    del model
    write_json(output / "summary.json", summary)
    print(f"{arm}: {len(train)} training rows; {coverage(train)['exact_roots']} exact roots; "
          f"{search_validation['positions']} held-out {canary_band}-edge decision roots",
          flush=True)
    result = train_arm("balanced" if arm == "balanced" else "control", output / "reference.pt",
                       train, validation, labels, output, settings, device)
    assert result["steps"] == steps
    if arm == "canary":
        assert result["exposures"]["exact_roots"] == steps * batch_size
        assert result["unique_exact_roots"] == min(steps * batch_size, len(train))
    if arm == "balanced":
        assert result["exposures"]["exact_roots"] == steps * (batch_size // 4)
    assert bundle_files(prepared) == inputs, "prepared replay changed during training"
    assert sha256(reference) == reference_hash == sha256(output / "reference.pt")
    summary.update(status="complete", result=result)
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    """Train an isolated replay or memorization arm on an existing verified bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "prepared", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "balanced", "canary"), required=True)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--min-undrawn", type=int, default=24)
    parser.add_argument("--canary-band", type=int, default=37)
    parser.add_argument("--canary-roots", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
