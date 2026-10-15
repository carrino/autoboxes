"""Train a matched policy-target comparison on a frozen ``policy_ab_data`` bundle.

Both arms resume the same checkpoint, use the same exact values, source-family split,
batch order, augmentation, optimizer and schedule. Only supervised root move targets
differ. Child examples without a searched policy receive value loss only. This is an
isolated experiment: no league state, default loop or source checkpoint is modified.
"""
# ruff: noqa: N806, N812
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from alpha_go.boxes.augment import augment, inverse_edge_perms
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import load_boxes_net, pick_device, save_boxes_net
from alpha_go.boxes.validation import replay_group


@dataclass(frozen=True)
class TrainSettings:
    """A fixed update budget shared by both arms; no wall-time training cutoff."""

    epochs: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 20261015
    split_seed: int = 0
    val_fraction: float = 0.1
    min_undrawn: int = 24

    def validate(self) -> None:
        """Reject invalid or unbounded training settings before creating outputs."""
        assert self.epochs > 0 and self.batch_size > 0
        assert math.isfinite(self.lr) and self.lr > 0
        assert math.isfinite(self.weight_decay) and self.weight_decay >= 0
        assert self.seed >= 0 and self.split_seed >= 0 and self.min_undrawn >= 0
        assert 0 < self.val_fraction < 1, "a separate validation split is required"


def sha256(path: Path) -> str:
    """Hash one frozen input or result."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle_files(prepared: Path) -> dict[str, str]:
    """Check the preparation manifest and all hashed artifacts, including relative paths."""
    metadata = json.loads((prepared / "preparation.json").read_text())
    assert metadata["schema"] == 1, "unsupported preparation schema"
    assert metadata["status"] == "complete", "preparation is incomplete; finish labelling first"
    hashes = metadata["artifact_sha256"]
    assert isinstance(hashes, dict) and hashes, "preparation lacks artifact hashes"
    assert {"control-manifest.txt", "exact-manifest.txt"} <= hashes.keys()
    root = prepared.resolve()
    for name, digest in hashes.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root), f"artifact escapes preparation directory: {name}"
        assert path.is_file() and sha256(path) == digest, f"prepared artifact changed: {name}"
    actual = {str(path.relative_to(root)) for path in root.rglob("*.npz")}
    recorded = {name for name in hashes if name.endswith(".npz")}
    assert actual == recorded, "unrecorded or missing NPZs in the prepared bundle"
    return {**hashes, "preparation.json": sha256(root / "preparation.json")}


def load_arm(prepared: Path, arm: str, features: str, minimum: int) -> BoxesDataset:
    """Resolve portable manifest entries relative to the prepared bundle."""
    root = prepared.resolve()
    manifest = root / f"{arm}-manifest.txt"
    directories = [(root / line.strip()).resolve() for line in manifest.read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
    assert directories and all(path.is_relative_to(root) and path.is_dir()
                               for path in directories), "invalid prepared manifest directory"
    return BoxesDataset([str(p) for p in directories], min_undrawn=minimum,
                        features=features, use_exact=True)


def matched_datasets(control: BoxesDataset, exact: BoxesDataset) -> dict[str, int]:
    """Require identical selected positions, lineage, value targets and supervision masks.

    Policy differences are permitted only on the declared, completely solved roots.
    Normalized policies are compared because visit totals do not change training targets.
    """
    assert len(control.games) == len(exact.games), "arms have different game counts"
    assert (control.rows, control.cols, len(control)) == (exact.rows, exact.cols, len(exact))
    roots = changed = value_only = 0
    families: set[bytes] = set()
    root_families: set[bytes] = set()
    fields = ("boards", "to_play", "moves", "result", "exact_margin_p1", "mcts_root_values",
              "train_mask", "policy_mask", "policy_ab_root_mask", "root_value_semantics")
    for index, (a, b) in enumerate(zip(control.games, exact.games)):
        assert replay_group(a, control.geo) == replay_group(b, exact.geo), "different lineage"
        for field in fields:
            assert (field in a) == (field in b), f"different field presence: {field}"
            if field in a:
                assert np.array_equal(a[field], b[field]), f"arms differ in {field}"
        kept = control.position_indices[index]
        assert np.array_equal(kept, exact.position_indices[index]), "different row selection"
        count = len(a["boards"])
        marked = np.asarray(a.get("policy_ab_root_mask", np.zeros(count, dtype=bool)))
        assert marked.shape == (count,) and marked.dtype == np.bool_, "invalid root mask"
        mask = np.asarray(a.get("policy_mask", np.ones(count, dtype=bool)))
        assert mask.shape == (count,) and mask.dtype == np.bool_, "invalid policy mask"
        assert not np.any(marked & ~mask), "exact roots must have policy supervision"
        assert ("mcts_visits" in a) == ("mcts_visits" in b)
        if "mcts_visits" in a:
            old, new = np.asarray(a["mcts_visits"]), np.asarray(b["mcts_visits"])
            assert old.shape == new.shape == (count, control.geo.num_edges)
            assert np.isfinite(old).all() and np.isfinite(new).all()
            assert (old >= 0).all() and (new >= 0).all()
            old_sum, new_sum = old.sum(axis=1, keepdims=True), new.sum(axis=1, keepdims=True)
            old = old / np.where(old_sum > 0, old_sum, 1)
            new = new / np.where(new_sum > 0, new_sum, 1)
            differs = np.any(old != new, axis=1)
            assert not np.any(differs & ~marked), "policy changed outside selected roots"
            assert np.all(old[marked].sum(axis=1) > 0), "control root lacks search visits"
            assert np.all(new[marked].sum(axis=1) > 0), "exact root lacks optimal moves"
            changed += int(differs[kept].sum())
        else:
            assert not marked.any(), "selected root lacks search targets"
        roots += int(marked[kept].sum())
        value_only += int((~mask[kept]).sum())
        if len(kept):
            families.add(replay_group(a, control.geo))
        if marked[kept].any():
            root_families.add(replay_group(a, control.geo))
    return dict(positions=len(control), exact_roots=roots, changed_policy_rows=changed,
                value_only_rows=value_only, games=len(control.games), families=len(families),
                exact_root_families=len(root_families))


class IndexedPositions(Dataset[dict[str, Any]]):
    """Expose stable row indices for update matching and exact-root evaluation."""

    def __init__(self, dataset: BoxesDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        game_index = int(np.searchsorted(self.dataset.cumsum[1:], index, side="right"))
        row = int(self.dataset.position_indices[game_index][index - int(
            self.dataset.cumsum[game_index])])
        game = self.dataset.games[game_index]
        mask = game.get("policy_ab_root_mask")
        return dict(sample, position=index, exact_root=False if mask is None else bool(mask[row]))


@torch.no_grad()
def evaluate(model: BoxesNet, dataset: BoxesDataset, batch_size: int,
             device: torch.device) -> dict[str, Any]:
    """Evaluate every model against the same exact-arm validation targets in FP32.

    Policy loss uses supervised rows only. Optimal-set mass/hits use only marked roots
    with complete exact policies, never ordinary MCTS visit targets or value-only rows.
    """
    was_training = model.training
    model.eval()
    loader = DataLoader(IndexedPositions(dataset), batch_size=batch_size, shuffle=False)
    count = policy_count = root_count = correct = winning = hits = 0
    policy_ce = value_ce = mae = optimal_mass = 0.0
    for batch in loader:
        planes = batch["planes"].to(device)
        target = batch["policy"].to(device)
        margin = batch["margin"].to(device)
        weight = batch["policy_weight"].to(device)
        logits, values = model(planes)
        policy_ce += float((-(target * F.log_softmax(logits, -1)).sum(-1) * weight).sum())
        value_ce += float(F.cross_entropy(values, margin + model.num_boxes, reduction="sum"))
        expected = model.expected_margin(values)
        mae += float((expected - margin).abs().sum())
        correct += int(((model.win_prob(values) > 0.5) == (margin > 0)).sum())
        winning += int((margin > 0).sum())
        count += len(margin)
        policy_count += int(weight.sum())
        marked = batch["exact_root"].to(device)
        if marked.any():
            legal = planes[:, 0].flatten(1)[:, model.edge_index_E] == 0
            probs = F.softmax(logits.masked_fill(~legal, float("-inf")), -1)
            optimal = target > 0
            optimal_mass += float((probs * optimal).sum(-1)[marked].sum())
            hits += int(optimal.gather(1, probs.argmax(-1, keepdim=True))[marked].sum())
            root_count += int(marked.sum())
    model.train(was_training)
    return dict(positions=count, policy_positions=policy_count, exact_roots=root_count,
                policy_ce=policy_ce / policy_count if policy_count else None,
                value_ce=value_ce / count if count else None,
                margin_mae=mae / count if count else None,
                value_sign=correct / count if count else None,
                winning_fraction=winning / count if count else None,
                optimal_mass=optimal_mass / root_count if root_count else None,
                optimal_move_rate=hits / root_count if root_count else None)


def learning_rate_multiplier(step: int, total: int) -> float:
    """The loop's capped linear warmup and cosine schedule, shared by both arms."""
    warmup = min(100, total // 10)
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train_arm(name: str, reference: Path, train: BoxesDataset, validation: BoxesDataset,
              output: Path, settings: TrainSettings, device: torch.device) -> dict[str, Any]:
    """Train one arm with a reset optimizer and reproducible batch/augmentation streams."""
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    model = load_boxes_net(reference, device)
    model.train()
    generator = torch.Generator().manual_seed(settings.seed)
    loader = DataLoader(IndexedPositions(train), batch_size=settings.batch_size,
                        shuffle=True, generator=generator, num_workers=0, drop_last=False)
    total_steps = settings.epochs * len(loader)
    assert total_steps > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr,
                                 weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_multiplier(step, total_steps))
    rng = np.random.default_rng(settings.seed)
    inverse = inverse_edge_perms(model.rows, model.cols, device)
    trace = hashlib.sha256()
    step = 0
    started = time.monotonic()
    for epoch in range(settings.epochs):
        for batch in loader:
            indices = batch["position"].numpy().astype("<i8")
            trace.update(indices.tobytes())
            trace.update(json.dumps(rng.bit_generator.state, sort_keys=True).encode())
            trace.update(np.asarray([optimizer.param_groups[0]["lr"]], dtype="<f8").tobytes())
            planes = batch["planes"].to(device)
            policy = batch["policy"].to(device)
            planes, policy = augment(planes, policy, model.rows, model.cols, inverse, rng)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                total, _, _ = model.compute_loss(
                    planes, policy, batch["margin"].to(device),
                    policy_weight_B=batch["policy_weight"].to(device))
            assert bool(torch.isfinite(total)), f"non-finite loss in {name} at step {step}"
            optimizer.zero_grad(set_to_none=True)
            total.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 100 == 0:
                print(f"{name}: step {step}/{total_steps}, loss {float(total.detach()):.4f}",
                      flush=True)
        print(f"{name}: epoch {epoch + 1}/{settings.epochs}, {step} steps, "
              f"{time.monotonic() - started:.0f}s", flush=True)
    assert step == total_steps
    checkpoint = output / f"{name}.pt"
    save_boxes_net(model, checkpoint, iteration=1, step=step, seed=settings.seed,
                   split_seed=settings.split_seed, policy_ab_arm=name)
    metrics = evaluate(model, validation, settings.batch_size, device)
    result = dict(arm=name, steps=step, training_positions=len(train),
                  update_trace_sha256=trace.hexdigest(),
                  elapsed_seconds=time.monotonic() - started,
                  checkpoint=checkpoint.name, checkpoint_sha256=sha256(checkpoint),
                  validation=metrics)
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(reference: Path, prepared: Path, output: Path, settings: TrainSettings,
        *, cpu: bool = False) -> dict[str, Any]:
    """Validate the frozen bundle, run both arms, and require matched update traces."""
    settings.validate()
    assert not output.exists(), f"output must be fresh: {output}"
    inputs = bundle_files(prepared)
    reference_hash = sha256(reference)
    device = pick_device("cpu" if cpu else None)
    assert cpu or device.type == "cuda", "CUDA unavailable; --cpu is for tiny smoke tests"
    model = load_boxes_net(reference, device)
    arms = {name: load_arm(prepared, name, model.features, settings.min_undrawn)
            for name in ("control", "exact")}
    assert (model.rows, model.cols) == (arms["control"].rows, arms["control"].cols)
    coverage = matched_datasets(arms["control"], arms["exact"])
    splits = {name: data.split_stable(settings.val_fraction, settings.split_seed)
              for name, data in arms.items()}
    training = matched_datasets(splits["control"][0], splits["exact"][0])
    validation = matched_datasets(splits["control"][1], splits["exact"][1])
    assert training["positions"] and training["changed_policy_rows"], (
        "no changed policy targets in training; collect more eligible training families")
    assert validation["positions"], "no validation families; collect more games"
    if not validation["exact_roots"]:
        print("Validation has no completely labelled roots; this run is an integration pilot, "
              "not an exact-policy validation read.", flush=True)
    output.mkdir(parents=True)
    shutil.copyfile(reference, output / "reference.pt")
    assert sha256(output / "reference.pt") == reference_hash
    summary: dict[str, Any] = dict(
        schema=1, status="running", settings=asdict(settings), device=str(device),
        reference_sha256=reference_hash, prepared_sha256=inputs, coverage=coverage,
        training=training, validation=validation, arms={},
        scope="Matched development experiment; no automatic promotion or strength claim",
        precision="bf16 training on CUDA; FP32 training on CPU and validation",
        reproducibility="Matched order/augmentation/schedule; CUDA bitwise identity not promised",
        source_sha256={name: sha256(Path(__file__).parent / name) for name in (
            "policy_target_ab.py", "policy_ab_data.py", "dataset.py", "model.py", "augment.py",
            "validation.py")})
    summary["reference_validation"] = evaluate(model, splits["exact"][1],
                                                settings.batch_size, device)
    summary["reference_training"] = evaluate(model, splits["exact"][0],
                                              settings.batch_size, device)
    del model
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Matched data: {training['positions']} training rows, "
          f"{training['changed_policy_rows']} changed root policies, "
          f"{training['value_only_rows']} value-only rows; "
          f"{validation['positions']} validation rows", flush=True)
    for name in ("control", "exact"):
        result = train_arm(name, output / "reference.pt", splits[name][0], splits["exact"][1],
                           output, settings, device)
        trained = load_boxes_net(output / f"{name}.pt", device)
        result["training_metrics"] = evaluate(trained, splits["exact"][0],
                                               settings.batch_size, device)
        del trained
        (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        summary["arms"][name] = result
        path.write_text(json.dumps(summary, indent=2) + "\n")
    first, second = summary["arms"]["control"], summary["arms"]["exact"]
    assert first["steps"] == second["steps"]
    assert first["update_trace_sha256"] == second["update_trace_sha256"], "update streams differ"
    assert bundle_files(prepared) == inputs, "frozen data changed during training"
    assert sha256(reference) == reference_hash, "source checkpoint changed during training"
    summary["status"] = "complete"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "status", "training", "validation", "reference_training", "reference_validation",
        "arms")}, indent=2), flush=True)
    return summary


def main() -> None:
    """Run two training arms on an existing, reusable labelled dataset."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--min-undrawn", type=int, default=24)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    settings = TrainSettings(args.epochs, args.batch_size, args.lr, args.weight_decay,
                             args.seed, args.split_seed, args.val_fraction, args.min_undrawn)
    run(args.reference, args.prepared, args.output, settings, cpu=args.cpu)


if __name__ == "__main__":
    main()
