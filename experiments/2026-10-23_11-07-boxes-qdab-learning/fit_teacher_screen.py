"""Short matched approximate-policy fitting screen; no arena or automatic promotion.

Teacher states have policy loss only. The control distils the frozen reference's
raw legal policy, not invented search visits. Both arms see exactly the same
replay/teacher indices, augmentation, optimizer and fixed update count.
"""

# ruff: noqa: N803, N806, N812
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from alpha_go.boxes.augment import augment, inverse_edge_perms
from alpha_go.boxes.balanced_policy import CyclingPool, coverage, root_reads
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.encode import encode
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import load_boxes_net, pick_device, save_boxes_net
from alpha_go.boxes.policy_target_ab import IndexedPositions, bundle_files, sha256
from alpha_go.boxes.rules import BoxesBoard

ARMS = ("reference-policy", "qdab-policy")


def write_json(path: Path, value: Any) -> None:
    """Publish finite progress atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def hashes(folder: Path) -> dict[str, str]:
    """Hash regular artifacts in an immutable completed stage."""
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def teacher_tensors(
    records: list[dict[str, Any]], rows: int, cols: int, features: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay explicit teacher states and preserve their approximate target semantics."""
    if not records:
        raise ValueError("Empty teacher split")
    planes, targets = [], []
    for record in records:
        board = BoxesBoard(rows, cols)
        for edge in record["prefix"]:
            if not board.play_edge(int(edge)):
                raise ValueError("Invalid teacher prefix")
        if board.is_game_over():
            raise ValueError("Terminal teacher state")
        if "teacher_policy" in record:
            policy = np.asarray(record["teacher_policy"], dtype=np.float32)
        else:
            policy = np.zeros(board.geo.num_edges, np.float32)
            policy[int(record["teacher_edge"])] = 1
        legal = np.array([board.is_legal_edge(e) for e in range(board.geo.num_edges)])
        if (
            policy.shape != legal.shape
            or not np.isfinite(policy).all()
            or (policy < 0).any()
            or not np.isclose(policy.sum(), 1)
            or policy[~legal].sum() != 0
        ):
            raise ValueError("Teacher policy must be a legal normalized distribution")
        planes.append(encode(board, features))
        targets.append(policy)
    return torch.from_numpy(np.stack(planes)), torch.from_numpy(np.stack(targets))


@torch.no_grad()  # type: ignore[untyped-decorator]
def legal_policies(
    model: BoxesNet, planes: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Cache FP32 raw legal policies without changing normalization statistics."""
    was_training = model.training
    model.eval()
    values = []
    for start in range(0, len(planes), batch_size):
        batch = planes[start : start + batch_size].to(device)
        logits, _ = model(batch)
        legal = batch[:, 0].flatten(1)[:, model.edge_index_E] == 0
        values.append(torch.softmax(logits.float().masked_fill(~legal, -torch.inf), -1).cpu())
    model.train(was_training)
    return torch.cat(values)


def teacher_reads(
    model: BoxesNet,
    records: list[dict[str, Any]],
    planes: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Read approximate action agreement; this is not exact move correctness."""
    policies = legal_policies(model, planes, batch_size, device)
    ce = -(targets * policies.clamp_min(1e-30).log()).sum(-1)
    probability = (policies * targets).sum(-1)
    mass = (policies * (targets > 0)).sum(-1)
    agrees = targets.gather(1, policies.argmax(-1, keepdim=True)).squeeze(1) > 0
    rows = [
        dict(
            source_family=r["source_family"],
            teacher_probability=float(probability[i]),
            teacher_mass=float(mass[i]),
            teacher_ce=float(ce[i]),
            agrees=bool(agrees[i]),
        )
        for i, r in enumerate(records)
    ]
    return dict(
        positions=len(rows),
        families=len({r["source_family"] for r in records}),
        mean_teacher_probability=float(probability.mean()),
        mean_teacher_mass=float(mass.mean()),
        teacher_ce=float(ce.mean()),
        top_move_agreement=float(agrees.float().mean()),
        records=rows,
        scope="Approximate Qdab action support; teacher_probability is target-weighted",
    )


def sample_plan(
    replay_count: int, teacher_count: int, settings: dict[str, Any]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Cycle independently through both pools without replacing a row within a pass."""
    streams = np.random.SeedSequence(settings["seed"]).spawn(2)
    replay = CyclingPool(np.arange(replay_count), np.random.default_rng(streams[0]))
    teacher = CyclingPool(np.arange(teacher_count), np.random.default_rng(streams[1]))
    quota = settings["teacher_per_batch"]
    return (
        np.stack([replay.take(settings["batch_size"] - quota) for _ in range(settings["steps"])]),
        np.stack([teacher.take(quota) for _ in range(settings["steps"])]),
    )


def mixed_loss(
    policy_BE: torch.Tensor,
    margin_BM: torch.Tensor,
    targets_BE: torch.Tensor,
    target_margin_B: torch.Tensor,
    policy_weight_B: torch.Tensor,
    replay_count: int,
    num_boxes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Policy loss on supervised rows; value CE only on replay, with its own mean.

    Slicing before CE also avoids zero times infinity for a masked-out teacher
    class in a remaining-margin model. Teacher outcomes never become labels.
    """
    if not 0 < replay_count < len(policy_BE):
        raise ValueError("Both replay and teacher rows are required")
    policy_B = -(targets_BE * F.log_softmax(policy_BE, -1)).sum(-1)
    policy_loss = (policy_B * policy_weight_B).sum() / policy_weight_B.sum().clamp_min(1)
    value_loss = F.cross_entropy(
        margin_BM[:replay_count], target_margin_B[:replay_count] + num_boxes
    )
    return policy_loss + value_loss, policy_loss, value_loss


def paired_teacher_difference(
    left: dict[str, Any], right: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Bootstrap whole heldout families, never independent rows from the same game."""
    if len(left["records"]) != len(right["records"]):
        raise ValueError("Teacher reads differ in length")
    families: dict[str, list[list[float]]] = {}
    for a, b in zip(left["records"], right["records"], strict=True):
        if a["source_family"] != b["source_family"]:
            raise ValueError("Teacher record order differs")
        families.setdefault(a["source_family"], []).append(
            [
                b["teacher_ce"] - a["teacher_ce"],
                b["teacher_probability"] - a["teacher_probability"],
                float(b["agrees"]) - float(a["agrees"]),
            ]
        )
    values = np.array([np.mean(rows, axis=0) for rows in families.values()])
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    intervals = np.quantile(draws, [0.025, 0.975], axis=0)
    return dict(
        families=len(values),
        bootstrap_seed=seed,
        resamples=10000,
        metrics={
            name: dict(delta=float(values[:, i].mean()), ci95=intervals[:, i].tolist())
            for i, name in enumerate(("teacher_ce", "teacher_probability", "top_move_agreement"))
        },
        scope="Equal-family paired development interval; no strength claim",
    )


def fit_arm(
    arm: str,
    reference: Path,
    train: BoxesDataset,
    validation: BoxesDataset,
    labels: dict[Any, Any],
    teachers: dict[str, dict[str, Any]],
    soft_targets: torch.Tensor,
    replay_indices: np.ndarray[Any, Any],
    teacher_indices: np.ndarray[Any, Any],
    output: Path,
    settings: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Fit one fixed endpoint from the reference with shared replay and teacher draws."""
    torch.manual_seed(settings["seed"])
    model = load_boxes_net(reference, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    indexed = IndexedPositions(train)
    rng = np.random.default_rng(settings["seed"] + 1)
    inverse = inverse_edge_perms(model.rows, model.cols, device)
    augmentation_hash = hashlib.sha256()
    selected_targets = soft_targets if arm == ARMS[0] else teachers["training"]["targets"]
    source_planes = teachers["training"]["planes"]
    replay_size = settings["batch_size"] - settings["teacher_per_batch"]
    policy_rows = 0
    started = time.monotonic()
    diagnostic_steps = sorted(
        {min(256, settings["steps"]), min(1024, settings["steps"]), settings["steps"]}
    )
    with (output / "updates.jsonl").open("w") as log:
        for step, (ordinary, teacher) in enumerate(
            zip(replay_indices, teacher_indices, strict=True), 1
        ):
            batch = default_collate([indexed[int(i)] for i in ordinary])
            augmentation_hash.update(json.dumps(rng.bit_generator.state, sort_keys=True).encode())
            planes = torch.cat([batch["planes"], source_planes[teacher]]).to(device)
            policies = torch.cat([batch["policy"], selected_targets[teacher]]).to(device)
            planes, policies = augment(planes, policies, model.rows, model.cols, inverse, rng)
            # Teacher margin placeholders are excluded before value CE is evaluated.
            margins = torch.cat([batch["margin"], torch.zeros(len(teacher), dtype=torch.long)]).to(
                device
            )
            policy_weight = torch.cat([batch["policy_weight"], torch.ones(len(teacher))]).to(device)
            policy_rows += int(policy_weight.sum())
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, values = model(planes)
                loss, policy_loss, value_loss = mixed_loss(
                    logits, values, policies, margins, policy_weight, replay_size, model.num_boxes
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            row = dict(
                step=step,
                loss=float(loss.detach()),
                policy=float(policy_loss.detach()),
                value=float(value_loss.detach()),
                teacher_rows=len(teacher),
            )
            log.write(json.dumps(row) + "\n")
            if step % 100 == 0 or step == settings["steps"]:
                print(
                    f"{arm}: step {step}/{settings['steps']}, loss {row['loss']:.4f}, "
                    f"{time.monotonic() - started:.0f}s",
                    flush=True,
                )
                log.flush()
            if step in diagnostic_steps:
                read = {
                    name: teacher_reads(
                        model,
                        t["records"],
                        t["planes"],
                        t["targets"],
                        settings["batch_size"],
                        device,
                    )
                    for name, t in teachers.items()
                }
                write_json(output / f"teacher-read-{step}.json", read)
    checkpoint = output / "model.pt"
    save_boxes_net(
        model,
        checkpoint,
        step=settings["steps"],
        seed=settings["seed"],
        teacher_screen_arm=arm,
        approximate_teacher=True,
    )
    result = dict(
        status="complete",
        arm=arm,
        steps=settings["steps"],
        elapsed_seconds=time.monotonic() - started,
        checkpoint=checkpoint.name,
        checkpoint_sha256=sha256(checkpoint),
        index_trace_sha256=hashlib.sha256(
            replay_indices.astype("<i8").tobytes() + teacher_indices.astype("<i8").tobytes()
        ).hexdigest(),
        augmentation_schedule_sha256=augmentation_hash.hexdigest(),
        teacher_exposures=int(teacher_indices.size),
        unique_teacher_rows=int(len(np.unique(teacher_indices))),
        teacher_fraction=float(teacher_indices.size / (settings["steps"] * settings["batch_size"])),
        policy_rows=policy_rows,
        teacher_fraction_of_policy_rows=float(teacher_indices.size / policy_rows),
        replay_value_exposures=int(replay_indices.size),
        teacher_value_exposures=0,
        teacher_reads=read,
        exact_roots=dict(
            training=root_reads(model, train, labels, settings["batch_size"], device),
            validation=root_reads(model, validation, labels, settings["batch_size"], device),
        ),
        diagnostic_steps=diagnostic_steps,
    )
    write_json(output / "summary.json", result)
    return result


def run(
    reference: Path,
    replay: Path,
    payload: Path,
    output: Path,
    *,
    steps: int = 2048,
    batch_size: int = 256,
    teacher_per_batch: int = 8,
    lr: float = 1e-4,
    seed: int = 20261128,
    cpu: bool = False,
    min_undrawn: int = 24,
) -> dict[str, Any]:
    """Reuse verified completed arms; retry an interrupted arm from frozen weights."""
    from prepare_teacher_screen import load_payload, load_replay

    if steps <= 0 or not 0 < teacher_per_batch < batch_size or seed < 0 or min_undrawn < 0:
        raise ValueError("Require positive steps, valid teacher quota, and nonnegative seed")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    device = pick_device("cpu" if cpu else None)
    if not cpu and device.type != "cuda":
        raise ValueError("CUDA required; --cpu is only for small smoke tests")
    settings = dict(
        steps=steps,
        batch_size=batch_size,
        teacher_per_batch=teacher_per_batch,
        lr=lr,
        seed=seed,
        min_undrawn=min_undrawn,
        weight_decay=1e-4,
        value_supervision="ordinary replay only",
        precision="bf16 CUDA / FP32 CPU",
        control="Frozen reference raw legal policy on identical teacher states",
    )
    content = load_payload(payload)
    frozen = dict(
        settings=settings,
        reference_sha256=sha256(reference),
        replay_sha256=bundle_files(replay),
        payload_sha256=hashes(payload),
        source_sha256={
            p.name: sha256(p)
            for p in (Path(__file__), Path(__file__).with_name("prepare_teacher_screen.py"))
        },
        device=str(device),
    )
    output.mkdir(parents=True, exist_ok=True)
    recipe = output / "recipe.json"
    if recipe.exists():
        if json.loads(recipe.read_text()) != frozen:
            raise ValueError("Fitter inputs, settings or source changed on resume")
    else:
        write_json(recipe, frozen)
        shutil.copyfile(reference, output / "reference.pt")
    if sha256(output / "reference.pt") != frozen["reference_sha256"]:
        raise ValueError("Saved reference changed")
    model = load_boxes_net(reference, device)
    train, validation, labels, filtering = load_replay(replay, content, model.features, min_undrawn)
    if not len(train) or not len(validation):
        raise ValueError("Empty filtered replay split")
    teachers = {}
    for name, key in (
        ("training", "train_records"),
        ("validation", "validation_records"),
        ("diagnostic", "diagnostic_records"),
    ):
        records = content[key]
        planes, targets = teacher_tensors(records, model.rows, model.cols, model.features)
        teachers[name] = dict(records=records, planes=planes, targets=targets)
    print(
        f"Teacher quota: {teacher_per_batch}/{batch_size} rows per update "
        f"({100 * teacher_per_batch / batch_size:.3f}%); "
        f"{steps * teacher_per_batch:,} explicit teacher exposures per arm. "
        "Teacher rows have no value loss.",
        flush=True,
    )
    soft_targets = legal_policies(model, teachers["training"]["planes"], batch_size, device)
    ordinary, taught = sample_plan(len(train), len(soft_targets), settings)
    plan_path = output / "sample-plan.npz"
    if plan_path.exists():
        with np.load(plan_path) as saved:
            if not np.array_equal(saved["replay"], ordinary) or not np.array_equal(
                saved["teacher"], taught
            ):
                raise ValueError("Saved sample plan changed")
    else:
        np.savez_compressed(plan_path, replay=ordinary, teacher=taught)
    summary_path = output / "summary.json"
    summary: dict[str, Any]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = dict(
            status="running",
            settings=settings,
            checkpoint_sha256=sha256(reference),
            replay_filtering=filtering,
            training=coverage(train),
            validation=coverage(validation),
            teacher_counts={name: len(t["records"]) for name, t in teachers.items()},
            reference=dict(
                teacher_reads={
                    name: teacher_reads(
                        model, t["records"], t["planes"], t["targets"], batch_size, device
                    )
                    for name, t in teachers.items()
                },
                exact_roots=dict(
                    training=root_reads(model, train, labels, batch_size, device),
                    validation=root_reads(model, validation, labels, batch_size, device),
                ),
            ),
            arms={},
            scope="Fixed-endpoint development screen; no searched canary, arena "
            "or automatic promotion",
        )
        write_json(summary_path, summary)
    del model
    for arm in ARMS:
        folder = output / arm
        folder.mkdir(exist_ok=True)
        receipt_path = folder / "completed.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            attempt = folder / receipt["attempt"]
            if hashes(attempt) != receipt["artifact_sha256"]:
                raise ValueError(f"Completed arm changed: {arm}")
            result = json.loads((attempt / "summary.json").read_text())
            print(f"Reusing completed arm: {arm}", flush=True)
        else:
            attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=folder))
            result = fit_arm(
                arm,
                output / "reference.pt",
                train,
                validation,
                labels,
                teachers,
                soft_targets,
                ordinary,
                taught,
                attempt,
                settings,
                device,
            )
            write_json(receipt_path, dict(attempt=attempt.name, artifact_sha256=hashes(attempt)))
        summary["arms"][arm] = result
        write_json(summary_path, summary)
    left, right = (summary["arms"][arm] for arm in ARMS)
    for key in (
        "steps",
        "index_trace_sha256",
        "augmentation_schedule_sha256",
        "teacher_exposures",
        "replay_value_exposures",
    ):
        if left[key] != right[key]:
            raise ValueError(f"Unmatched training arms: {key}")
    summary["paired_teacher_difference"] = {
        name: paired_teacher_difference(
            left["teacher_reads"][name], right["teacher_reads"][name], seed
        )
        for name in ("validation", "diagnostic")
    }
    if (
        bundle_files(replay) != frozen["replay_sha256"]
        or hashes(payload) != frozen["payload_sha256"]
    ):
        raise ValueError("Training inputs changed")
    summary["status"] = "complete"
    write_json(summary_path, summary)
    print(
        json.dumps(
            dict(
                status="complete",
                teacher_counts=summary["teacher_counts"],
                paired_teacher_difference=summary["paired_teacher_difference"],
            ),
            indent=2,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    """Run the isolated teaching screen or reuse its completed arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "replay", "payload", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teacher-per-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261128)
    parser.add_argument("--min-undrawn", type=int, default=24)
    parser.add_argument("--cpu", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
