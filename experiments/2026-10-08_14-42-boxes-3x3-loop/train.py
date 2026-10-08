"""Train BoxesNet for one iteration on a manifest of game directories.

Policy loss: cross-entropy against the MCTS visit distribution (every position, both
players); value loss: cross-entropy over the final margin for the side to move. Every batch
is augmented with a random lattice symmetry (8 for square boards). Prints a ===RESULT===
JSON line and saves checkpoints/iter{N}.pt next to this file.
"""
# ruff: noqa: N806
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import pick_device, save_boxes_net
from alpha_go.boxes.rules import geometry
from alpha_go.boxes.symmetry import apply, edge_permutation, transforms

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")).resolve()
MODEL = dict(channels=64, n_blocks=6, value_hidden=64)
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 100


def torch_apply(x: torch.Tensor, k: int) -> torch.Tensor:
    """Same transform as alpha_go.boxes.symmetry.apply, on tensors (last two axes)."""
    out = torch.flip(x, dims=(-1,)) if k >= 4 else x
    return torch.rot90(out, k % 4, dims=(-2, -1))


def check_torch_apply(rows: int, cols: int) -> None:
    grid = np.random.default_rng(0).random((3, 2 * rows + 1, 2 * cols + 1)).astype(np.float32)
    for k in transforms(rows, cols):
        assert np.array_equal(torch_apply(torch.from_numpy(grid), k).numpy(), apply(grid, k))


def augment(planes: torch.Tensor, policy: torch.Tensor, rows: int, cols: int,
            inverse_perms: dict[int, torch.Tensor], rng: np.random.Generator
            ) -> tuple[torch.Tensor, torch.Tensor]:
    ks = transforms(rows, cols)
    choice = rng.choice(len(ks), size=planes.shape[0])
    for i, k in enumerate(ks):
        mask = torch.from_numpy(choice == i).to(planes.device)
        if mask.any():
            planes[mask] = torch_apply(planes[mask], k)
            policy[mask] = policy[mask][:, inverse_perms[k]]
    return planes, policy


def manifest_dirs(path: Path) -> list[Path]:
    dirs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            dirs.append(GAME_DATA_DIR / line)
    return dirs


def schedule(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model: BoxesNet, loader: DataLoader, device: torch.device, max_batches: int = 40) -> dict:
    model.eval()
    policy_hits = value_hits = n = 0
    loss_sum = 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        planes = batch["planes"].to(device)
        policy, margin = batch["policy"].to(device), batch["margin"].to(device)
        total, _, _ = model.compute_loss(planes, policy, margin)
        logits_BE, margin_BM = model(planes)
        policy_hits += int((logits_BE.argmax(-1) == policy.argmax(-1)).sum())
        value_hits += int(((model.win_prob(margin_BM) > 0.5) == (margin > 0)).sum())
        loss_sum += float(total) * planes.shape[0]
        n += planes.shape[0]
    model.train()
    return {"loss": loss_sum / max(1, n), "policy_acc": policy_hits / max(1, n),
            "value_acc": value_hits / max(1, n)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-txt", required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--time-budget", type=int, default=180)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = pick_device("cpu" if args.cpu else None)
    assert args.cpu or device.type == "cuda", "CUDA not available; pass --cpu to run on CPU"
    rows, cols = args.rows, args.cols or args.rows
    check_torch_apply(rows, cols)
    t0 = time.time()
    dataset = BoxesDataset(manifest_dirs(EXP_DIR / args.dataset_txt))
    assert (dataset.rows, dataset.cols) == (rows, cols), (dataset.rows, dataset.cols)
    print(f"dataset: {len(dataset):,} positions from {len(dataset.games)} games "
          f"({dataset.num_with_mcts} searched), {dataset.footprint_bytes() / 1e6:.1f} MB in RAM, "
          f"loaded in {time.time() - t0:.1f}s")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        drop_last=len(dataset) > BATCH_SIZE)
    steps_per_epoch = max(1, len(loader))
    max_steps = args.max_epochs * steps_per_epoch

    model = BoxesNet(rows, cols, **MODEL).to(device)
    if args.resume_from:
        state = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"resumed from {args.resume_from}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: BoxesNet {MODEL} {n_params:,} params on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = schedule(optimizer, max_steps)
    use_amp = device.type == "cuda"
    geo = geometry(rows, cols)
    inverse_perms = {k: torch.from_numpy(np.argsort(edge_permutation(geo, k))).to(device)
                     for k in transforms(rows, cols)}
    rng = np.random.default_rng(args.iteration)

    model.train()
    step, start = 0, time.time()
    while step < max_steps and time.time() - start < args.time_budget:
        for batch in loader:
            if step >= max_steps or time.time() - start > args.time_budget:
                break
            planes = batch["planes"].to(device)
            policy, margin = batch["policy"].to(device), batch["margin"].to(device)
            planes, policy = augment(planes, policy, rows, cols, inverse_perms, rng)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                total, policy_loss, value_loss = model.compute_loss(planes, policy, margin)
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 100 == 0:
                print(f"  step {step}: loss={total.item():.4f} policy={policy_loss.item():.4f} "
                      f"value={value_loss.item():.4f} ({time.time() - start:.0f}s)")
    elapsed = time.time() - start
    train_eval = evaluate(model, loader, device)
    ckpt_dir = EXP_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"iter{args.iteration}.pt"
    save_boxes_net(model, ckpt_path, iteration=args.iteration, step=step)
    print(f"saved {ckpt_path} after {step} steps in {elapsed:.0f}s; train loss "
          f"{train_eval['loss']:.4f} policy_acc {train_eval['policy_acc']:.3f} "
          f"value_acc {train_eval['value_acc']:.3f}")
    print("\n===RESULT===")
    print(json.dumps({
        "iteration": args.iteration, "train_loss": round(train_eval["loss"], 6),
        "train_policy_acc": round(train_eval["policy_acc"], 4),
        "train_value_acc": round(train_eval["value_acc"], 4), "steps_completed": step,
        "positions": len(dataset), "elapsed_seconds": round(elapsed), "n_params": n_params,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if use_amp else 0,
        "checkpoint": str(ckpt_path), "resumed_from": args.resume_from or "",
    }))


if __name__ == "__main__":
    main()
