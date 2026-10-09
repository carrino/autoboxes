"""Train BoxesNet for one iteration on a manifest of game directories.

Policy loss: cross-entropy against the MCTS visit distribution (every position, both
players); value loss: cross-entropy over the final margin for the side to move, plus with
--q-mix q > 0 a binary cross-entropy on P(win) against (1 - q) * outcome + q * root Q
(BoxesZero uses q = 0.25). A held-out split by game (--val-fraction) reports validation
loss and accuracies next to the training ones so overfitting is visible. Every batch
is augmented with a random lattice symmetry (8 for square boards; `alpha_go.boxes.augment`,
tested in tests/test_boxes_augment.py). Prints a ===RESULT===
JSON line and saves checkpoints/<tag>/iter{N}.pt next to this file (tag = <rows>x<cols>).
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

from alpha_go.boxes.augment import augment, inverse_edge_perms
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import pick_device, save_boxes_net

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")).resolve()
MODELS = {  # by box rows: 3x3 trains in minutes, 5x5 overnight on one 8 GB GPU
    3: dict(channels=64, n_blocks=6, value_hidden=64),
    5: dict(channels=128, n_blocks=10, value_hidden=64),
}
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 100


def manifest_dirs(path: Path) -> list[Path]:
    dirs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            dirs.append(GAME_DATA_DIR / line)
    return dirs


def schedule(optimizer: torch.optim.Optimizer, total_steps: int
             ) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay; the warmup never exceeds a tenth of the steps (a
    small replay window with TRAIN_EPOCHS=2 is only ~100 steps per iteration)."""
    warmup = min(WARMUP_STEPS, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model: BoxesNet, loader: DataLoader, device: torch.device,
             max_batches: int = 40) -> dict:
    model.eval()
    policy_hits = value_hits = n = 0
    loss_sum = 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches or len(loader) == 0:
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
    p.add_argument("--tag", default=None, help="checkpoint subdir; default <rows>x<cols>")
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="share of games held out for validation metrics (0 = none)")
    p.add_argument("--q-mix", type=float, default=0.0,
                   help="weight of the root Q in the P(win) target (0 = pure outcome)")
    p.add_argument("--min-undrawn", type=int, default=0,
                   help="train only on positions with at least this many undrawn edges "
                        "(a solver run's net never needs the solver zone)")
    p.add_argument("--channels", type=int, default=None, help="override the net width")
    p.add_argument("--n-blocks", type=int, default=None, help="override the residual depth")
    p.add_argument("--features", choices=["basic", "chains"], default="basic",
                   help="input planes (encode.py): chains adds the chain / loop structure")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = pick_device("cpu" if args.cpu else None)
    assert args.cpu or device.type == "cuda", "CUDA not available; pass --cpu to run on CPU"
    rows, cols = args.rows, args.cols or args.rows
    t0 = time.time()
    dataset = BoxesDataset(manifest_dirs(EXP_DIR / args.dataset_txt), min_undrawn=args.min_undrawn,
                           features=args.features)
    assert (dataset.rows, dataset.cols) == (rows, cols), (dataset.rows, dataset.cols)
    print(f"dataset: {len(dataset):,} positions with >= {args.min_undrawn} undrawn edges from "
          f"{len(dataset.games)} games ({dataset.num_with_mcts} searched), "
          f"{dataset.footprint_bytes() / 1e6:.1f} MB in RAM, loaded in {time.time() - t0:.1f}s")
    train_set, val_set = dataset.split(args.val_fraction, seed=args.iteration)
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        drop_last=len(train_set) > BATCH_SIZE)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"split: {len(train_set):,} training / {len(val_set):,} validation positions "
          f"({len(val_set.games)} held-out games); q-mix {args.q_mix}")
    steps_per_epoch = max(1, len(loader))
    max_steps = args.max_epochs * steps_per_epoch

    model_cfg = {**MODELS.get(rows, MODELS[5]),
                 **{k: v for k, v in dict(channels=args.channels, n_blocks=args.n_blocks).items()
                    if v is not None}}
    model = BoxesNet(rows, cols, features=args.features, **model_cfg).to(device)
    if args.resume_from:
        state = torch.load(args.resume_from, map_location=device, weights_only=False)
        assert state["config"].get("features", "basic") == args.features, state["config"]
        model.load_state_dict(state["model_state_dict"])
        print(f"resumed from {args.resume_from}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: BoxesNet {model_cfg} {n_params:,} params on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = schedule(optimizer, max_steps)
    use_amp = device.type == "cuda"
    inverse_perms = inverse_edge_perms(rows, cols, device)
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
            target_win = None
            if args.q_mix > 0:
                target_win = ((1 - args.q_mix) * batch["win"]
                              + args.q_mix * batch["root_value"]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                total, policy_loss, value_loss = model.compute_loss(planes, policy, margin,
                                                                    target_win)
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
    val_eval = evaluate(model, val_loader, device, max_batches=10**9)
    ckpt_dir = EXP_DIR / "checkpoints" / (args.tag or f"{rows}x{cols}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"iter{args.iteration}.pt"
    save_boxes_net(model, ckpt_path, iteration=args.iteration, step=step)
    print(f"saved {ckpt_path} after {step} steps in {elapsed:.0f}s; train loss "
          f"{train_eval['loss']:.4f} policy_acc {train_eval['policy_acc']:.3f} "
          f"value_acc {train_eval['value_acc']:.3f}; val loss {val_eval['loss']:.4f} "
          f"policy_acc {val_eval['policy_acc']:.3f} value_acc {val_eval['value_acc']:.3f}")
    print("\n===RESULT===")
    print(json.dumps({
        "iteration": args.iteration, "train_loss": round(train_eval["loss"], 6),
        "train_policy_acc": round(train_eval["policy_acc"], 4),
        "train_value_acc": round(train_eval["value_acc"], 4),
        "val_loss": round(val_eval["loss"], 6), "val_policy_acc": round(val_eval["policy_acc"], 4),
        "val_value_acc": round(val_eval["value_acc"], 4), "val_positions": len(val_set),
        "q_mix": args.q_mix, "min_undrawn": args.min_undrawn, "features": args.features,
        "model": model_cfg,
        "steps_completed": step,
        "positions": len(train_set), "elapsed_seconds": round(elapsed), "n_params": n_params,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if use_amp else 0,
        "checkpoint": str(ckpt_path), "resumed_from": args.resume_from or "",
    }))


if __name__ == "__main__":
    main()
