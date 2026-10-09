"""Symmetry augmentation for training batches: the same lattice transforms as
`alpha_go.boxes.symmetry`, applied to plane tensors and to edge-indexed policy targets."""
# ruff: noqa: N803, N806
from __future__ import annotations

import numpy as np
import torch

from alpha_go.boxes.rules import geometry
from alpha_go.boxes.symmetry import edge_permutation, transforms


def torch_apply(x: torch.Tensor, k: int) -> torch.Tensor:
    """Same transform as `symmetry.apply`, on tensors (acts on the last two axes)."""
    out = torch.flip(x, dims=(-1,)) if k >= 4 else x
    return torch.rot90(out, k % 4, dims=(-2, -1))


def inverse_edge_perms(rows: int, cols: int, device: torch.device) -> dict[int, torch.Tensor]:
    """For every transform k the index vector that carries a policy over edges along with it:
    `policy_sym = policy[:, inverse_edge_perms[k]]` matches `torch_apply(planes, k)`."""
    geo = geometry(rows, cols)
    return {k: torch.from_numpy(np.argsort(edge_permutation(geo, k))).to(device)
            for k in transforms(rows, cols)}


def augment(planes_BKHW: torch.Tensor, policy_BE: torch.Tensor, rows: int, cols: int,
            inverse_perms: dict[int, torch.Tensor], rng: np.random.Generator
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independently drawn random symmetry to every sample of the batch, in place."""
    ks = transforms(rows, cols)
    choice = rng.choice(len(ks), size=planes_BKHW.shape[0])
    for i, k in enumerate(ks):
        mask = torch.from_numpy(choice == i).to(planes_BKHW.device)
        if mask.any():
            planes_BKHW[mask] = torch_apply(planes_BKHW[mask], k)
            policy_BE[mask] = policy_BE[mask][:, inverse_perms[k]]
    return planes_BKHW, policy_BE
