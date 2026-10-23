"""Untrained, isolated direct-structure value head; no production loader changes."""
# ruff: noqa: N803, N806
from __future__ import annotations

import torch
from torch import nn

from alpha_go.boxes.model import BoxesNet


class StructuralValueHead(nn.Module):
    """Small margin readout, initialized to zero for safe residual attachment."""

    def __init__(self, feature_count: int, num_margins: int, hidden: int = 64) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.hidden = nn.Linear(feature_count, hidden)
        self.readout = nn.Linear(hidden, num_margins)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, features_BF: torch.Tensor) -> torch.Tensor:
        """Return margin logits or residual logits, shape B by margin classes."""
        if features_BF.ndim != 2 or features_BF.shape[1] != self.feature_count:
            raise ValueError("Structural feature shape does not match the frozen schema")
        return self.readout(torch.relu(self.hidden(features_BF)))


class StructuralValueResidual(nn.Module):
    """Frozen existing BoxesNet plus an independently trainable value correction.

    The base remains in evaluation mode, including BatchNorm. The policy path,
    trunk, existing readout, and checkpoint shapes are untouched. This wrapper
    deliberately does not implement checkpoint loading or arena registration.
    """

    def __init__(self, base: BoxesNet, feature_count: int, hidden: int = 64) -> None:
        super().__init__()
        if base.value_target != "final":
            raise ValueError("This first isolated prototype requires final-margin weights")
        self.base = base.eval()
        self.base.requires_grad_(False)
        self.correction = StructuralValueHead(feature_count, base.num_margins, hidden)

    def train(self, mode: bool = True) -> StructuralValueResidual:
        """Train only the correction; do not update frozen trunk normalization."""
        super().train(mode)
        self.base.eval()
        return self

    def forward(
        self, planes_BKHW: torch.Tensor, features_BF: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preserve base policy and add the structural residual to value logits."""
        if planes_BKHW.shape[0] != features_BF.shape[0]:
            raise ValueError("Board and structural feature batches differ")
        with torch.no_grad():
            policy_BE, margin_BM = self.base(planes_BKHW)
        return policy_BE, margin_BM + self.correction(features_BF)
