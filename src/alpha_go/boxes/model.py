"""Boxes policy/value network on the lattice.

Trunk: upstream's masked residual blocks (`alpha_go.model.MaskedResBlock`) with an all-ones
mask, so the same code runs on any R x C lattice. Policy: a 1x1 conv gives one logit per
lattice cell; the logits at the E edge cells are gathered into `policy_BE`. Value: masked
average pool plus the margin scalar -> FC -> logits over the 2RC+1 final margins
`-RC..+RC` for the side to move; `win_prob()` derives P(margin > 0) + 0.5 P(margin = 0),
which is what the search consumes (upstream's scalar [0, 1] value convention).

Dimension key: B batch, K input planes, H/W lattice, E edges, M margin classes.
"""
# ruff: noqa: N803, N806, N812  (dimension-suffixed tensor names, as in alpha_go.model)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_go.boxes.encode import NUM_PLANES, edge_flat_index
from alpha_go.model import MaskedBatchNorm2d, MaskedGroupNorm2d, MaskedResBlock


class BoxesNet(nn.Module):
    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        channels: int = 64,
        n_blocks: int = 6,
        value_hidden: int = 64,
        norm_type: str = "bn",
        use_se: bool = False,
    ) -> None:
        super().__init__()
        cols = cols or rows
        self.rows, self.cols = rows, cols
        self.channels, self.n_blocks, self.value_hidden = channels, n_blocks, value_hidden
        self.norm_type, self.use_se = norm_type, use_se
        self.num_boxes = rows * cols
        self.num_margins = 2 * self.num_boxes + 1
        norm_cls: type[nn.Module] = MaskedGroupNorm2d if norm_type == "gn" else MaskedBatchNorm2d
        self.input_conv = nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False)
        self.input_bn = norm_cls(channels)
        self.blocks = nn.ModuleList(
            [MaskedResBlock(channels, norm_cls=norm_cls, use_se=use_se) for _ in range(n_blocks)]
        )
        self.policy_conv = nn.Conv2d(channels, 1, 1, bias=True)
        self.value_fc1 = nn.Linear(channels + 1, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, self.num_margins)
        self.edge_index_E: torch.Tensor
        self.margins_M: torch.Tensor
        self.register_buffer("edge_index_E", torch.from_numpy(edge_flat_index(rows, cols)))
        margins = torch.arange(-self.num_boxes, self.num_boxes + 1, dtype=torch.float32)
        self.register_buffer("margins_M", margins)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init readouts: uniform policy and a flat margin distribution at start.
        for readout in (self.policy_conv, self.value_fc2):
            nn.init.zeros_(readout.weight)
            assert readout.bias is not None
            nn.init.zeros_(readout.bias)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index_E.numel())

    def forward(self, planes_BKHW: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_BE logits over edges, margin_BM logits over final margins)."""
        B, _, H, W = planes_BKHW.shape
        mask_B1HW = torch.ones(B, 1, H, W, device=planes_BKHW.device, dtype=planes_BKHW.dtype)
        x_BCHW = F.relu(self.input_bn(self.input_conv(planes_BKHW), mask_B1HW))
        for block in self.blocks:
            x_BCHW = block(x_BCHW, mask_B1HW)
        logits_BHW = self.policy_conv(x_BCHW).view(B, H * W)
        policy_BE = logits_BHW[:, self.edge_index_E]
        pooled_BC = x_BCHW.mean(dim=(2, 3))
        margin_B1 = planes_BKHW[:, 10].mean(dim=(1, 2)).unsqueeze(1)
        hidden = F.relu(self.value_fc1(torch.cat([pooled_BC, margin_B1], dim=1)))
        margin_BM = self.value_fc2(hidden)
        return policy_BE, margin_BM

    def win_prob(self, margin_BM: torch.Tensor) -> torch.Tensor:
        """P(margin > 0) + 0.5 P(margin = 0) for the side to move, shape (B,)."""
        probs_BM = F.softmax(margin_BM.float(), dim=-1)
        weight_M = (self.margins_M > 0).float() + 0.5 * (self.margins_M == 0).float()
        return probs_BM @ weight_M

    def expected_margin(self, margin_BM: torch.Tensor) -> torch.Tensor:
        return F.softmax(margin_BM.float(), dim=-1) @ self.margins_M

    def compute_loss(
        self,
        planes_BKHW: torch.Tensor,
        target_policy_BE: torch.Tensor,
        target_margin_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Policy CE against a distribution over edges + CE over the final margin class."""
        policy_BE, margin_BM = self.forward(planes_BKHW)
        policy_loss = -(target_policy_BE * F.log_softmax(policy_BE, dim=-1)).sum(dim=-1).mean()
        value_loss = F.cross_entropy(margin_BM, (target_margin_B + self.num_boxes).long())
        return policy_loss + value_loss, policy_loss, value_loss
