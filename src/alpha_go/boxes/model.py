"""Boxes policy/value network on the lattice.

Trunk: upstream's masked residual blocks (`alpha_go.model.MaskedResBlock`) with an all-ones
mask, so the same code runs on any R x C lattice. Policy: a 1x1 conv gives one logit per
lattice cell; the logits at the E edge cells are gathered into `policy_BE`. Value: masked
average pool plus the margin scalar -> FC -> logits over the 2RC+1 final margins
`-RC..+RC` for the side to move; `win_prob()` derives P(margin > 0) + 0.5 P(margin = 0),
which is what the search consumes (upstream's scalar [0, 1] value convention).
With `value_target="remaining"` the head instead predicts the unclaimed boxes' margin;
forward masks impossible outcomes and adds the known score exactly, preserving that API.

Dimension key: B batch, K input planes, H/W lattice, E edges, M margin classes.
"""
# ruff: noqa: N803, N806, N812  (dimension-suffixed tensor names, as in alpha_go.model)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_go.boxes.encode import edge_flat_index, num_planes
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
        features: str = "basic",
        value_target: str = "final",
    ) -> None:
        super().__init__()
        assert value_target in ("final", "remaining"), value_target
        cols = cols or rows
        self.rows, self.cols = rows, cols
        self.channels, self.n_blocks, self.value_hidden = channels, n_blocks, value_hidden
        self.norm_type, self.use_se, self.features = norm_type, use_se, features
        self.value_target = value_target
        self.num_boxes = rows * cols
        self.num_margins = 2 * self.num_boxes + 1
        norm_cls: type[nn.Module] = MaskedGroupNorm2d if norm_type == "gn" else MaskedBatchNorm2d
        self.input_conv = nn.Conv2d(num_planes(features), channels, 3, padding=1, bias=False)
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

    def reset_value_head(self, value_target: str) -> None:
        """Explicitly change the value target, resetting only its final readout.

        Reinterpreting a trained final-margin head as remaining margins is invalid. The
        trunk, policy and first value layer are retained; policy evaluation is unchanged.
        """
        assert value_target in ("final", "remaining"), value_target
        self.value_target = value_target
        nn.init.zeros_(self.value_fc2.weight)
        assert self.value_fc2.bias is not None
        nn.init.zeros_(self.value_fc2.bias)

    def remaining_to_final_logits(
        self, remaining_BM: torch.Tensor, planes_BKHW: torch.Tensor,
    ) -> torch.Tensor:
        """Shift remaining-margin logits by the exact current score without losing mass.

        With U unclaimed boxes, remaining margins are -U, -U+2, ..., U. Masking that
        support before shifting means every feasible class stays within -RC..RC. Count
        ownership planes instead of undoing the normalised score plane under autocast.
        """
        owned_B2 = planes_BKHW[:, 1:3].sum(dim=(2, 3), dtype=torch.float32).long()
        score_B1 = (owned_B2[:, 0] - owned_B2[:, 1]).unsqueeze(1)
        unclaimed_B1 = self.num_boxes - owned_B2.sum(dim=1, keepdim=True)
        source_BM = self.margins_M.long().unsqueeze(0) - score_B1
        feasible_BM = ((source_BM.abs() <= unclaimed_B1)
                       & ((source_BM - unclaimed_B1).remainder(2) == 0))
        index_BM = (source_BM + self.num_boxes).clamp(0, self.num_margins - 1)
        final_BM = remaining_BM.gather(1, index_BM)
        return final_BM.masked_fill(~feasible_BM, float("-inf"))

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
        if self.value_target == "remaining":
            margin_BM = self.remaining_to_final_logits(margin_BM, planes_BKHW)
        return policy_BE, margin_BM

    def win_prob(self, margin_BM: torch.Tensor) -> torch.Tensor:
        """P(margin > 0) + 0.5 P(margin = 0) for the side to move, shape (B,)."""
        probs_BM = F.softmax(margin_BM.float(), dim=-1)
        weight_M = (self.margins_M > 0).float() + 0.5 * (self.margins_M == 0).float()
        # An autocast matmul would downcast this probability: 1 - 1e-6 then rounds to
        # one under bf16, making the optional binary loss produce 0 * log(0) = NaN.
        return (probs_BM * weight_M).sum(dim=-1)

    def expected_margin(self, margin_BM: torch.Tensor) -> torch.Tensor:
        return (F.softmax(margin_BM.float(), dim=-1) * self.margins_M).sum(dim=-1)

    def compute_loss(
        self,
        planes_BKHW: torch.Tensor,
        target_policy_BE: torch.Tensor,
        target_margin_B: torch.Tensor,
        target_win_B: torch.Tensor | None = None,
        weight_B: torch.Tensor | None = None,
        policy_weight_B: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Policy CE against a distribution over edges + CE over the final margin class.

        With `target_win_B` (e.g. BoxesZero's 0.75 outcome + 0.25 root-Q mix) a binary
        cross-entropy on the derived P(win) is added to the value loss. `weight_B` scales each
        position's value loss (a position with weight w counts w times; ones = unweighted).
        `policy_weight_B` gives nonnegative finite weights for the policy loss, normalised
        by their sum. Zero masks exact value-only rows without diluting the policy loss;
        an all-zero mask gives a differentiable zero policy loss.
        Remaining-head logits are shifted by the known score in forward: final-margin CE
        then equals CE against (target final margin - current score) on the remaining head.
        """
        policy_BE, margin_BM = self.forward(planes_BKHW)
        policy_B = -(target_policy_BE * F.log_softmax(policy_BE, dim=-1)).sum(dim=-1)
        if policy_weight_B is None:
            policy_loss = policy_B.mean()
        else:
            if policy_weight_B.shape != policy_B.shape or not policy_weight_B.is_floating_point():
                raise ValueError("policy_weight_B must have one floating weight per position")
            policy_loss = (policy_B * policy_weight_B).sum() / policy_weight_B.sum().clamp_min(1)
        value_B = F.cross_entropy(margin_BM, (target_margin_B + self.num_boxes).long(),
                                  reduction="none")
        if target_win_B is not None:
            win_B = self.win_prob(margin_BM).clamp(1e-6, 1 - 1e-6)
            value_B = value_B - (target_win_B * win_B.log()
                                 + (1 - target_win_B) * (1 - win_B).log())
        value_loss = (value_B if weight_B is None else value_B * weight_B).mean()
        return policy_loss + value_loss, policy_loss, value_loss
