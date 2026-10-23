"""Initialization and isolation checks; these do not measure playing strength."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from structural_features import from_board
from structural_value import StructuralValueHead, StructuralValueResidual

from alpha_go.boxes.encode import encode
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.rules import BoxesBoard


def test_zero_residual_preserves_outputs_and_only_value_correction_learns() -> None:
    torch.manual_seed(20261023)
    torch.set_num_threads(1)
    boards = [BoxesBoard(2), BoxesBoard(2)]
    for edge in (0, 1, 3, 8):
        boards[1].play_edge(edge)
    planes = torch.from_numpy(np.stack([encode(board, "chains") for board in boards]))
    features = torch.from_numpy(np.stack([from_board(board) for board in boards]))
    base = BoxesNet(2, channels=8, n_blocks=1, value_hidden=8, features="chains").eval()
    # Make the fixture nontrivial: production checkpoints already have trained readouts.
    with torch.no_grad():
        base.policy_conv.weight.normal_(std=.01)
        base.value_fc2.weight.normal_(std=.01)
        expected_policy, expected_value = base(planes)
    frozen = {name: value.clone() for name, value in base.state_dict().items()}
    fused = StructuralValueResidual(base, features.shape[1], hidden=8).train()
    policy, value = fused(planes, features)
    assert torch.equal(policy, expected_policy)
    assert torch.equal(value, expected_value)
    assert not fused.base.training
    assert not policy.requires_grad
    loss = torch.nn.functional.cross_entropy(value, torch.tensor([3, 7]))
    loss.backward()
    parameters = list(fused.correction.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in parameters)
    assert fused.correction.readout.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in base.parameters())
    torch.optim.SGD(parameters, lr=.01).step()
    changed_policy, changed_value = fused(planes, features)
    assert torch.equal(changed_policy, expected_policy)
    assert not torch.equal(changed_value, expected_value)
    assert all(torch.equal(value, frozen[name]) for name, value in base.state_dict().items())


def test_standalone_head_and_shape_validation() -> None:
    head = StructuralValueHead(123, 51)
    assert sum(parameter.numel() for parameter in head.parameters()) == 11251
    assert torch.equal(head(torch.ones(2, 123)), torch.zeros(2, 51))
    with pytest.raises(ValueError, match="schema"):
        head(torch.ones(2, 122))
