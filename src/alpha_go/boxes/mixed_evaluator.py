"""Route policy and value from independently trained, complete Boxes networks."""
from __future__ import annotations

import math
from typing import Any

import torch

from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import BoxesLeafEvaluator, Evaluation


class MixedLeafEvaluator(BoxesLeafEvaluator):
    """Diagnostic evaluator; different models require two forwards per leaf batch.

    Each network uses its own feature encoding. Routing outputs preserves the
    learned trunk/readout pairing; it does not transplant incompatible head weights.
    Search budgets remain comparable, but mixed inference is not an equal-time test.
    """

    def __init__(self, policy_model: BoxesNet, value_model: BoxesNet, device: torch.device,
                 policy_temperature: float = 1.0, margin_utility_lambda: float = 0.0,
                 margin_utility_k: float = 6.0) -> None:
        if (policy_model.rows, policy_model.cols) != (value_model.rows, value_model.cols):
            raise ValueError("Policy and value checkpoints must use the same board size")
        if (not all(math.isfinite(x) for x in
                    (policy_temperature, margin_utility_lambda, margin_utility_k))
                or policy_temperature <= 0 or margin_utility_lambda < 0 or margin_utility_k <= 0):
            raise ValueError("Invalid policy temperature or margin utility settings")
        super().__init__(policy_model, device, policy_temperature, margin_utility_lambda,
                         margin_utility_k)
        self.value_evaluator = (None if value_model is policy_model else BoxesLeafEvaluator(
            value_model, device, policy_temperature, margin_utility_lambda, margin_utility_k
        ))

    def batch_evaluate(self, boards: list[Any]) -> list[Evaluation]:
        policy_results = super().batch_evaluate(boards)
        if self.value_evaluator is None:
            return policy_results
        value_results = self.value_evaluator.batch_evaluate(boards)
        return [(policy, value) for (policy, _), (_, value)
                in zip(policy_results, value_results)]
