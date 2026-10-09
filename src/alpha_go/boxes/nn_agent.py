"""Boxes MCTS agents backed by BoxesNet.

`BoxesLeafEvaluator` owns a net and evaluates a batch of leaves in one forward pass (used by
the leaf-parallel C++ search inside one game thread). `BoxesEngineEvaluator` submits leaves
to a shared `PlaneBatchedEngine` so many game threads share forwards. Both return, per
board, a policy over its legal edges and the side-to-move win probability (optionally
shaped by the margin-utility term from PLAN.md decision 4).
"""
# ruff: noqa: N803, N806  (dimension-suffixed tensor names)
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import torch
from numpy.typing import NDArray

from alpha_go.agents.base import Agent, register_agent
from alpha_go.boxes.encode import encode_batch
from alpha_go.boxes.inference import PlaneBatchedEngine
from alpha_go.boxes.model import BoxesNet

Evaluation = tuple[dict[int, float], float]


def pick_device(device: str | None = None) -> torch.device:
    return torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))


def save_boxes_net(model: BoxesNet, path: str | Path, **extra: Any) -> None:
    """Checkpoint with enough config to rebuild the net."""
    config = dict(
        rows=model.rows, cols=model.cols, channels=model.channels, n_blocks=model.n_blocks,
        value_hidden=model.value_hidden, norm_type=model.norm_type, use_se=model.use_se,
    )
    torch.save({"model_state_dict": model.state_dict(), "config": config, **extra}, path)


def load_boxes_net(path: str | Path, device: torch.device) -> BoxesNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = BoxesNet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def policy_dict(
    logits_E: NDArray[np.float32], legal: list[int], temperature: float
) -> dict[int, float]:
    """Softmax over the legal edges only."""
    scaled = logits_E[legal] / temperature
    probs = np.exp(scaled - scaled.max())
    probs /= probs.sum()
    return dict(zip(legal, probs.tolist()))


def shaped_value(win: float, expected_margin: float, lam: float, k: float) -> float:
    """Decision 4: u = win_prob + lambda * tanh(expected_margin / k); lambda=0 is plain win prob."""
    return win + lam * math.tanh(expected_margin / k)


class BoxesLeafEvaluator:
    """One forward pass per batch of leaves; fp16 autocast on CUDA."""

    def __init__(
        self,
        model: BoxesNet,
        device: torch.device,
        policy_temperature: float = 1.0,
        margin_utility_lambda: float = 0.0,
        margin_utility_k: float = 6.0,
        checkpoint_path: str | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.policy_temperature = policy_temperature
        self.margin_utility_lambda = margin_utility_lambda
        self.margin_utility_k = margin_utility_k
        self.checkpoint_path = checkpoint_path
        self.use_fp16 = device.type == "cuda"

    def evaluate(self, board: Any) -> Evaluation:
        return self.batch_evaluate([board])[0]

    @torch.no_grad()
    def batch_evaluate(self, boards: list[Any]) -> list[Evaluation]:
        planes_BKHW = torch.from_numpy(encode_batch(boards)).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_fp16):
            policy_BE, margin_BM = self.model(planes_BKHW)
        logits = policy_BE.float().cpu().numpy()
        win = self.model.win_prob(margin_BM).cpu().numpy()
        expected = self.model.expected_margin(margin_BM).cpu().numpy()
        return [
            (
                policy_dict(logits[i], b.get_legal_moves_flat(), self.policy_temperature),
                shaped_value(float(win[i]), float(expected[i]),
                             self.margin_utility_lambda, self.margin_utility_k),
            )
            for i, b in enumerate(boards)
        ]

    def close(self) -> None:
        pass


class BoxesEngineEvaluator:
    """Submits every leaf to a shared PlaneBatchedEngine (cross-game batching)."""

    def __init__(
        self,
        engine: PlaneBatchedEngine,
        policy_temperature: float = 1.0,
        margin_utility_lambda: float = 0.0,
        margin_utility_k: float = 6.0,
        checkpoint_path: str | None = None,
    ) -> None:
        self.engine = engine
        self.policy_temperature = policy_temperature
        self.margin_utility_lambda = margin_utility_lambda
        self.margin_utility_k = margin_utility_k
        self.checkpoint_path = checkpoint_path

    def evaluate(self, board: Any) -> Evaluation:
        return self.batch_evaluate([board])[0]

    def batch_evaluate(self, boards: list[Any]) -> list[Evaluation]:
        logits_NE, win_N, expected_N = self.engine.submit(encode_batch(boards)).result()
        return [
            (
                policy_dict(logits_NE[i], b.get_legal_moves_flat(), self.policy_temperature),
                shaped_value(float(win_N[i]), float(expected_N[i]),
                             self.margin_utility_lambda, self.margin_utility_k),
            )
            for i, b in enumerate(boards)
        ]

    def close(self) -> None:
        pass


@dataclass
class BoxesSearchResult:
    """Wraps the C++ tree; `gameplay.play_game` reads `.tree` for move metrics."""

    tree: Any

    @property
    def N(self) -> int:  # noqa: N802
        return int(self.tree.get_root_visit_count())

    @property
    def Q(self) -> float:  # noqa: N802
        return float(self.tree.get_root_q_value())


class BoxesMCTSAgent(Agent):
    """C++ MCTS over `BoxesMCTSTree` with a BoxesNet evaluator."""

    def __init__(
        self,
        evaluator: BoxesLeafEvaluator | BoxesEngineEvaluator,
        num_simulations: int = 64,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        temperature_cutoff: int | None = None,
        add_noise: bool = False,
        noise_alpha: float = 0.3,
        noise_weight: float = 0.25,
        leaf_batch_size: int = 8,
        resign_threshold: float = 0.0,
        resign_consec_turns: int = 5,
        pcr_sims: list[int] | None = None,
        pcr_probs: list[float] | None = None,
        forced_collapse: bool = True,
    ) -> None:
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.forced_collapse = forced_collapse
        self.temperature = temperature
        self.temperature_cutoff = temperature_cutoff
        self.leaf_batch_size = leaf_batch_size
        self.resign_threshold = resign_threshold
        self.resign_consec_turns = resign_consec_turns
        self._consec_below = 0
        self.last_search_result: BoxesSearchResult | None = None
        self.cpp_config = alpha_go_cpp.MCTSConfig()
        self.cpp_config.c_puct = c_puct
        self.cpp_config.dirichlet_alpha = noise_alpha if add_noise else 0.0
        self.cpp_config.dirichlet_weight = noise_weight
        self.cpp_config.temperature = temperature
        if pcr_sims is not None and pcr_probs is not None:
            assert len(pcr_sims) == len(pcr_probs) and abs(sum(pcr_probs) - 1.0) < 1e-4
            self.cpp_config.pcr_sims = list(pcr_sims)
            self.cpp_config.pcr_probs = list(pcr_probs)

    @property
    def checkpoint_path(self) -> str | None:
        return self.evaluator.checkpoint_path

    def start_game(self, board_size: int) -> None:
        self._consec_below = 0

    def search(self, board: Any) -> BoxesSearchResult:
        """Search from a board or a BoxesSearchState (collapsed position)."""
        if isinstance(board, alpha_go_cpp.BoxesSearchState):
            tree = alpha_go_cpp.BoxesSearchMCTSTree(board, self.cpp_config)
        else:
            tree = alpha_go_cpp.BoxesMCTSTree(board, self.cpp_config)
        if self.leaf_batch_size > 0:
            tree.run_simulations_batched(
                self.num_simulations, self.leaf_batch_size, self.evaluator.batch_evaluate
            )
        else:
            tree.run_simulations(self.num_simulations, self.evaluator.evaluate)
        return BoxesSearchResult(tree)

    def select_move(self, board: Any, seed: int) -> tuple[int, int]:
        torch.manual_seed(seed)
        self.last_search_result = None
        if self.forced_collapse:
            state = alpha_go_cpp.BoxesSearchState(board)
            if state.prefix():  # a forced capture: play it, no search needed
                row, col = board.row_col(state.prefix()[0])
                return int(row), int(col)
            result = self.search(state)
        else:
            result = self.search(board)
        self.last_search_result = result
        if self.resign_threshold > 0:
            losing = 1.0 - result.Q < self.resign_threshold
            self._consec_below = self._consec_below + 1 if losing else 0
            if self._consec_below >= self.resign_consec_turns:
                return (-2, -2)  # RESIGN
        cutoff = self.temperature_cutoff
        past_cutoff = cutoff is not None and board.move_count() >= cutoff
        edge = result.tree.select_action(0.0 if past_cutoff else self.temperature)
        row, col = board.row_col(edge)
        return int(row), int(col)

    def close(self) -> None:
        self.evaluator.close()


def register_boxes_mcts_agent(
    name: str,
    checkpoint: str | Path | None,
    rows: int,
    cols: int | None = None,
    device: str | None = None,
    engine: PlaneBatchedEngine | None = None,
    net_kwargs: dict[str, Any] | None = None,
    evaluator_kwargs: dict[str, Any] | None = None,
    **mcts_kwargs: Any,
) -> str:
    """Register a no-arg agent class under `name` (the registry instantiates agents by name).

    With `checkpoint` the net is loaded from it; without, a fresh `BoxesNet(rows, cols,
    **net_kwargs)` is used (random play, for smoke tests and bootstrap). With `engine`,
    leaves go through the shared engine (its model is used); otherwise each agent owns a
    `BoxesLeafEvaluator`.
    """

    class _Agent(BoxesMCTSAgent):
        def __init__(self) -> None:
            if engine is not None:
                evaluator: BoxesLeafEvaluator | BoxesEngineEvaluator = BoxesEngineEvaluator(
                    engine, checkpoint_path=str(checkpoint) if checkpoint else None,
                    **(evaluator_kwargs or {}),
                )
            else:
                dev = pick_device(device)
                model = (load_boxes_net(checkpoint, dev) if checkpoint
                         else BoxesNet(rows, cols, **(net_kwargs or {})))
                evaluator = BoxesLeafEvaluator(
                    model, dev, checkpoint_path=str(checkpoint) if checkpoint else None,
                    **(evaluator_kwargs or {}),
                )
            super().__init__(evaluator, **mcts_kwargs)

    _Agent.__name__ = name
    register_agent(name)(_Agent)
    return name


# Random-weight 3x3 agent so the self-play / metrics pipeline can be exercised without a
# checkpoint (and so `self_play --black boxes-mcts-untrained-3x3` works out of the box).
register_boxes_mcts_agent(
    "boxes-mcts-untrained-3x3", None, 3, net_kwargs=dict(channels=16, n_blocks=2),
    num_simulations=32, device="cpu",
)
