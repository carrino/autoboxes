"""Boxes MCTS agents backed by BoxesNet.

`BoxesLeafEvaluator` owns a net and evaluates a batch of leaves in one forward pass (used by
the leaf-parallel C++ search inside one game thread). `BoxesEngineEvaluator` submits leaves
to a shared `PlaneBatchedEngine` so many game threads share forwards. Both return, per
board, a policy over its legal edges and the side-to-move win probability (optionally
shaped by the margin-utility term from PLAN.md decision 4).
"""
# ruff: noqa: N803, N806  (dimension-suffixed tensor names)
from __future__ import annotations

import argparse
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
from alpha_go.boxes.rules import BoxesGeometry, geometry

Evaluation = tuple[dict[int, float], float]


def pick_device(device: str | None = None) -> torch.device:
    return torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))


def save_boxes_net(model: BoxesNet, path: str | Path, **extra: Any) -> None:
    """Checkpoint with enough config to rebuild the net."""
    config = dict(
        rows=model.rows, cols=model.cols, channels=model.channels, n_blocks=model.n_blocks,
        value_hidden=model.value_hidden, norm_type=model.norm_type, use_se=model.use_se,
        features=model.features,
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


def add_search_flags(parser: argparse.ArgumentParser) -> None:
    """The search knobs every Boxes CLI exposes, with the loop's original values as defaults.

    experiments/2026-10-09_01-05-boxes-3x3-search-autoresearch tunes them against the exact
    oracle; `search_flags` turns the parsed values into the agent and evaluator kwargs.
    """
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument("--leaf_batch_size", type=int, default=16,
                        help="leaves evaluated per search step (smaller: less virtual loss)")
    parser.add_argument("--policy_temperature", type=float, default=1.0,
                        help="prior = softmax(logits / T) over legal edges; T < 1 sharpens")
    parser.add_argument("--margin_utility_lambda", type=float, default=0.0,
                        help="leaf value = P(win) + lambda * tanh(E[margin] / k)")
    parser.add_argument("--margin_utility_k", type=float, default=6.0)


def search_flags(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """(MCTS kwargs, evaluator kwargs) from `add_search_flags` arguments."""
    return (
        dict(c_puct=args.c_puct, leaf_batch_size=args.leaf_batch_size),
        dict(policy_temperature=args.policy_temperature,
             margin_utility_lambda=args.margin_utility_lambda,
             margin_utility_k=args.margin_utility_k),
    )


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
        planes_BKHW = torch.from_numpy(encode_batch(boards, self.model.features)).to(self.device)
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
        planes_NKHW = encode_batch(boards, self.engine.model.features)
        logits_NE, win_N, expected_N = self.engine.submit(planes_NKHW).result()
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


def optimal_edges(solver: Any, board: Any, geo: BoxesGeometry) -> list[int]:
    """Every legal edge whose exact child value equals the position's value (C++ solver).

    A capturing edge keeps the move: child value = boxes gained + value after; otherwise the
    opponent moves: child value = -value after.
    """
    mask = int(board.edges())
    value = int(solver.value(mask))

    def child(e: int) -> int:
        after = mask | 1 << e
        gained = sum(all(after >> s & 1 for s in geo.box_edges[b]) for b in geo.edge_boxes[e])
        rest = int(solver.value(after))
        return gained + rest if gained else -rest

    return [e for e in board.get_legal_moves_flat() if child(e) == value]


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


@dataclass
class BoxesSolvedResult:
    """A solver-played move, in the shape `gameplay.play_game` reads from a search tree: one
    visit per optimal edge, so the policy target is uniform over the optimal set instead of
    a one-hot of an arbitrary optimal edge; Q is the exact outcome."""

    optimal: list[int]
    final_margin: int  # exact final margin for the side to move (score so far + remaining)

    @property
    def tree(self) -> BoxesSolvedResult:
        return self

    @property
    def N(self) -> int:  # noqa: N802
        return len(self.optimal)

    @property
    def Q(self) -> float:  # noqa: N802
        return float(self.get_root_q_value())

    def win(self) -> float:
        return 1.0 if self.final_margin > 0 else 0.5 if self.final_margin == 0 else 0.0

    def get_child_visit_counts(self) -> dict[int, int]:
        return {e: 1 for e in self.optimal}

    def get_child_q_values(self) -> dict[int, float]:
        return {e: self.win() for e in self.optimal}

    def get_root_policy_priors(self) -> dict[int, float]:
        return {e: 1.0 / len(self.optimal) for e in self.optimal}

    def get_root_q_value(self) -> float:  # from the parent's (opponent's) perspective, as a tree
        return 1.0 - self.win()

    def get_root_visit_count(self) -> int:
        return len(self.optimal)


class BoxesMCTSAgent(Agent):
    """C++ MCTS over `BoxesMCTSTree` with a BoxesNet evaluator."""

    _solver_announced = False

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
        solver_max_undrawn: int = 0,
        solver_node_budget: int = 20_000,
        solver_table_entries: int = 1 << 20,
        merge_equivalent: bool = False,
    ) -> None:
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.forced_collapse = forced_collapse
        # Exact endgame solver (PLAN.md §4.1): positions with <= solver_max_undrawn undrawn
        # edges that it settles within solver_node_budget nodes are terminal for the search.
        self.solver_max_undrawn = solver_max_undrawn
        self.solver_node_budget = solver_node_budget
        self.solver_table_entries = solver_table_entries
        self.solver: Any = None  # built on first use, one per agent (tables are per thread)
        self.geo: Any = None  # geometry tables for the solver's optimal-edge sets
        # One action per independent chain or loop in quiet positions (BoxesZero's
        # equivalent edges); visits and policy targets land on the representative edge.
        self.merge_equivalent = merge_equivalent
        self.temperature = temperature
        self.temperature_cutoff = temperature_cutoff
        self.leaf_batch_size = leaf_batch_size
        self.resign_threshold = resign_threshold
        self.resign_consec_turns = resign_consec_turns
        self._consec_below = 0
        self.last_search_result: BoxesSearchResult | BoxesSolvedResult | None = None
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

    def search_state(self, board: Any) -> Any:
        """Collapsed position, solver-terminated when a solver is configured."""
        if self.solver_max_undrawn <= 0:
            return alpha_go_cpp.BoxesSearchState(board, None, 0, 0, self.merge_equivalent)
        if self.solver is None:
            self.solver = alpha_go_cpp.BoxesSolver(int(board.rows()), int(board.cols()),
                                                   self.solver_table_entries)
            self.geo = geometry(int(board.rows()), int(board.cols()))
            if not BoxesMCTSAgent._solver_announced:  # footprint once per process
                BoxesMCTSAgent._solver_announced = True
                print(f"BoxesSolver: max_undrawn={self.solver_max_undrawn} "
                      f"node_budget={self.solver_node_budget} "
                      f"table={self.solver.table_entries()} entries = "
                      f"{self.solver.table_bytes() / 1e6:.1f} MB per agent", flush=True)
        return alpha_go_cpp.BoxesSearchState(board, self.solver, self.solver_max_undrawn,
                                             self.solver_node_budget, self.merge_equivalent)

    def select_move(self, board: Any, seed: int) -> tuple[int, int]:
        torch.manual_seed(seed)
        self.last_search_result = None
        if self.forced_collapse:
            state = self.search_state(board)
            if state.prefix():  # a forced capture: play it, no search needed
                row, col = board.row_col(state.prefix()[0])
                return int(row), int(col)
            if state.solved():  # exact endgame: one of the optimal edges, no search needed
                solved = BoxesSolvedResult(optimal_edges(self.solver, board, self.geo),
                                           int(state.solved_margin()))
                self.last_search_result = solved
                edge = solved.optimal[np.random.default_rng(seed).integers(len(solved.optimal))]
                row, col = board.row_col(edge)
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
