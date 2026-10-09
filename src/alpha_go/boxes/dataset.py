"""Training positions from Boxes games saved by `alpha_go.gameplay.save_game_data`.

Each NPZ holds one game: `boards` (n, H, W) lattice grids before each move, `to_play`
(n,) sides to move, `moves` (n, 2) lattice coordinates, `result` ("B+3.0" = player 1 won
by 3), and, when the agents searched, `mcts_visits` (n, E) with `mcts_temperatures` and
`mcts_root_values`. A sample is the encoded planes, the policy target (visit distribution
`N^(1/tau)`, or a label-smoothed one-hot of the played edge without search data), the
final margin for the side to move, and the root value for the z/Q mix option.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from alpha_go.boxes.encode import encode_grid, lattice_rows_cols
from alpha_go.boxes.rules import geometry
from alpha_go.self_play import parse_score_from_result


class BoxesDataset(Dataset[dict[str, Any]]):
    def __init__(self, data_dirs: list[str | Path], smooth_eps: float = 0.1,
                 games: list[dict[str, Any]] | None = None,
                 like: BoxesDataset | None = None) -> None:
        self.smooth_eps = smooth_eps
        paths = [p for d in data_dirs for p in sorted(Path(d).rglob("*.npz"))]
        self.games = [dict(np.load(p)) for p in paths] if games is None else games
        assert self.games or like is not None, f"no .npz games under {data_dirs}"
        if self.games:
            self.rows, self.cols = lattice_rows_cols(self.games[0]["boards"].shape)
        else:
            assert like is not None
            self.rows, self.cols = like.rows, like.cols
        self.geo = geometry(self.rows, self.cols)
        self.cumsum = np.cumsum([0] + [int(g["num_moves"]) for g in self.games])
        self.num_with_mcts = sum("mcts_visits" in g for g in self.games)

    def split(self, val_fraction: float, seed: int = 0) -> tuple[BoxesDataset, BoxesDataset]:
        """Held-out split by game (positions of one game never straddle the two sets)."""
        order = np.random.default_rng(seed).permutation(len(self.games))
        n_val = int(round(val_fraction * len(self.games)))
        val = [self.games[i] for i in order[:n_val]]
        train = [self.games[i] for i in order[n_val:]]
        return (BoxesDataset([], self.smooth_eps, games=train, like=self),
                BoxesDataset([], self.smooth_eps, games=val, like=self))

    def footprint_bytes(self) -> int:
        return sum(
            sum(int(v.nbytes) for v in g.values() if hasattr(v, "nbytes")) for g in self.games
        )

    def __len__(self) -> int:
        return int(self.cumsum[-1])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        game_idx = int(np.searchsorted(self.cumsum[1:], idx, side="right"))
        local = idx - int(self.cumsum[game_idx])
        game = self.games[game_idx]
        to_play = int(game["to_play"][local]) + 1
        planes = encode_grid(game["boards"][local], to_play)
        score_p1 = parse_score_from_result(str(game["result"]))
        margin = int(score_p1) if to_play == 1 else -int(score_p1)
        if "mcts_visits" in game:
            visits = game["mcts_visits"][local].astype(np.float32)
            temperature = float(game["mcts_temperatures"][local])
            policy = np.zeros_like(visits)
            if temperature == 0:
                policy[int(np.argmax(visits))] = 1.0
            else:
                powered = np.power(visits, 1.0 / temperature)
                policy = powered / max(powered.sum(), 1e-8)
            root_value = float(game["mcts_root_values"][local])
        else:
            r, c = (int(x) for x in game["moves"][local])
            num_edges = self.geo.num_edges
            policy = np.full(num_edges, self.smooth_eps / num_edges, dtype=np.float32)
            policy[int(self.geo.lattice_edge[r, c])] += 1.0 - self.smooth_eps
            root_value = float(margin > 0) + 0.5 * float(margin == 0)
        return {
            "planes": torch.from_numpy(planes),
            "policy": torch.from_numpy(policy.astype(np.float32)),
            "margin": torch.tensor(margin, dtype=torch.long),
            "win": torch.tensor(float(margin > 0) + 0.5 * float(margin == 0)),
            "root_value": torch.tensor(root_value, dtype=torch.float32),
            "has_mcts": "mcts_visits" in game,
        }
