"""Compare policy/value routing and search settings on frozen development positions.

Only roots and selected children receive bounded exact solves. Metrics use the same
known positions across every requested case; unknown labels never count as mistakes.
This is a development diagnostic, not an independent strength test or equal-time match.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np

from alpha_go.boxes.mixed_evaluator import MixedLeafEvaluator
from alpha_go.boxes.nn_agent import BoxesMCTSAgent, load_boxes_net, pick_device
from alpha_go.boxes.rules import geometry
from alpha_go.boxes.validation import replay_group

MIXES = {
    "reference": ("reference", "reference"),
    "candidate": ("candidate", "candidate"),
    "old-policy-new-value": ("reference", "candidate"),
    "new-policy-old-value": ("candidate", "reference"),
}


def sha256(path: Path) -> str:
    """Hash an input without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Replace a progress artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Position:
    """Replayable position with its source and stable training-family identity."""

    rows: int
    cols: int
    prefix: list[int]
    source: str
    source_sha256: str
    family: str

    def board(self) -> Any:
        """Rebuild score ownership and side to move, including branch prefixes."""
        board = alpha_go_cpp.BoxesBoard(self.rows, self.cols)
        for edge in self.prefix:
            if edge not in board.get_legal_moves_flat():
                raise ValueError(f"Illegal replay edge {edge} in {self.source}")
            board.play_edge(edge)
        return board


def position_key(board: Any) -> tuple[int, int, int]:
    """Deduplicate states without discarding score or side to move."""
    return int(board.edges()), int(board.to_play()), int(board.margin())


def sample_positions(directories: list[Path], rows: int, cols: int, minimum: int,
                     maximum: int, count: int, seed: int) -> list[Position]:
    """Sample unique search-decided roots from sequential NPZ games only."""
    geo = geometry(rows, cols)
    if any(not directory.is_dir() for directory in directories):
        raise ValueError("Every --games source must be an existing directory")
    unique: dict[tuple[int, int, int], Position] = {}
    paths = sorted({p.resolve() for d in directories for p in d.rglob("*.npz")
                    if not p.name.endswith(".tmp.npz")})
    for path in paths:
        with np.load(path, allow_pickle=False) as loaded:
            game = dict(loaded)
        if str(game.get("termination", "")) == "solver":
            continue  # independent labelled positions are not a playable trajectory
        grids = np.asarray(game["boards"])
        if grids.shape[1:] != (2 * rows + 1, 2 * cols + 1):
            raise ValueError(f"Wrong board dimensions in {path}")
        moves = np.asarray(game["moves"]).reshape(-1, 2)
        starts = np.asarray(game.get("start_moves", []), dtype=np.int64).reshape(-1, 2)
        family = replay_group(game, geo).hex()
        source_hash = sha256(path)
        board = alpha_go_cpp.BoxesBoard(rows, cols)
        prefix: list[int] = []
        for row, col in starts:
            edge = int(board.edge_index(int(row), int(col)))
            if edge not in board.get_legal_moves_flat():
                raise ValueError(f"Illegal branch prefix in {path}")
            board.play_edge(edge)
            prefix.append(edge)
        if len(grids) != len(moves):
            raise ValueError(f"Mismatched moves and boards in {path}")
        for index, (row, col) in enumerate(moves):
            if row < 0:
                continue
            if not np.array_equal(grids[index], board.to_numpy()):
                raise ValueError(f"Stored board disagrees with replay in {path} at row {index}")
            if "to_play" in game and int(game["to_play"][index]) != board.player():
                raise ValueError(f"Stored mover disagrees with replay in {path} at row {index}")
            undrawn = geo.num_edges - len(prefix)
            if (minimum <= undrawn <= maximum
                    and not alpha_go_cpp.BoxesSearchState(board).prefix()):
                unique.setdefault(position_key(board), Position(
                    rows, cols, prefix.copy(), str(path), source_hash, family))
            edge = int(board.edge_index(int(row), int(col)))
            if edge not in board.get_legal_moves_flat():
                raise ValueError(f"Illegal recorded move in {path}")
            board.play_edge(edge)
            prefix.append(edge)
    keys = sorted(unique)
    random.Random(seed).shuffle(keys)
    return [unique[key] for key in keys[:count]]


def load_positions(path: Path, rows: int, cols: int, minimum: int,
                   maximum: int) -> list[Position]:
    """Load exactly the frozen sample, rejecting incompatible sampling arguments."""
    positions = [Position(**json.loads(line)) for line in path.read_text().splitlines()
                 if line.strip()]
    seen: set[tuple[int, int, int]] = set()
    for position in positions:
        if (position.rows, position.cols) != (rows, cols):
            raise ValueError("Position-file dimensions do not match --rows/--cols")
        board = position.board()
        if not minimum <= board.num_edges() - board.move_count() <= maximum:
            raise ValueError("Frozen position outside requested undrawn band")
        if alpha_go_cpp.BoxesSearchState(board).prefix():
            raise ValueError("Frozen position has a forced capture")
        key = position_key(board)
        if key in seen:
            raise ValueError("Duplicate state in frozen position file")
        seen.add(key)
    return positions


class ExactLabels:
    """One bounded solver/table and persistent exact-only cache for a board geometry."""

    def __init__(self, rows: int, cols: int, budget: int, table_entries: int,
                 cache: Path | None = None) -> None:
        if budget <= 0:
            raise ValueError("Exact label node budget must be positive")
        self.rows, self.cols, self.budget = rows, cols, budget
        self.full = (1 << geometry(rows, cols).num_edges) - 1
        self.solver = alpha_go_cpp.BoxesSolver(rows, cols, table_entries)
        self.values: dict[str, int] = {}
        self.unknown: set[int] = set()
        self.calls = 0
        self.hits = 0
        self.seconds = 0.0
        if cache is not None:
            document = json.loads(cache.read_text())
            if (document["rows"], document["cols"], document["schema"]) != (rows, cols, 1):
                raise ValueError("Label cache geometry/schema mismatch")
            for key, value in document["remaining_values"].items():
                if type(value) is not int or abs(value) > rows * cols:
                    raise ValueError("Invalid exact remaining value in cache")
                self.values[key] = value
        print(f"Label solver table: {self.solver.table_bytes() / 1e6:.1f} MB; "
              f"budget {budget:,} nodes per solve", flush=True)

    def remaining(self, mask: int) -> int | None:
        """Return an exact remaining margin, or unknown after the finite node budget."""
        key = str(mask)
        if mask == self.full:
            self.values[key] = 0
        if key in self.values:
            self.hits += 1
            return self.values[key]
        if mask in self.unknown:
            return None
        started = time.monotonic()
        value = self.solver.value_within(mask, self.budget)
        self.seconds += time.monotonic() - started
        self.calls += 1
        if value is None:
            self.unknown.add(mask)
            return None
        self.values[key] = int(value)
        return int(value)

    def final(self, board: Any, player: int) -> int | None:
        """Exact final score margin in a fixed player's perspective."""
        remaining = self.remaining(int(board.edges()))
        if remaining is None:
            return None
        final = int(board.margin()) + remaining
        return final if int(board.to_play()) == player else -final

    def save(self, path: Path) -> None:
        """Persist completed solves; failed solves may be retried in a later run."""
        write_json(path, dict(schema=1, rows=self.rows, cols=self.cols,
                              remaining_values=self.values, calls=self.calls,
                              cache_hits=self.hits, unknown_masks=sorted(self.unknown),
                              node_budget=self.budget, seconds=self.seconds))


def selected_final(labels: ExactLabels, board: Any, edge: int) -> int | None:
    """Solve one chosen child, keeping the root mover's perspective on extra turns."""
    child = board.copy()
    if edge not in child.get_legal_moves_flat():
        raise ValueError(f"Selected illegal edge {edge}")
    child.play_edge(edge)
    return labels.final(child, int(board.to_play()))


def summarize(records: list[dict[str, Any]], cases: list[str], total: int
              ) -> dict[str, Any]:
    """Compare all cases only where root and selected children are commonly known."""
    common = set(range(total))
    for case in cases:
        common &= {int(r["position"]) for r in records if r["case"] == case
                   and all(r[key] is not None for key in
                           ("root_final", "policy_final", "search_final"))}
    result: dict[str, Any] = {
        "sampled": total, "common_known": len(common), "excluded": total - len(common),
        "common_position_indices": sorted(common),
        "regret_units": "final score-margin points, not boxes captured",
        "cases": {},
    }
    for case in cases:
        all_rows = [r for r in records if r["case"] == case]
        rows = [r for r in all_rows if r["position"] in common]
        winning = [r for r in rows if r["root_final"] > 0]
        metrics: dict[str, Any] = {
            "common_known": len(rows), "winning_roots": len(winning),
            "total_search_seconds": sum(r["search_seconds"] for r in all_rows),
            "root_known": sum(r["root_final"] is not None for r in all_rows),
            "policy_child_known": sum(r["policy_final"] is not None for r in all_rows),
            "search_child_known": sum(r["search_final"] is not None for r in all_rows),
        }
        for choice in ("policy", "search"):
            key = f"{choice}_final"
            regrets = [r["root_final"] - r[key] for r in rows]
            if any(regret < 0 for regret in regrets):
                raise ValueError("Exact child value exceeds exact minimax root value")
            metrics.update({
                f"{choice}_optimal": (sum(v == 0 for v in regrets) / len(rows)
                                      if rows else None),
                f"{choice}_margin_lost": sum(regrets) / len(rows) if rows else None,
                f"{choice}_wins_preserved": sum(r[key] > 0 for r in winning),
                f"{choice}_win_to_loss": sum(r[key] < 0 for r in winning),
                f"{choice}_win_to_draw": sum(r[key] == 0 for r in winning),
            })
        result["cases"][case] = metrics
    return result


def parser() -> argparse.ArgumentParser:
    """Arguments with a small four-case default; bigger sweeps are explicit."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reference", type=Path, required=True)
    result.add_argument("--candidate", type=Path, required=True)
    sources = result.add_mutually_exclusive_group(required=True)
    sources.add_argument("--games", type=Path, nargs="+")
    sources.add_argument("--positions-file", type=Path)
    result.add_argument("--output", type=Path, required=True, help="fresh output directory")
    result.add_argument("--rows", type=int, default=5)
    result.add_argument("--cols", type=int, default=None)
    result.add_argument("--min-undrawn", type=int, default=35)
    result.add_argument("--max-undrawn", type=int, default=38)
    result.add_argument("--positions", type=int, default=100,
                        help="number to sample; positions-file reuses every saved position")
    result.add_argument("--simulations", type=int, nargs="+", default=[400])
    result.add_argument("--lambdas", type=float, nargs="+", default=[0.5])
    result.add_argument("--mixes", choices=list(MIXES), nargs="+", default=list(MIXES))
    result.add_argument("--label-budget", type=int, default=20_000_000)
    result.add_argument("--label-table-entries", type=int, default=1 << 22)
    result.add_argument("--label-cache", type=Path, help="read-only cache from an earlier probe")
    result.add_argument("--solver", type=int, default=30)
    result.add_argument("--solver-budget", type=int, default=50_000)
    result.add_argument("--solver-table-entries", type=int, default=1 << 20)
    result.add_argument("--c-puct", type=float, default=1.5)
    result.add_argument("--leaf-batch-size", type=int, default=16)
    result.add_argument("--policy-temperature", type=float, default=1.0)
    result.add_argument("--margin-utility-k", type=float, default=6.0)
    result.add_argument("--prove-terminals", type=int, choices=[0, 1], default=1)
    result.add_argument("--shape-terminals", type=int, choices=[0, 1], default=1)
    result.add_argument("--seed", type=int, default=20261014)
    result.add_argument("--cpu", action="store_true")
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the fixed-position diagnostic and save auditable, resumable label artifacts."""
    args = parser().parse_args(argv)
    args.cols = args.cols or args.rows
    edge_count = geometry(args.rows, args.cols).num_edges
    if (not 0 <= args.solver < args.min_undrawn <= args.max_undrawn <= edge_count
            or args.positions <= 0 or args.label_budget <= 0 or args.solver_budget <= 0
            or min(args.simulations) <= 0 or min(args.lambdas) < 0
            or args.label_table_entries <= 0 or args.solver_table_entries <= 0
            or args.leaf_batch_size < 0 or args.c_puct <= 0):
        raise ValueError("Invalid budgets/band: probe roots must be above the search solver")
    if args.output.exists():
        raise ValueError("--output must be a fresh directory; reuse positions/cache explicitly")
    device = pick_device("cpu" if args.cpu else None)
    if not args.cpu and device.type != "cuda":
        raise ValueError("CUDA unavailable; pass --cpu for a CPU probe")
    args.output.mkdir(parents=True)
    config: dict[str, Any] = dict(
        complete=False, arguments=vars(args), device=str(device),
        checkpoint_sha256={"reference": sha256(args.reference),
                           "candidate": sha256(args.candidate)},
        code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                     ("search_probe.py", "mixed_evaluator.py", "nn_agent.py", "model.py")},
        cpp_sha256=sha256(Path(alpha_go_cpp.__file__)),
        scope="development diagnostic; source games may have been used for training",
        timing="mixed evaluators require two forwards; simulation counts are not equal time",
        randomness="noise disabled and temperature zero; C++ RNG is not controlled by --seed",
    )
    write_json(args.output / "summary.json", config)
    positions = (load_positions(args.positions_file, args.rows, args.cols, args.min_undrawn,
                                args.max_undrawn) if args.positions_file else
                 sample_positions(args.games, args.rows, args.cols, args.min_undrawn,
                                  args.max_undrawn, args.positions, args.seed))
    if not positions:
        raise ValueError("No search-decided positions found in the requested band")
    frozen = args.output / "positions.jsonl"
    frozen.write_text("".join(json.dumps(asdict(p)) + "\n" for p in positions))
    config.update(positions_sha256=sha256(frozen),
                  source_families=len({p.family for p in positions}))
    print(f"Frozen {len(positions)} positions from {config['source_families']} source families",
          flush=True)
    models = {name: load_boxes_net(path, device) for name, path in
              (("reference", args.reference), ("candidate", args.candidate))}
    if any((m.rows, m.cols) != (args.rows, args.cols) for m in models.values()):
        raise ValueError("Checkpoint geometry does not match the requested board")
    labels = ExactLabels(args.rows, args.cols, args.label_budget, args.label_table_entries,
                         args.label_cache)
    roots: list[int | None] = []
    for i, position in enumerate(positions):
        board = position.board()
        roots.append(labels.final(board, int(board.to_play())))
        labels.save(args.output / "labels.json")
        print(f"Roots labelled {i + 1}/{len(positions)}; "
              f"{sum(v is not None for v in roots)} exact", flush=True)
    records: list[dict[str, Any]] = []
    cases: list[str] = []
    with (args.output / "positions.csv").open("w", newline="") as output:
        writer: csv.DictWriter[str] | None = None
        for mix, simulations, weight in itertools.product(
                dict.fromkeys(args.mixes), dict.fromkeys(args.simulations),
                dict.fromkeys(args.lambdas)):
            case = f"{mix}:s{simulations}:lambda{weight}"
            cases.append(case)
            policy_model, value_model = (models[name] for name in MIXES[mix])
            evaluator = MixedLeafEvaluator(policy_model, value_model, device,
                                            args.policy_temperature, weight,
                                            args.margin_utility_k)
            agent = BoxesMCTSAgent(
                evaluator, num_simulations=simulations, temperature=0, add_noise=False,
                c_puct=args.c_puct, leaf_batch_size=args.leaf_batch_size,
                solver_max_undrawn=args.solver, solver_node_budget=args.solver_budget,
                solver_table_entries=args.solver_table_entries,
                prove_terminals=bool(args.prove_terminals),
                shape_terminals=bool(args.shape_terminals))
            for i, position in enumerate(positions):
                board = position.board()
                started = time.monotonic()
                priors, _ = evaluator.evaluate(board)
                policy_edge = max(priors, key=lambda edge: priors[edge])
                policy_seconds = time.monotonic() - started
                started = time.monotonic()
                agent.start_game(args.rows)
                row, col = agent.select_move(board.copy(), args.seed + i)
                search_seconds = time.monotonic() - started
                search_edge = int(board.edge_index(row, col))
                record = dict(
                    position=i, case=case, mix=mix, simulations=simulations,
                    margin_utility_lambda=weight, undrawn=edge_count - len(position.prefix),
                    source_family=position.family, root_final=roots[i],
                    policy_edge=policy_edge, search_edge=search_edge,
                    policy_final=selected_final(labels, board, policy_edge),
                    search_final=selected_final(labels, board, search_edge),
                    policy_seconds=policy_seconds, search_seconds=search_seconds)
                records.append(record)
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
                output.flush()
                labels.save(args.output / "labels.json")
                print(f"{case}: {i + 1}/{len(positions)}", flush=True)
            agent.close()
            config["completed_cases"] = cases.copy()
            write_json(args.output / "summary.json", config)
    config.update(summarize(records, cases, len(positions)), complete=True)
    write_json(args.output / "summary.json", config)
    print(json.dumps(config["cases"], indent=2), flush=True)


if __name__ == "__main__":
    main()
