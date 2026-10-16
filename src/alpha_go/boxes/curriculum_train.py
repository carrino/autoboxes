"""Conservative root-policy training with a fixed, family-excluded development monitor.

Prepared ``curriculum_labels`` bundles supply complete exact root policies and every
legal child's final margin. Distinct root rows receive one pass per epoch, with at
most 25% searched opening rehearsal. This development selector is not an arena gate
or an untouched test: the final full-game comparison remains necessary.
"""
# ruff: noqa: N806
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from alpha_go.boxes.dataset import BoxesDataset, undrawn_edges
from alpha_go.boxes.nn_agent import load_boxes_net, pick_device
from alpha_go.boxes.policy_ab_data import canonical_opening, game_opening, write_json
from alpha_go.boxes.policy_target_ab import (
    IndexedPositions,
    TrainSettings,
    evaluate,
    sha256,
    train_arm,
)
from alpha_go.boxes.rules import BoxesBoard, BoxesGeometry
from alpha_go.boxes.validation import replay_group

State = tuple[int, int, int]


@dataclass(frozen=True)
class CurriculumSettings:
    """Small learning rate and fixed passes over distinct accumulated root positions."""

    epochs: int = 12
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 20261016
    max_undrawn: int = 34
    rehearsal_fraction: float = 0.25
    rehearsal_min_undrawn: int = 39
    improvement: float = 0.05

    def validate(self) -> None:
        """Reject invalid sampling fractions and training budgets."""
        TrainSettings(self.epochs, self.batch_size, self.lr, self.weight_decay,
                      self.seed).validate()
        assert self.max_undrawn > 0 and self.rehearsal_min_undrawn > 0
        assert 0 <= self.rehearsal_fraction <= 0.25
        assert math.isfinite(self.improvement) and self.improvement > 0


def state_key(game: dict[str, Any], row: int, geo: BoxesGeometry) -> State:
    """Identify a root including accumulated scores and the actual mover."""
    grid = game["boards"][row]
    edges = sum(1 << edge for edge, (r, c) in enumerate(geo.edge_rc) if grid[r, c] == 1)
    owners = grid[1::2, 1::2]
    score = int((owners == 1).sum() - (owners == 2).sum())
    return edges, int(game["to_play"][row]), score


def subset(dataset: BoxesDataset, masks: list[np.ndarray[Any, Any]]) -> BoxesDataset:
    """Select existing rows with masks; never replicate or oversample positions."""
    games = [dict(game, train_mask=mask) for game, mask in zip(dataset.games, masks)
             if mask.any()]
    return BoxesDataset([], games=games, like=dataset)


def load_roots(directories: list[Path], features: str
               ) -> tuple[BoxesDataset, dict[State, dict[str, Any]], dict[str, str]]:
    """Read exact roots and verify complete child margins against NPZ policy targets."""
    assert directories and all(path.is_dir() for path in directories), "missing root directory"
    directories = sorted({path.resolve() for path in directories})
    paths = sorted({path for folder in directories for path in folder.rglob("*.npz")})
    assert paths, "no prepared root NPZs"
    manifests = sorted({folder.parent / "roots.jsonl" for folder in directories})
    assert all(path.is_file() for path in manifests), "root directory lacks ../roots.jsonl"
    dataset = BoxesDataset([str(p) for p in directories], features=features)
    labels: dict[State, dict[str, Any]] = {}
    for manifest in manifests:
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            assert (record["rows"], record["cols"]) == (dataset.rows, dataset.cols)
            board = BoxesBoard(dataset.rows, dataset.cols)
            for edge in record["prefix"]:
                assert board.play_edge(int(edge)), "illegal labelled root prefix"
            key = board.edges, board.player(), round(board.score())
            children = {int(edge): int(value) for edge, value in record["child_margins"].items()}
            assert set(children) == set(board.get_legal_moves_flat()), "incomplete child labels"
            assert max(children.values()) == record["root_margin"], "inconsistent root margin"
            record = dict(record, child_margins=children,
                          opening=canonical_opening(record["prefix"], dataset.geo))
            if key in labels:
                assert labels[key]["child_margins"] == children, "conflicting exact labels"
                labels[key]["families"].add(record["family"])
                labels[key]["openings"].add(record["opening"])
            else:
                labels[key] = dict(record, families={record["family"]},
                                   openings={record["opening"]})
    for game, indices in zip(dataset.games, dataset.position_indices):
        assert game["boards"].shape[1:] == dataset.geo.lattice_shape
        for row in indices:
            assert bool(game["policy_ab_root_mask"][row]), "non-root row in exact root dataset"
            assert bool(game.get("policy_mask", np.ones(len(game["boards"]), bool))[row])
            key = state_key(game, int(row), dataset.geo)
            record = labels[key]
            target = np.asarray(game["mcts_visits"][row])
            assert np.isfinite(target).all() and (target >= 0).all()
            optimal = {edge for edge, margin in record["child_margins"].items()
                       if margin == record["root_margin"]}
            assert set(np.flatnonzero(target)) == optimal, "exact policy/child margins disagree"
            assert np.all(target[list(optimal)] == target[list(optimal)][0]), (
                "exact root policy must be uniform over the optimal set")
            sign = 1 if key[1] == 0 else -1
            assert sign * int(game["exact_margin_p1"][row]) == record["root_margin"]
            assert replay_group(game, dataset.geo).hex() in record["families"], (
                "root lineage/label family disagree")
    return dataset, labels, {str(path): sha256(path) for path in paths + manifests}


def select_roots(dataset: BoxesDataset, labels: dict[State, dict[str, Any]],
                 *, maximum: int | None = None, blocked_families: set[str] | None = None,
                 blocked_openings: set[str] | None = None,
                 blocked_states: set[State] | None = None) -> BoxesDataset:
    """Deduplicate root states and exclude every fixed-monitor family and opening."""
    seen = set(blocked_states or ())
    masks = []
    for game, indices in zip(dataset.games, dataset.position_indices):
        mask = np.zeros(len(game["boards"]), bool)
        undrawn = undrawn_edges(game["boards"], dataset.geo.num_edges)
        family = replay_group(game, dataset.geo).hex()
        for row in indices:
            key = state_key(game, int(row), dataset.geo)
            if (maximum is not None and undrawn[row] > maximum
                    or family in (blocked_families or set())
                    or labels[key]["openings"] & (blocked_openings or set()) or key in seen):
                continue
            mask[row] = True
            seen.add(key)
        masks.append(mask)
    return subset(dataset, masks)


def root_keys(dataset: BoxesDataset) -> set[State]:
    """The exact score-aware states retained by a dataset."""
    return {state_key(game, int(row), dataset.geo)
            for game, rows in zip(dataset.games, dataset.position_indices) for row in rows}


def rehearsal(dataset: BoxesDataset, directories: list[Path], settings: CurriculumSettings,
              excluded_families: set[str], excluded_openings: set[str],
              excluded_states: set[State]) -> tuple[BoxesDataset, dict[str, int]]:
    """Select a deterministic, distinct subset of searched opening rows for retention."""
    maximum = math.floor(len(dataset) * settings.rehearsal_fraction /
                         (1 - settings.rehearsal_fraction))
    if not directories or maximum == 0:
        return dataset, dict(eligible=0, selected=0, families=0)
    assert all(p.is_dir() for p in directories), "missing rehearsal directory"
    for path in directories:
        assert "5x5-mid" not in path.resolve().parts, "diagnostic data cannot be rehearsal"
    source = BoxesDataset(sorted({str(p.resolve()) for p in directories}),
                          min_undrawn=settings.rehearsal_min_undrawn,
                          features=dataset.features)
    assert (source.rows, source.cols) == (dataset.rows, dataset.cols)
    eligible: list[tuple[int, int]] = []
    seen = set(excluded_states)
    for index, (game, rows) in enumerate(zip(source.games, source.position_indices)):
        if (str(game.get("termination", "")) == "solver"
                or replay_group(game, source.geo).hex() in excluded_families
                or game_opening(game, source.geo) in excluded_openings):
            continue
        visits = game.get("mcts_visits")
        if visits is None:
            continue
        policy_mask = game.get("policy_mask", np.ones(len(game["boards"]), bool))
        for row in rows:
            key = state_key(game, int(row), source.geo)
            if not policy_mask[row] or not visits[row].sum() > 0 or key in seen:
                continue
            seen.add(key)
            eligible.append((index, int(row)))
    order = np.random.default_rng(settings.seed).permutation(len(eligible))[:maximum]
    masks = [np.zeros(len(game["boards"]), bool) for game in source.games]
    for position in order:
        index, row = eligible[int(position)]
        masks[index][row] = True
    selected = subset(source, masks)
    # Rehearsal is never counted as a newly exact root in the training evaluator.
    games = [dict(game, policy_ab_root_mask=np.zeros(len(game["boards"]), bool))
             for game in selected.games]
    combined = BoxesDataset([], games=dataset.games + games, like=dataset)
    return combined, dict(eligible=len(eligible), selected=len(selected),
                          families=len({replay_group(g, source.geo) for g in games}))


@torch.no_grad()
def evaluate_roots(model: Any, dataset: BoxesDataset, labels: dict[State, dict[str, Any]],
                   batch_size: int, device: torch.device) -> dict[str, Any]:
    """Score legal policy choices by exact regret and preservation of winning roots."""
    metrics = evaluate(model, dataset, batch_size, device)
    model.eval()
    keys = [state_key(game, int(row), dataset.geo)
            for game, rows in zip(dataset.games, dataset.position_indices) for row in rows]
    families = [replay_group(game, dataset.geo).hex()
                for game, rows in zip(dataset.games, dataset.position_indices) for _ in rows]
    rows = []
    for batch in DataLoader(IndexedPositions(dataset), batch_size=batch_size, shuffle=False):
        planes = batch["planes"].to(device)
        logits, _ = model(planes)
        legal = planes[:, 0].flatten(1)[:, model.edge_index_E] == 0
        probs = logits.masked_fill(~legal, float("-inf")).softmax(-1).cpu().numpy()
        for index, probability in zip(batch["position"].tolist(), probs):
            key = keys[index]
            record = labels[key]
            edge = int(probability.argmax())
            children = record["child_margins"]
            root_margin = int(record["root_margin"])
            rows.append(dict(edges=key[0], mover=key[1], score_p1=key[2],
                             family=families[index],
                             undrawn=dataset.geo.num_edges-key[0].bit_count(),
                             edge=edge, root_margin=root_margin, child_margin=children[edge],
                             regret=root_margin-children[edge],
                             expected_regret=sum(float(probability[e]) * (root_margin-v)
                                                 for e, v in children.items())))
    winning = [row for row in rows if row["root_margin"] > 0]
    preserved = sum(row["child_margin"] > 0 for row in winning)
    metrics.update(families=len({row["family"] for row in rows}), winning_roots=len(winning),
                   wins_preserved=preserved,
                   win_preservation=preserved / len(winning) if winning else None,
                   mean_regret=float(np.mean([r["regret"] for r in rows])) if rows else None,
                   expected_regret=float(np.mean([r["expected_regret"] for r in rows]))
                   if rows else None, per_position=rows)
    return metrics


def read_bands(model: Any, dataset: BoxesDataset, labels: dict[State, dict[str, Any]],
               settings: CurriculumSettings, device: torch.device) -> dict[str, Any]:
    """Read the full fixed monitor, the current curriculum band, and both named bands."""
    reports = {}
    for name, low, high in [("all", 1, dataset.geo.num_edges),
                            ("current_band", 1, settings.max_undrawn),
                            ("31..34", 31, 34), ("35..36", 35, 36)]:
        masks = []
        for game, indices in zip(dataset.games, dataset.position_indices):
            count = undrawn_edges(game["boards"], dataset.geo.num_edges)
            mask = np.zeros(len(count), bool)
            mask[indices] = (count[indices] >= low) & (count[indices] <= high)
            masks.append(mask)
        reports[name] = evaluate_roots(model, subset(dataset, masks), labels,
                                       settings.batch_size, device)
    return reports


def select_candidate(reference: dict[str, Any], candidate: dict[str, Any],
                     improvement: float = 0.05) -> dict[str, Any]:
    """A development-only selector requiring lower regret without losing more wins."""
    assert reference["positions"] == candidate["positions"]
    assert reference["winning_roots"] == candidate["winning_roots"]
    enough = bool(reference["positions"])
    delta = candidate["mean_regret"] - reference["mean_regret"] if enough else None
    accept = delta is not None and delta <= -improvement and (
        candidate["wins_preserved"] >= reference["wins_preserved"])
    return dict(accepted=accept, regret_delta=delta, required_improvement=improvement,
                wins_preserved_delta=candidate["wins_preserved"]-reference["wins_preserved"],
                scope="development monitor selection; not an arena promotion")


def advance_curriculum(accepted: bool, current: dict[str, Any], maximum: int) -> bool:
    """Move 34 to 36 only after a sufficiently populated accepted monitor read."""
    return bool(accepted and maximum == 34 and current["positions"] >= 32
                and current["families"] >= 8 and current["winning_roots"] >= 8
                and current["win_preservation"] >= 0.9 and current["mean_regret"] <= 1.0)


def run(reference: Path, train_dirs: list[Path], validation_dirs: list[Path], output: Path,
        settings: CurriculumSettings | None = None, *, rehearsal_dirs: list[Path] | None = None,
        cpu: bool = False) -> dict[str, Any]:
    """Train once from the accepted parent, preserving source data and checkpoint bytes."""
    settings = settings or CurriculumSettings()
    settings.validate()
    assert not output.exists(), "output must be a fresh directory"
    device = pick_device("cpu" if cpu else None)
    assert cpu or device.type == "cuda", "CUDA unavailable; --cpu is for tiny smoke tests"
    model = load_boxes_net(reference, device)
    training, train_labels, train_hashes = load_roots(train_dirs, model.features)
    validation, validation_labels, validation_hashes = load_roots(validation_dirs, model.features)
    assert (model.rows, model.cols) == (training.rows, training.cols) == (
        validation.rows, validation.cols)
    validation = select_roots(validation, validation_labels)
    validation_keys = root_keys(validation)
    families = {replay_group(g, validation.geo).hex() for g in validation.games}
    openings = {opening for key in validation_keys
                for opening in validation_labels[key]["openings"]}
    # Exclude the monitor's attempted families too: slow/unsolved examples must not
    # silently enter training when a later label batch succeeds on them.
    for manifest in {p.resolve().parent / "validation-sources.jsonl" for p in validation_dirs}:
        if manifest.is_file():
            validation_hashes[str(manifest)] = sha256(manifest)
            for line in manifest.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    assert (record["rows"], record["cols"]) == (validation.rows, validation.cols)
                    families.add(record["family"])
                    openings.add(canonical_opening(record["prefix"], validation.geo))
    original_count = len(training)
    training = select_roots(training, train_labels, maximum=settings.max_undrawn,
                            blocked_families=families, blocked_openings=openings,
                            blocked_states=validation_keys)
    assert len(training) and len(validation), "need distinct training roots and monitor families"
    combined, rehearsal_counts = rehearsal(training, rehearsal_dirs or [], settings, families,
                                           openings, validation_keys | root_keys(training))
    output.mkdir(parents=True)
    shutil.copyfile(reference, output / "reference.pt")
    frozen_hash = sha256(reference)
    summary: dict[str, Any] = dict(
        schema=1, status="running", settings=asdict(settings), device=str(device),
        reference_sha256=frozen_hash, root_source_sha256={**train_hashes, **validation_hashes},
        training=dict(roots=len(training), original_roots=original_count,
                      families=len({replay_group(g, training.geo) for g in training.games}),
                      rehearsal=rehearsal_counts, positions=len(combined)),
        validation_scope="fixed development monitor; repeated selection is not an untouched test",
        reference_training=evaluate_roots(model, training, train_labels, settings.batch_size,
                                         device),
        reference_validation=read_bands(model, validation, validation_labels, settings, device))
    write_json(output / "summary.json", summary)
    print(f"Curriculum: {len(training)} distinct exact roots, "
          f"{rehearsal_counts['selected']} opening rehearsal rows; "
          f"{len(validation)} fixed monitor roots; RAM {combined.footprint_bytes()/1e6:.1f} MB",
          flush=True)
    del model
    trained = train_arm("candidate", output / "reference.pt", combined, validation, output,
                        TrainSettings(settings.epochs, settings.batch_size, settings.lr,
                                      settings.weight_decay, settings.seed, min_undrawn=0), device)
    model = load_boxes_net(output / "candidate.pt", device)
    summary.update(training_run=trained,
                   candidate_training=evaluate_roots(model, training, train_labels,
                                                     settings.batch_size, device),
                   candidate_validation=read_bands(model, validation, validation_labels,
                                                   settings, device))
    summary["selection"] = select_candidate(summary["reference_validation"]["current_band"],
                                             summary["candidate_validation"]["current_band"],
                                             settings.improvement)
    summary["candidate"] = str((output / "candidate.pt").resolve())
    summary["accepted"] = summary["selection"]["accepted"]
    summary["advance_curriculum"] = advance_curriculum(
        summary["accepted"], summary["candidate_validation"]["current_band"],
        settings.max_undrawn)
    assert sha256(reference) == frozen_hash == sha256(output / "reference.pt")
    assert all(sha256(Path(p)) == digest for p, digest in summary["root_source_sha256"].items())
    summary["status"] = "complete"
    write_json(output / "summary.json", summary)
    print(json.dumps(dict(status="complete", training=summary["training"],
                          selection=summary["selection"])), flush=True)
    return summary


def main() -> None:
    """Train a fresh curriculum stage using explicit prepared directories."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--train-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--validation-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--rehearsal-dirs", type=Path, nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-undrawn", type=int, default=34)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    settings = CurriculumSettings(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                  max_undrawn=args.max_undrawn, seed=args.seed)
    run(args.reference, args.train_dirs, args.validation_dirs, args.output, settings,
        rehearsal_dirs=args.rehearsal_dirs, cpu=args.cpu)


if __name__ == "__main__":
    main()
