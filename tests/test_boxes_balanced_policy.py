"""The sampling experiment keeps labels, update budget and augmentation controlled."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import alpha_go_cpp
import numpy as np
import pytest
import torch

from alpha_go.boxes.balanced_policy import (
    CyclingPool,
    batches,
    clean_validation,
    coverage,
    input_key,
    read_labels,
    root_reads,
    run,
)
from alpha_go.boxes.dataset import BoxesDataset
from alpha_go.boxes.encode import encode_grid
from alpha_go.boxes.model import BoxesNet
from alpha_go.boxes.nn_agent import load_boxes_net, save_boxes_net
from alpha_go.boxes.policy_ab_data import prepare
from alpha_go.boxes.policy_labels import LabelSettings
from alpha_go.boxes.policy_target_ab import bundle_files, load_arm, sha256
from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.symmetry import apply, transforms
from alpha_go.boxes.validation import replay_group
from alpha_go.model import MaskedBatchNorm2d


def test_cycles_and_balanced_quota_preserve_nonroot_distribution() -> None:
    pool = CyclingPool(np.arange(7), np.random.default_rng(4))
    drawn = np.concatenate([pool.take(4), pool.take(6), pool.take(11)])
    for chunk in drawn.reshape(3, 7):
        assert sorted(chunk) == list(range(7))
    marked = np.array([True] * 3 + [False] * 9)
    first = list(batches(marked, 8, 12, 123, 2))
    second = list(batches(marked, 8, 12, 123, 2))
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert all(marked[row].sum() == 2 for row in first)
    counts = np.bincount(np.concatenate(first), minlength=len(marked))
    # Nine nonroots each seen eight times, regardless of their policy/value-only role.
    assert list(counts) == [8] * 12
    uniform = np.concatenate(list(batches(marked, 8, 3, 123, None)))
    assert np.bincount(uniform).tolist() == [2] * 12


@pytest.mark.parametrize("shape", [(2, 3), (3, 3)])
@pytest.mark.parametrize("features", ["basic", "chains"])
def test_compact_key_equals_full_model_input_equivalence(shape: tuple[int, int],
                                                        features: str) -> None:
    geo = geometry(*shape)
    board = BoxesBoard(*shape)
    rng = np.random.default_rng(42)
    compact_to_full = {}
    full_to_compact = {}
    for _ in range(geo.num_edges):
        grid = board.to_numpy()
        mover = board.player()
        # Renaming players and rotating/reflection must change neither equivalence class.
        swapped = grid.copy()
        owners = grid[1::2, 1::2]
        swapped[1::2, 1::2] = np.where(owners == 0, 0, 3-owners)
        for candidate, player in [(grid, mover), (swapped, 1-mover)]:
            for k in transforms(*shape):
                transformed = apply(candidate, k)
                key = input_key(transformed, player, geo)
                planes = encode_grid(transformed, player+1, features)
                full = min(apply(planes, j).tobytes() for j in transforms(*shape))
                assert compact_to_full.setdefault(key, full) == full
                assert full_to_compact.setdefault(full, key) == key
                assert key == input_key(grid, mover, geo)
        board.play_edge(int(rng.choice(board.get_legal_moves_flat())))


def test_owner_arrangement_is_not_replaced_by_score() -> None:
    geo = geometry(3, 3)
    grid = np.zeros(geo.lattice_shape, np.int8)
    for edge in [0, 2, 4, 10, 13]:
        grid[geo.edge_rc[edge]] = 1
    a, b = grid.copy(), grid.copy()
    a[1, 1], a[1, 3] = 1, 2
    b[1, 1], b[3, 3] = 1, 2
    assert input_key(a, 0, geo) != input_key(b, 0, geo)
    # Dots have no input plane and must not prevent a true overlap from being removed.
    noisy = a.copy()
    noisy[0::2, 0::2] = 9
    assert input_key(noisy, 0, geo) == input_key(a, 0, geo)


def synthetic(grid: np.ndarray[Any, Any], family: str, root: bool) -> dict[str, Any]:
    geo = geometry((grid.shape[0]-1)//2, (grid.shape[1]-1)//2)
    visits = np.array([[int(grid[rc] == 0) for rc in geo.edge_rc]], np.int16)
    return dict(boards=grid[None], to_play=np.array([0]), moves=np.array([geo.edge_rc[0]]),
                result=np.array("B+1.0"), source_replay_group=np.array(family),
                exact_margin_p1=np.array([1]), policy_ab_root_mask=np.array([root]),
                mcts_visits=visits)


def test_validation_removes_all_input_leaks_and_prioritizes_roots() -> None:
    board = BoxesBoard(3, 3)
    board.play_edge(0)
    train_grid = board.to_numpy()
    board.play_edge(4)
    novel = board.to_numpy()
    training = BoxesDataset([], games=[synthetic(train_grid, "aa", False)])
    validation = BoxesDataset([], games=[synthetic(apply(train_grid, 1), "bb", True),
        synthetic(novel, "cc", False), synthetic(apply(novel, 2), "dd", True),
        synthetic(novel, "ee", True)])
    cleaned, report = clean_validation(training, validation)
    assert len(training) == 1 and len(cleaned) == 1
    assert coverage(cleaned)["exact_roots"] == 1
    assert replay_group(cleaned.games[0], cleaned.geo) == bytes.fromhex("dd")
    assert report["training_overlap"] == report["training_overlap_roots"] == 1
    assert report["validation_duplicate"] == 2
    assert report["validation_duplicate_roots"] == 1
    assert report["before"]["positions"] == 4


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("balanced-policy")
    source = root / "source"
    source.mkdir()
    paths = []
    geo = geometry(3, 3)
    for seed in range(32):
        board = BoxesBoard(3, 3)
        rng = np.random.default_rng(seed)
        grids, players, moves, visits = [], [], [], []
        while board.get_legal_moves_flat():
            legal = board.get_legal_moves_flat()
            edge = int(rng.choice(legal))
            grids.append(board.to_numpy())
            players.append(board.player())
            moves.append(geo.edge_rc[edge])
            visits.append([int(e in legal) for e in range(geo.num_edges)])
            board.play_edge(edge)
        score = round(board.score())
        path = source / f"{seed:03d}.npz"
        np.savez_compressed(path, boards=np.stack(grids), to_play=np.array(players, np.int8),
            moves=np.array(moves), mcts_visits=np.array(visits, np.int16),
            mcts_root_values=np.full(len(moves), .5, np.float32),
            result=np.array(f"{'B' if score > 0 else 'W'}+{abs(score)}.0"),
            game=np.array("boxes"), num_moves=len(moves))
        paths.append(path)
    output = root / "prepared"
    prepare(paths, output, LabelSettings(3, 3, 9, 12, 100_000, 100_000, 1 << 12, 2, 2))
    return output


def test_raw_cache_uses_raw_mover_and_score(prepared: Path) -> None:
    labels, sources = read_labels(prepared, geometry(3, 3))
    solver = alpha_go_cpp.BoxesSolver(3, 3, 1 << 12)
    changed_mover = retained_mover = scored = 0
    for record in sources.values():
        root_mover = int(record["mover"])
        for edges, mover, score, final in record["raw_children"]:
            exact = solver.value_within(edges, 100_000)
            assert exact is not None
            assert int(exact) == (final-score) * (1 if mover == 0 else -1)
            changed_mover += mover != root_mover
            retained_mover += mover == root_mover
            scored += score != 0
    assert labels and changed_mover and retained_mover and scored


def test_fixed_updates_masks_augmentation_bn_and_endpoint(prepared: Path, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(7)
    reference = tmp_path / "reference.pt"
    save_boxes_net(BoxesNet(3, 3, channels=8, n_blocks=1, value_hidden=8,
                           features="chains"), reference)
    checkpoint_hash = sha256(reference)
    inputs = bundle_files(prepared)
    compute = BoxesNet.compute_loss
    seen_weights = []
    bn_forward = MaskedBatchNorm2d.forward
    training_calls: dict[MaskedBatchNorm2d, int] = {}

    def checked_bn(self: MaskedBatchNorm2d, *args: Any, **kwargs: Any) -> Any:
        if self.training:
            training_calls[self] = training_calls.get(self, 0) + 1
        return bn_forward(self, *args, **kwargs)

    monkeypatch.setattr(MaskedBatchNorm2d, "forward", checked_bn)

    def checked_loss(self: BoxesNet, *args: Any, **kwargs: Any) -> Any:
        weight = kwargs["policy_weight_B"]
        seen_weights.append(weight.detach().cpu().numpy())
        assert torch.all((weight == 0) | (weight == 1))
        assert torch.all(args[1][weight == 0] == 0), "invented child policy"
        return compute(self, *args, **kwargs)

    monkeypatch.setattr(BoxesNet, "compute_loss", checked_loss)
    output = tmp_path / "result"
    summary = run(reference, prepared, output, steps=3, batch_size=16, seed=12,
                  root_fraction=.25, cpu=True, min_undrawn=0)
    assert summary["status"] == "complete"
    assert sha256(reference) == checkpoint_hash == sha256(output / "reference.pt")
    assert bundle_files(prepared) == inputs
    assert summary["validation"]["exact_roots"] > 0
    control, balanced = [summary["arms"][arm] for arm in ("control", "balanced")]
    assert control["steps"] == balanced["steps"] == 3
    assert control["augmentation_schedule_sha256"] == balanced["augmentation_schedule_sha256"]
    assert control["index_trace_sha256"] != balanced["index_trace_sha256"]
    assert balanced["exposures"]["exact_roots"] == 12
    assert balanced["exposures"]["rows"] == control["exposures"]["rows"] == 48
    assert np.concatenate(seen_weights).min() == 0
    for name in ("control", "balanced"):
        model = load_boxes_net(output / f"{name}.pt", torch.device("cpu"))
        assert any(key.endswith("running_mean") for key in model.state_dict())
        result = summary["arms"][name]
        assert result["primary_step"] == 3
        assert result["final"]["validation_roots"]["all"]["mean_regret"] >= 0
        assert result["training"]["positions"] == summary["training"]["positions"]
        assert result["validation"]["positions"] == summary["validation"]["positions"]
    assert len(training_calls) == 6 and set(training_calls.values()) == {3}
    assert json.loads((output / "summary.json").read_text()) == summary
    positions = [json.loads(line) for line in (output / "validation-positions.jsonl").read_text()
                 .splitlines()]
    cache = json.loads((output / "validation-labels.json").read_text())
    assert len(positions) == summary["search_validation"]["positions"]
    assert cache["schema"] == 1 and (cache["rows"], cache["cols"]) == (3, 3)
    with pytest.raises(AssertionError, match="fresh"):
        run(reference, prepared, output, steps=3, batch_size=16, cpu=True)


def test_intermediate_root_reads_restore_training_without_bn_updates(prepared: Path) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(18)
    model = BoxesNet(3, 3, channels=8, n_blocks=1, value_hidden=8, features="basic")
    model.train()
    data = load_arm(prepared, "exact", "basic", 0)
    labels, _ = read_labels(prepared, data.geo)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    root_reads(model, data, labels, 32, torch.device("cpu"))
    assert model.training
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
