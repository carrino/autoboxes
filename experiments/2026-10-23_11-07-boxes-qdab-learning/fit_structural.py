"""Cache frozen Boxes features and fit three isolated structural-value readouts.

No production checkpoint, policy, BatchNorm statistic or search setting is changed.
Only independently solved values are targets. Full-child decisions are diagnostic;
this script neither runs an arena nor promotes a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import alpha_go_cpp  # type: ignore[import-not-found]
import numpy as np
import torch
from structural_features import SCHEMA, feature_names, from_board
from structural_value import StructuralValueHead
from torch import nn
from torch.nn import functional

from alpha_go.boxes.encode import encode
from alpha_go.boxes.nn_agent import load_boxes_net
from alpha_go.boxes.policy_value_audit import average_ranks
from alpha_go.boxes.rules import BoxesBoard

ARMS = ("calibration", "structural_only", "structural_residual")
EXACT_STATUSES = {"exact", "exact_capture_collapse", "exact_all_children", "exact_terminal"}
UNKNOWN_STATUSES = {"not_attempted", "deadline", "node_budget", "excluded"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replay(row: dict[str, Any]) -> BoxesBoard:
    """Independently reconstruct ownership as well as the mask, mover and scores."""
    board = BoxesBoard(int(row["rows"]), int(row["cols"]))
    for edge in row["prefix"]:
        if not isinstance(edge, int) or not board.play_edge(edge):
            raise ValueError("Illegal state prefix")
    mask = int(row["edges"], 0) if str(row["edges"]).startswith("0x") else int(row["edges"])
    if (board.edges != mask or board.to_play() != row["to_play"] or
            board.boxes != row["scores"] or int(board.score()) != row["score_p1"]):
        raise ValueError("Replayed mask, mover or scores differ")
    label = row["exact_margin_p1"]
    if (label is None and row["status"] not in UNKNOWN_STATUSES or
            label is not None and row["status"] not in EXACT_STATUSES):
        raise ValueError("Label status does not establish an exact value")
    if label is not None:
        remaining = board.geo.num_boxes - sum(board.boxes)
        delta = label - int(board.score())
        if (type(label) is not int or abs(delta) > remaining or
                (delta - remaining) % 2 != 0):
            raise ValueError("Impossible exact final P1 margin")
    return board


def semantic_key(board: BoxesBoard) -> bytes:
    """D4 mask and mover-relative scores; owner placement/color is not game value."""
    grid = board.to_numpy().copy()
    grid[1::2, 1::2] = 0
    versions: list[bytes] = []
    for turns in range(4):
        rotated = np.rot90(grid, turns)
        if rotated.shape == grid.shape:
            versions.extend((rotated.tobytes(), np.fliplr(rotated).tobytes()))
    mover = board.player()
    return (f"{board.rows()}:{board.cols()}:{board.boxes[mover]}:"
            f"{board.boxes[1 - mover]}:".encode() + min(versions))


def load_bundle(prepared: Path, development: bool) -> tuple[
        dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[BoxesBoard]]:
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("status") not in ("complete", "partial"):
        raise ValueError("Unrecognized structural bundle")
    for name in ("states.jsonl", "roots.jsonl", "exclusion_receipt.json"):
        if manifest.get("artifact_sha256", {}).get(name) != sha256(prepared / name):
            raise ValueError(f"Bundle digest mismatch: {name}")
    if not manifest.get("known_exclusions_verified", False):
        raise ValueError("Known cloud-pilot/diagnostic exclusions have not been verified")
    receipt = json.loads((prepared / "exclusion_receipt.json").read_text())
    if (manifest.get("exclusion_receipt_sha256") != sha256(prepared / "exclusion_receipt.json")
            or not receipt.get("known_exclusions_verified", False)
            or receipt.get("eligible_train_validation_intersection") != 0):
        raise ValueError("Invalid known-exclusions receipt")
    if not exclusions_cleared(manifest, receipt) and not development:
        raise ValueError("Actual benchmark exclusions pending; explicit development flag required")
    states = read_jsonl(prepared / "states.jsonl")
    roots = read_jsonl(prepared / "roots.jsonl")
    if not states or not roots:
        raise ValueError("Empty structural bundle")
    if len({row["state_id"] for row in states}) != len(states):
        raise ValueError("Duplicate state identifiers")
    if len({row["root_id"] for row in roots}) != len(roots):
        raise ValueError("Duplicate root identifiers")
    boards = [replay(row) for row in states]
    geometries = {(board.rows(), board.cols()) for board in boards}
    if len(geometries) != 1:
        raise ValueError("Mixed board geometries")
    seen: dict[bytes, str] = {}
    families: dict[str, str] = {}
    for row, board in zip(states, boards):
        split = row["split"]
        if split not in ("training", "validation"):
            raise ValueError("Unknown family split")
        family = row["source_family"]
        if families.setdefault(family, split) != split:
            raise ValueError("Source family occurs in both splits")
        if row["training_eligible"]:
            key = semantic_key(board)
            if seen.setdefault(key, split) != split:
                raise ValueError("D4-equivalent state/score leakage across splits")
    indexed = {row["state_id"]: (row, board) for row, board in zip(states, boards)}
    for root in roots:
        row, board = indexed[root["state_id"]]
        root_board = BoxesBoard(board.rows(), board.cols())
        for edge in root["prefix"]:
            if not root_board.play_edge(edge):
                raise ValueError("Illegal root prefix")
        if (root["source_family"] != row["source_family"] or root["split"] != row["split"] or
                root["to_play"] != board.to_play() or root_board.to_play() != board.to_play() or
                root["undrawn"] != board.num_edges() - board.move_count() or
                not np.array_equal(root_board.to_numpy(), board.to_numpy())):
            raise ValueError("Root identity disagrees with state")
        if root["root_id"] not in row["root_groups"]:
            raise ValueError("Missing root membership")
        edges = [child["edge"] for child in root["children"]]
        if len(set(edges)) != len(edges) or any(not board.is_legal_edge(e) for e in edges):
            raise ValueError("Invalid/duplicate child edge")
        if root["complete_children"] and sorted(edges) != board.get_legal_moves_flat():
            raise ValueError("Complete group omits legal children")
        for child in root["children"]:
            raw, raw_board = indexed[child["raw_state_id"]]
            collapsed, collapsed_board = indexed[child["collapsed_state_id"]]
            expected = board.copy()
            expected.play_edge(child["edge"])
            if not np.array_equal(expected.to_numpy(), raw_board.to_numpy()):
                raise ValueError("Raw child does not follow its parent edge")
            if raw_board.to_play() != expected.to_play():
                raise ValueError("Raw child mover differs")
            native = alpha_go_cpp.BoxesBoard(board.rows(), board.cols())
            for edge in raw["prefix"]:
                native.play_edge(edge)
            collapsed_native = alpha_go_cpp.BoxesSearchState(native).board()
            if (collapsed_native.to_play() != collapsed_board.to_play() or
                    not np.array_equal(collapsed_native.to_numpy(), collapsed_board.to_numpy())):
                raise ValueError("Collapsed child does not match forced-capture semantics")
            for state in (raw, collapsed):
                if (state["source_family"] != root["source_family"] or
                        state["split"] != root["split"]):
                    raise ValueError("Child escaped its source family")
                if root["root_id"] not in state["root_groups"]:
                    raise ValueError("Missing child root membership")
                if (child["exact_margin_p1"] is not None and
                        child["exact_margin_p1"] != state["exact_margin_p1"]):
                    raise ValueError("Raw/collapsed exact child values disagree")
            if child["exact_margin_p1"] != raw["exact_margin_p1"]:
                raise ValueError("Child/raw label disagreement")
            if root["complete_children"] and child["exact_margin_p1"] is None:
                raise ValueError("Complete group contains an unknown child")
        if root["complete_children"] and root["children"]:
            sign = 1 if root["to_play"] == 1 else -1
            best = max(sign * child["exact_margin_p1"] for child in root["children"])
            if row["exact_margin_p1"] is not None and sign * row["exact_margin_p1"] != best:
                raise ValueError("Exact root disagrees with complete children")
    return manifest, states, roots, boards


def exclusions_cleared(manifest: dict[str, Any], receipt: dict[str, Any]) -> bool:
    """The manifest and hashed exclusion receipt must agree before clearing scope."""
    return bool(manifest.get("exclusions_verified", False)
                and not manifest.get("overlap_pending_actual_gpu", True)
                and receipt.get("exclusions_verified", False)
                and not receipt.get("overlap_pending_actual_gpu", True))


def cache_features(model: nn.Module, boards: list[BoxesBoard], batch_size: int,
                   device: torch.device) -> dict[str, np.ndarray]:
    """Capture the existing value_fc1 input without reimplementing the trunk."""
    if model.value_target != "final":
        raise ValueError("Only final-margin base weights are supported")
    model.eval().requires_grad_(False)
    if (model.rows, model.cols) != (boards[0].rows(), boards[0].cols()):
        raise ValueError("Checkpoint and data geometries differ")
    pooled = []
    logits = []

    def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        pooled.append(inputs[0].detach().cpu().numpy().copy())

    hook = model.value_fc1.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for offset in range(0, len(boards), batch_size):
                planes = torch.from_numpy(np.stack([
                    encode(board, model.features) for board in boards[offset:offset + batch_size]
                ])).to(device)
                _, value = model(planes)
                logits.append(value.cpu().numpy())
    finally:
        hook.remove()
    arrays = dict(base_logits=np.concatenate(logits).astype(np.float32),
                  pooled_score=np.concatenate(pooled).astype(np.float32),
                  structure=np.stack([from_board(board) for board in boards]))
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Nonfinite cached features")
    return arrays


def training_groups(states: list[dict[str, Any]], roots: list[dict[str, Any]],
                    boards: list[BoxesBoard]) -> dict[str, list[list[int]]]:
    """One vote per family, root and unique known root/collapsed state."""
    indices = {row["state_id"]: i for i, row in enumerate(states)}
    groups: dict[str, list[list[int]]] = defaultdict(list)
    for root in sorted(roots, key=lambda row: row["root_id"]):
        if root["split"] != "training" or not root["training_eligible"]:
            continue
        ids = [root["state_id"]] + [child["collapsed_state_id"] for child in root["children"]]
        chosen: dict[bytes, int] = {}
        for state_id in ids:
            index = indices[state_id]
            row = states[index]
            if row["training_eligible"] and row["exact_margin_p1"] is not None:
                chosen.setdefault(semantic_key(boards[index]), index)
        if chosen:
            groups[root["source_family"]].append(list(chosen.values()))
    if not groups:
        raise ValueError("No eligible known training root/collapsed states")
    return dict(groups)


def sample_indices(groups: dict[str, list[list[int]]], steps: int, batch_size: int,
                   seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    families = sorted(groups)
    samples: np.ndarray = np.empty((steps, batch_size), np.int32)
    for batch in samples:
        for j in range(batch_size):
            roots = groups[families[int(rng.integers(len(families)))]]
            states = roots[int(rng.integers(len(roots)))]
            batch[j] = states[int(rng.integers(len(states)))]
    return samples


def fit_heads(arrays: dict[str, np.ndarray], targets: np.ndarray, indices: np.ndarray,
              *, lr: float, seed: int, device: torch.device,
              output: Path | None = None,
              provenance: dict[str, Any] | None = None
              ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The same index matrix and final-margin CE train all three small readouts."""
    tensors = {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}
    labels = torch.from_numpy(targets).long().to(device)
    predictions = {"base": arrays["base_logits"].copy()}
    training = {}
    for arm in ARMS:
        torch.manual_seed(seed)
        features = tensors["pooled_score" if arm == "calibration" else "structure"]
        head = StructuralValueHead(features.shape[1], arrays["base_logits"].shape[1]).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        trace = []
        for step, sampled in enumerate(indices, 1):
            index = torch.from_numpy(sampled.astype(np.int64)).to(device)
            value = head(features[index])
            if arm != "structural_only":
                value = value + tensors["base_logits"][index]
            loss = functional.cross_entropy(value, labels[index])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite fitting loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1)
            optimizer.step()
            if step == 1 or step % 128 == 0 or step == len(indices):
                trace.append(dict(step=step, loss=float(loss.detach())))
                print(f"{arm}: {step}/{len(indices)} updates, CE {float(loss.detach()):.4f}",
                      flush=True)
        head.eval()
        with torch.no_grad():
            value = head(features)
            if arm != "structural_only":
                value = value + tensors["base_logits"]
            predictions[arm] = value.cpu().numpy()
        training[arm] = dict(trace=trace, parameters=sum(p.numel() for p in head.parameters()))
        if output is not None:
            torch.save(dict(schema=1, arm=arm, hidden=64, feature_count=features.shape[1],
                            num_margins=arrays["base_logits"].shape[1],
                            feature_schema=SCHEMA, model_state_dict=head.state_dict(),
                            provenance=provenance,
                            scope="Isolated diagnostic head; not a production Boxes checkpoint"),
                       output / f"{arm}.pt")
    return predictions, training


def probabilities(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
    result: np.ndarray = np.asarray(exp / exp.sum(axis=1, keepdims=True), dtype=np.float64)
    return result


def choice_metrics(predicted: np.ndarray, exact: np.ndarray, edges: list[int]) -> dict[str, Any]:
    best = int(np.argmax(predicted))
    delta = exact[:, None] - exact[None, :]
    predicted_delta = predicted[:, None] - predicted[None, :]
    distinct = np.triu(delta != 0, 1)
    agreement = np.where(predicted_delta * delta > 0, 1.,
                         np.where(predicted_delta == 0, .5, 0.))[distinct]
    left, right = average_ranks(predicted), average_ranks(exact)
    spearman = (float(np.corrcoef(left, right)[0, 1])
                if left.std() > 0 and right.std() > 0 else None)
    winning = bool(exact.max() > 0)
    return dict(selected_edge=edges[best], regret=float(exact.max() - exact[best]),
                optimal=int(exact[best] == exact.max()), winning_root=winning,
                win_preserved=int(exact[best] > 0) if winning else None,
                pairwise_accuracy=float(agreement.mean()) if agreement.size else None,
                distinct_pairs=int(distinct.sum()), spearman=spearman)


def child_reads(predictions: dict[str, np.ndarray], states: list[dict[str, Any]],
                roots: list[dict[str, Any]], boards: list[BoxesBoard]) -> list[dict[str, Any]]:
    indices = {row["state_id"]: i for i, row in enumerate(states)}
    margins = np.arange(-boards[0].geo.num_boxes, boards[0].geo.num_boxes + 1)
    win_weights = (margins > 0) + .5 * (margins == 0)
    result = []
    for arm, logits in predictions.items():
        probability = probabilities(logits)
        for root in roots:
            children = sorted(root["children"], key=lambda child: child["edge"])
            if not root["training_eligible"] or not root["complete_children"] or not children:
                continue
            sign = 1 if root["to_play"] == 1 else -1
            exact = np.array([sign * c["exact_margin_p1"] for c in children], dtype=float)
            for mode in ("raw", "collapsed"):
                chosen = [indices[c[f"{mode}_state_id"]] for c in children]
                if any(not states[i]["training_eligible"] for i in chosen):
                    continue
                values = probability[chosen].copy()
                terminal_count = 0
                if mode == "collapsed":
                    for row, i in enumerate(chosen):
                        if boards[i].is_game_over():
                            label = states[i]["exact_margin_p1"]
                            label *= 1 if states[i]["to_play"] == 1 else -1
                            values[row] = 0
                            values[row, int(label + boards[i].geo.num_boxes)] = 1
                            terminal_count += 1
                win, margin = values @ win_weights, values @ margins
                switched = np.array([states[i]["to_play"] != root["to_play"] for i in chosen])
                win = np.where(switched, 1 - win, win)
                margin = np.where(switched, -margin, margin)
                for predictor, scores in (("win", win), ("margin", margin),
                                          ("utility", win + .5 * np.tanh(margin / 6))):
                    result.append(dict(arm=arm, split=root["split"], family=root["source_family"],
                        root_id=root["root_id"], band=root["undrawn"], mode=mode,
                        predictor=predictor, children=len(children), terminal_exact=terminal_count,
                        **choice_metrics(scores, exact, [c["edge"] for c in children])))
    return result


def state_reads(predictions: dict[str, np.ndarray], states: list[dict[str, Any]],
                boards: list[BoxesBoard]) -> list[dict[str, Any]]:
    result = []
    size = boards[0].geo.num_boxes
    margins = np.arange(-size, size + 1)
    win_weights = (margins > 0) + .5 * (margins == 0)
    for arm, logits in predictions.items():
        probs = probabilities(logits)
        for split in ("training", "validation"):
            indices = [i for i, row in enumerate(states) if row["split"] == split and
                       row["training_eligible"] and row["exact_margin_p1"] is not None]
            if not indices:
                continue
            labels = np.array([states[i]["exact_margin_p1"] *
                               (1 if states[i]["to_play"] == 1 else -1) for i in indices])
            win = probs[indices] @ win_weights
            expected = probs[indices] @ margins
            truth = (labels > 0) + .5 * (labels == 0)
            decided = labels != 0
            ce = -np.log(np.maximum(probs[indices, labels.astype(int) + size], 1e-300))
            result.append(dict(arm=arm, split=split, states=len(indices),
                families=len({states[i]["source_family"] for i in indices}),
                ce=float(ce.mean()),
                margin_mae=float(np.abs(expected - labels).mean()),
                brier=float(np.square(win - truth).mean()),
                sign_accuracy=float(((win[decided] >= .5) == (labels[decided] > 0)).mean())
                    if decided.any() else None,
                constant_mover_wins_accuracy=float((labels[decided] > 0).mean())
                    if decided.any() else None,
                weighting="Cached state rows; child-choice reads use root/family weighting"))
    return result


def summarize_children(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Root means within family, then equal families; paired family delta bootstrap."""
    metrics = ("regret", "optimal", "win_preserved", "pairwise_accuracy", "spearman")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["split"], row["mode"], row["predictor"], row["band"])].append(row)
    means = []
    for key, group in sorted(grouped.items()):
        record = dict(zip(("arm", "split", "mode", "predictor", "band"), key))
        record.update(roots=len(group), families=len({r["family"] for r in group}),
                      winning_roots=sum(r["winning_root"] for r in group),
                      wins_preserved=sum(r["win_preserved"] or 0 for r in group))
        for metric in metrics:
            by_family: dict[str, list[float]] = defaultdict(list)
            for row in group:
                if row[metric] is not None:
                    by_family[row["family"]].append(row[metric])
            record[metric] = (float(np.mean([np.mean(v) for v in by_family.values()]))
                              if by_family else None)
        means.append(record)
    deltas = []
    for reference in ("base", "calibration"):
        for candidate in ARMS:
            if candidate == reference:
                continue
            for mode in ("raw", "collapsed"):
                for predictor in ("win", "margin", "utility"):
                    ref = {r["root_id"]: r for r in rows if r["arm"] == reference and
                           r["split"] == "validation" and r["mode"] == mode and
                           r["predictor"] == predictor}
                    cand = {r["root_id"]: r for r in rows if r["arm"] == candidate and
                            r["split"] == "validation" and r["mode"] == mode and
                            r["predictor"] == predictor}
                    for metric in metrics:
                        differences: dict[str, list[float]] = defaultdict(list)
                        for root_id in sorted(ref.keys() & cand.keys()):
                            a, b = ref[root_id], cand[root_id]
                            if a[metric] is not None and b[metric] is not None:
                                differences[a["family"]].append(b[metric] - a[metric])
                        if not differences:
                            continue
                        values = np.array([np.mean(v) for _, v in sorted(differences.items())])
                        interval = None
                        if len(values) >= 2:
                            rng = np.random.default_rng(seed)
                            draws = rng.integers(0, len(values), size=(2000, len(values)))
                            interval = np.quantile(
                                values[draws].mean(axis=1), [.025, .975]).tolist()
                        deltas.append(dict(reference=reference, candidate=candidate, mode=mode,
                            predictor=predictor, band="all", metric=metric,
                            delta=float(values.mean()),
                            family_ci95=interval, families=len(values), resamples=2000,
                            scope="Development comparison; few families are not strong evidence"))
    return dict(group_means=means, validation_deltas=deltas,
                weighting="Equal families/roots within family; exact ties excluded from pairs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps <= 0 or args.batch_size <= 0 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Invalid fit budget")
    torch.set_num_threads(args.threads)
    if sha256(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("Frozen checkpoint SHA256 mismatch")
    manifest, states, roots, boards = load_bundle(
        args.prepared, args.development_with_pending_exclusions)
    if (manifest.get("checkpoint_sha256") is not None and
            manifest["checkpoint_sha256"] != args.expected_checkpoint_sha256):
        raise ValueError("Collection and fit reference checkpoints differ")
    source_files = [Path(__file__), Path(__file__).with_name("structural_features.py"),
                    Path(__file__).with_name("structural_value.py")]
    import alpha_go.boxes.chains as chains_module
    import alpha_go.boxes.encode as encode_module
    import alpha_go.boxes.model as model_module
    import alpha_go.boxes.rules as rules_module
    source_files += [Path(module.__file__) for module in
                     (encode_module, model_module, chains_module, rules_module)]
    provenance = dict(schema=1, checkpoint_sha256=args.expected_checkpoint_sha256,
        inputs={name: sha256(args.prepared / name) for name in
                ("manifest.json", "states.jsonl", "roots.jsonl", "exclusion_receipt.json")},
        source_sha256={str(path): sha256(path) for path in source_files},
        extension_sha256=sha256(Path(alpha_go_cpp.__file__)), torch_version=str(torch.__version__),
        numpy_version=np.__version__,
        feature_schema=SCHEMA, feature_names=list(feature_names(boards[0].geo)),
        state_ids=[row["state_id"] for row in states],
        preprocessing="Actual replay orientation; no augmentation; base eval/no_grad float32")
    settings = dict(steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                    device=args.device, hidden=64, optimizer="fresh AdamW", weight_decay=1e-4,
                    gradient_clip=1., final_checkpoint_only=True,
                    development_with_pending_exclusions=args.development_with_pending_exclusions)
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(provenance=provenance, settings=settings)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing output has different inputs/settings/code")
    write_json(config_path, config)
    (args.output / "inputs").mkdir(exist_ok=True)
    for name in provenance["inputs"]:
        shutil.copyfile(args.prepared / name, args.output / "inputs" / name)
    cache_path, receipt_path = args.output / "cache.npz", args.output / "cache.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["provenance"] != provenance or receipt["sha256"] != sha256(cache_path):
            raise ValueError("Frozen feature cache mismatch")
        with np.load(cache_path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
    else:
        print(f"Caching frozen base inference for {len(states)} states", flush=True)
        model = load_boxes_net(args.checkpoint, torch.device(args.device))
        arrays = cache_features(model, boards, args.batch_size, torch.device(args.device))
        del model
        np.savez_compressed(cache_path, allow_pickle=False, **arrays)
        write_json(receipt_path, dict(provenance=provenance, sha256=sha256(cache_path)))
    groups = training_groups(states, roots, boards)
    indices = sample_indices(groups, args.steps, args.batch_size, args.seed)
    np.save(args.output / "sample_indices.npy", indices)
    size = boards[0].geo.num_boxes
    targets = np.array([(-999 if row["exact_margin_p1"] is None else
                        row["exact_margin_p1"] * (1 if row["to_play"] == 1 else -1) + size)
                       for row in states], dtype=np.int64)
    if np.any(targets[indices] < 0):
        raise ValueError("Unknown label entered fitting indices")
    receipt = json.loads((args.prepared / "exclusion_receipt.json").read_text())
    pending = not exclusions_cleared(manifest, receipt)
    scope = ("PROVISIONAL DEVELOPMENT: actual GPU benchmark exclusions pending; no promotion"
             if pending else "Held-out source-family development diagnostic; no arena or promotion")
    print(scope, flush=True)
    predictions, training = fit_heads(arrays, targets, indices, lr=args.lr, seed=args.seed,
                                     device=torch.device(args.device), output=args.output,
                                     provenance=provenance)
    np.savez_compressed(args.output / "predictions.npz", allow_pickle=False, **predictions)
    reads = child_reads(predictions, states, roots, boards)
    (args.output / "child_reads.jsonl").write_text("".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in reads))
    ranking_coverage = {}
    for split in ("training", "validation"):
        complete = [row for row in reads if row["arm"] == "base" and row["split"] == split
                    and row["mode"] == "collapsed" and row["predictor"] == "utility"]
        ranking_coverage[split] = dict(complete_roots=len(complete),
            complete_families=len({row["family"] for row in complete}))
    report = dict(schema=1, status="complete", scope=scope, config=config,
        base_frozen=True, policy_unchanged=True, arena_eligible=False,
        training_families=len(groups), training_root_groups=sum(map(len, groups.values())),
        unique_sampled_states=len(np.unique(indices)),
        sample_indices_sha256=sha256(args.output / "sample_indices.npy"),
        source_status=manifest["status"], state_count=len(states), root_count=len(roots),
        complete_child_groups=sum(root["complete_children"] for root in roots),
        ranking_coverage=ranking_coverage,
        ranking_evidence=("No held-out complete-family ranking evidence yet"
            if ranking_coverage["validation"]["complete_families"] == 0 else
            "Development only; report family count before interpreting correlated child reads"),
        training=training, states=state_reads(predictions, states, boards),
        children=summarize_children(reads, args.seed),
        model_provenance="Heads use config.json/cache.json; base checkpoint not embedded")
    if sha256(args.checkpoint) != args.expected_checkpoint_sha256 or any(
            sha256(args.prepared / name) != digest
            for name, digest in provenance["inputs"].items()):
        raise ValueError("Frozen inputs changed during fit")
    write_json(args.output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "prepared", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--development-with-pending-exclusions", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "summary": str(args.output / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
