"""Matched Qdab-policy training, then a fixed paired match for both endpoints."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
BASE = REPO / "experiments/2026-10-08_14-42-boxes-3x3-loop"
CHECKPOINT_SHA = "c24bf71cba7e7b358881a966ab35b33174db0715f7dfd2e79f397f331f9e873f"
ROLES = ("learner-targets", "qdab-targets")


def read_json(path: Path) -> Any:
    """Read an existing receipt."""
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    """Commit a finite receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    """Hash an input or saved result."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    """Record regular files only, rejecting unexpected links."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {path}")
        if path.is_file():
            result[str(path.relative_to(root))] = digest(path)
    return result


def unpack_payload(source: Path | None, destination: Path) -> None:
    """Copy an explicit payload or safely unpack the hash-checked bundled archive."""
    if source is not None:
        tree_hashes(source)
        shutil.copytree(source, destination)
        return
    data = SCRIPT.parent / "data"
    encoded = (data / "teacher-payload.tgz.b64").read_text()
    blob = base64.b64decode("".join(encoded.split()), validate=True)
    expected = (data / "teacher-payload.sha256").read_text().split()[0]
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError("Bundled teacher payload failed SHA256 verification")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        members = archive.getmembers()
        names = set()
        for member in members:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not (member.isfile() or member.isdir())
                or member.name in names
            ):
                raise ValueError(f"Unsafe or duplicate payload member: {member.name}")
            names.add(member.name)
        if sum(m.size for m in members) > 100_000_000:
            raise ValueError("Unexpectedly large teacher payload")
        destination.mkdir(parents=True)
        archive.extractall(destination, filter="data")
    if not (destination / "manifest.json").is_file():
        raise ValueError("Payload archive must place manifest.json at its root")


def fresh_schedule(games: int, seed: int, excluded: set[str]) -> list[dict[str, Any]]:
    """Reject training/diagnostic families and freeze both seats before fitting."""
    from alpha_go.boxes.arena import random_opening
    from alpha_go.boxes.qdab_training_collect import opening_family

    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even number per arm")
    rng = np.random.default_rng(seed)
    used = set(excluded)
    slots: list[dict[str, Any]] = []
    while len(slots) < games:
        opening = [list(move) for move in random_opening(5, 5, 4, rng)]
        family = opening_family(opening)
        if family in used:
            continue
        used.add(family)
        pair = len(slots) // 2
        for first in (True, False):
            index = len(slots)
            slots.append(
                dict(
                    index=index,
                    panel="paired",
                    pair=pair,
                    a_first=first,
                    seed=seed + index,
                    opening=opening,
                )
            )
    return slots


def replay_families(replay: Path) -> set[str]:
    """Exclude every recoverable four-edge source family in the existing replay."""
    from alpha_go.boxes.policy_target_ab import bundle_files

    spec = importlib.util.spec_from_file_location(
        "teacher_payload", SCRIPT.with_name("prepare_teacher.py")
    )
    assert spec and spec.loader
    preparer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparer)
    files = bundle_files(replay)
    families = set()
    for name in files:
        if not name.endswith(".npz"):
            continue
        with np.load(replay / name, allow_pickle=False) as game:
            family = preparer.game_family(dict(game))
            if family is not None:
                families.add(family)
    return families


def create(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Freeze private local inputs, the learning recipe and the final sample."""
    import torch

    from alpha_go.boxes import qdab_benchmark as match
    from alpha_go.boxes.policy_target_ab import bundle_files
    from alpha_go.boxes.qdab_training_collect import opening_family

    checkpoint, replay = args.checkpoint.resolve(), args.replay.resolve()
    if digest(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Expected the unchanged ordinary-four cycle-4 checkpoint")
    if not args.prepare_only and not torch.cuda.is_available():
        raise ValueError("CUDA required for this training run; --prepare-only performs no fitting")
    replay_hashes = bundle_files(replay)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="qdab-teacher-ab-", dir=args.output_root)).resolve()
    inputs = output / "inputs"
    inputs.mkdir()
    shutil.copy2(checkpoint, inputs / "reference.pt")
    if digest(inputs / "reference.pt") != CHECKPOINT_SHA:
        raise ValueError("Checkpoint changed while freezing the run")
    unpack_payload(args.payload.resolve() if args.payload else None, inputs / "payload")
    manifest = read_json(inputs / "payload/manifest.json")
    if manifest["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Teacher payload uses a different source checkpoint")
    excluded = replay_families(replay)
    excluded.update(manifest["opening_families"])
    excluded.update(manifest["benchmark_opening_families"])
    excluded.update(opening_family(slot["opening"]) for slot in match.schedule(8, 0, 20261116))
    slots = fresh_schedule(args.games, 20261121, excluded)
    write_json(inputs / "opening-exclusions.json", sorted(excluded))
    write_json(inputs / "replay-hashes.json", replay_hashes)
    sources = [
        SCRIPT,
        SCRIPT.with_suffix(".sh"),
        SCRIPT.with_name("prepare_teacher.py"),
        *sorted((REPO / "src/alpha_go").rglob("*.py")),
    ]
    config = dict(
        schema=1,
        checkpoint_sha256=CHECKPOINT_SHA,
        replay=str(replay),
        input_sha256=tree_hashes(inputs),
        runtime=match.runtime(),
        code_sha256={str(p.relative_to(REPO)): digest(p) for p in sources},
        steps=2048,
        batch_size=256,
        lr=0.0001,
        seed=20261120,
        min_undrawn=24,
        snapshot_every=1024,
        games_per_arm=args.games,
        final_seed=20261121,
        schedule=slots,
        search=match.SEARCH,
        evaluator=match.EVALUATOR,
        release=str(args.release.resolve()),
        seconds=5,
        hard_timeout=30,
        scope="Fixed matched approximate-Qdab-policy intervention; no automatic promotion. "
        "Both arms use ordinary sampling, identical positions, values and updates. "
        "The final paired Qdab score difference is primary; no snapshot selection.",
    )
    write_json(output / "config.json", config)
    (output / "config.sha256").write_text(digest(output / "config.json") + "\n")
    (output / "code-commit.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True)
    )
    command = shlex.join(["bash", str(SCRIPT.with_suffix(".sh")), "--resume", str(output)])
    (output / "resume.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd "
        + shlex.quote(str(REPO))
        + "\n"
        + command
        + "\n"
    )
    return output, config


def verify(output: Path, config: dict[str, Any]) -> None:
    """Require the same recipe, source, runtime and immutable inputs on resume."""
    from alpha_go.boxes import qdab_benchmark as match
    from alpha_go.boxes.policy_target_ab import bundle_files

    if digest(output / "config.json") != (output / "config.sha256").read_text().strip():
        raise ValueError("Run configuration changed")
    if config["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Unexpected starting checkpoint")
    if digest(output / "inputs/reference.pt") != CHECKPOINT_SHA:
        raise ValueError("Frozen reference checkpoint changed")
    if tree_hashes(output / "inputs") != config["input_sha256"]:
        raise ValueError("Frozen run inputs changed")
    if bundle_files(Path(config["replay"])) != read_json(output / "inputs/replay-hashes.json"):
        raise ValueError("Source replay changed")
    if match.runtime() != config["runtime"]:
        raise ValueError("Runtime changed; keep the same Python, native extension and GPU")
    for name, expected in config["code_sha256"].items():
        if digest(REPO / name) != expected:
            raise ValueError(f"Experiment source changed during run: {name}")


def job(command: list[str], log: Path, cap_seconds: int) -> None:
    """Run one bounded preparer/fit with a heartbeat and process-group cleanup."""
    print("Running: " + shlex.join(command) + f"\nLog: {log}", flush=True)
    started = time.monotonic()
    with log.open("w") as file:
        process = subprocess.Popen(
            command, cwd=REPO, stdout=file, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    lines = log.read_text(errors="replace").splitlines()
                    print(
                        f"{log.parent.parent.name}: {elapsed:.0f}s; "
                        + (lines[-1] if lines else "starting"),
                        flush=True,
                    )
                    if elapsed >= cap_seconds:
                        raise TimeoutError(f"Stage exceeded {cap_seconds / 60:g} minutes: {log}")
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def stage(
    output: Path, name: str, command: Callable[[Path], list[str]], cap: int, artifact: str
) -> Path:
    """Reuse verified complete stages; preserve failed attempts and retry only on resume."""
    folder = output / name
    folder.mkdir(exist_ok=True)
    receipt = folder / "completed.json"
    if receipt.exists():
        saved = read_json(receipt)
        attempt = folder / str(saved["attempt"])
        if tree_hashes(attempt / artifact) != saved["artifact_sha256"]:
            raise ValueError(f"Completed stage changed: {name}")
        return attempt / artifact
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=folder))
    argv = command(attempt)
    write_json(attempt / "command.json", argv)
    job(argv, attempt / "run.log", cap)
    target = attempt / artifact
    if not target.is_dir():
        raise ValueError(f"Stage did not produce {target}")
    write_json(receipt, dict(attempt=attempt.name, artifact_sha256=tree_hashes(target)))
    return target


def matched_traces(paths: dict[str, Path], bundles: Path) -> dict[str, Any]:
    """Require identical training/symmetry draws and count actual approximate supervision."""
    from alpha_go.boxes.policy_target_ab import load_arm

    reports = {role: read_json(path / "summary.json") for role, path in paths.items()}
    left, right = (reports[role] for role in ROLES)
    if any(
        r["status"] != "complete" or r["result"]["status"] != "complete" for r in reports.values()
    ):
        raise ValueError("Both training endpoints must be complete")
    for key in ("index_trace_sha256", "augmentation_schedule_sha256", "steps"):
        if left["result"][key] != right["result"][key]:
            raise ValueError(f"Training arms are not matched: {key}")
    if (
        left["reference_sha256"] != right["reference_sha256"]
        or left["settings"] != right["settings"]
    ):
        raise ValueError("Training starts/settings differ")
    if left["result"]["steps"] != 2048:
        raise ValueError("Expected the fixed 2048-update endpoint")
    saved = [np.load(paths[role] / "control-indices.npz")["indices"] for role in ROLES]
    if not np.array_equal(saved[0], saved[1]):
        raise ValueError("Saved training indices differ")
    data = load_arm(bundles / ROLES[0], "exact", "chains", 24)
    train, _ = data.split_stable(0.1, 0)
    marked = np.concatenate(
        [
            np.asarray(game.get("approximate_teacher_mask", np.zeros(len(game["boards"]), bool)))[
                rows
            ]
            for game, rows in zip(train.games, train.position_indices)
        ]
    )
    if saved[0].min() < 0 or saved[0].max() >= len(marked):
        raise ValueError("Saved indices do not match the actual training population")
    if not marked.sum() or not marked[saved[0]].sum():
        raise ValueError("No approximate teacher target was consumed by training")
    return dict(
        index_trace_sha256=left["result"]["index_trace_sha256"],
        augmentation_schedule_sha256=left["result"]["augmentation_schedule_sha256"],
        updates=2048,
        positions=len(marked),
        eligible_teacher_rows=int(marked.sum()),
        teacher_exposures=int(marked[saved[0]].sum()),
        unique_teacher_rows_exposed=int(marked[np.unique(saved[0])].sum()),
        all_training_indices_identical=True,
        exact_targets_and_values_matched=True,
    )


def teacher_reads(payload: Path, checkpoints: dict[str, Path]) -> dict[str, Any]:
    """Read approximate-action agreement without fitting or querying Qdab again."""
    import torch

    from alpha_go.boxes.encode import encode
    from alpha_go.boxes.nn_agent import load_boxes_net
    from alpha_go.boxes.rules import BoxesBoard

    rows = [read_json(path) for path in sorted((payload / "teacher").glob("*.json"))]
    if not rows:
        raise ValueError("Payload has no teacher recommendations")
    boards = []
    for row in rows:
        board = BoxesBoard(5, 5)
        for edge in row["root"]["prefix"]:
            if not board.play_edge(edge):
                raise ValueError("Illegal heldout teacher prefix")
        if not board.is_legal_edge(row["teacher_edge"]):
            raise ValueError("Illegal teacher action")
        boards.append(board)
    torch.set_num_threads(1)
    result: dict[str, Any] = {}
    for label, checkpoint in checkpoints.items():
        model = load_boxes_net(checkpoint, torch.device("cpu"))
        model.eval()
        records = []
        with torch.inference_mode():
            for start in range(0, len(rows), 64):
                subset = boards[start : start + 64]
                planes = torch.from_numpy(np.stack([encode(b, model.features) for b in subset]))
                policy, _ = model(planes)
                legal = torch.zeros_like(policy, dtype=torch.bool)
                for index, board in enumerate(subset):
                    legal[index, board.get_legal_moves_flat()] = True
                probs = policy.masked_fill(~legal, float("-inf")).softmax(-1).numpy()
                for index, probabilities in enumerate(probs):
                    row = rows[start + index]
                    records.append(
                        dict(
                            source_family=row["source_family"],
                            split=row["split"],
                            move_count=row["root"]["move_count"],
                            teacher_edge=row["teacher_edge"],
                            probability=float(probabilities[row["teacher_edge"]]),
                            agrees=bool(probabilities.argmax() == row["teacher_edge"]),
                        )
                    )
        metrics = {}
        for split in ("training", "validation"):
            selected = [r for r in records if r["split"] == split]
            metrics[split] = dict(
                positions=len(selected),
                families=len({r["source_family"] for r in selected}),
                mean_teacher_probability=float(np.mean([r["probability"] for r in selected])),
                top_move_agreement=float(np.mean([r["agrees"] for r in selected])),
            )
        result[label] = dict(metrics=metrics, records=records, checkpoint_sha256=digest(checkpoint))
    result["scope"] = (
        "Approximate teacher agreement, not exact correctness. Splits are the "
        "collector's frozen source families; some training-family rows may be "
        "excluded from fitting by the replay split or semantic filters."
    )
    return result


def run_match(output: Path, role: str, checkpoint: Path, config: dict[str, Any]) -> Path:
    """Finish exactly the frozen game slots, preserving each verified completed game."""
    from alpha_go.boxes import qdab_benchmark as match

    folder = output / ("arena-" + role)
    folder.mkdir(exist_ok=True)
    pointer = folder / "match.json"
    if pointer.exists():
        record = read_json(pointer)
        match_dir = folder / record["directory"]
        if digest(match_dir / "config.json") != record["config_sha256"]:
            raise ValueError("Frozen match configuration changed")
        recipe = read_json(match_dir / "config.json")
    else:
        args = argparse.Namespace(
            checkpoint=checkpoint,
            checkpoint_sha256=digest(checkpoint),
            output_root=folder,
            release=Path(config["release"]),
            games=config["games_per_arm"],
            empty_games=0,
            seed=config["final_seed"],
            seconds=5,
            hard_timeout=30,
            cpu=False,
        )
        match_dir, recipe = match.initialize(args)
        # Replace the default schedule before any game; the outer protocol froze this
        # rejection-sampled schedule before either arm was trained.
        recipe.update(
            schedule=config["schedule"], search=config["search"], evaluator=config["evaluator"]
        )
        write_json(match_dir / "config.json", recipe)
        write_json(
            pointer, dict(directory=match_dir.name, config_sha256=digest(match_dir / "config.json"))
        )
    if (
        recipe["checkpoint_sha256"] != digest(checkpoint)
        or recipe["schedule"] != config["schedule"]
    ):
        raise ValueError("Wrong endpoint or opening schedule for match")
    print(f"{role}: {config['games_per_arm']} fixed full games against Qdab", flush=True)
    with Path("/tmp/autoboxes-qdab-12345.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            match.run(match_dir, recipe)
        finally:
            match.report(match_dir, recipe)
    return Path(match_dir)


def compare_matches(matches: dict[str, Path], config: dict[str, Any]) -> dict[str, Any]:
    """Compute the predeclared score difference with complete opening clusters."""
    from alpha_go.boxes import qdab_benchmark as match

    data = {
        role: match.completed(path, read_json(path / "config.json"))
        for role, path in matches.items()
    }
    if any(len(rows) != config["games_per_arm"] for rows in data.values()):
        raise ValueError("Complete both fixed matches before interpreting the comparison")
    old, new = (data[role] for role in ROLES)
    if [r["slot"] for r in old] != [r["slot"] for r in new]:
        raise ValueError("The two endpoints used different openings")
    scores = np.array(
        [[float(row["a_margin"] > 0) for row in old], [float(row["a_margin"] > 0) for row in new]]
    ).T
    pairs = scores.reshape(-1, 2, 2).mean(axis=1)
    rng = np.random.default_rng(config["final_seed"])
    indices = rng.integers(0, len(pairs), (10000, len(pairs)))
    samples = pairs[indices].mean(axis=1)
    return dict(
        status="complete",
        games_per_arm=len(old),
        opening_pairs=len(pairs),
        roles=list(ROLES),
        wins=scores.sum(axis=0).astype(int).tolist(),
        score=scores.mean(axis=0).tolist(),
        qdab_target_minus_control=float((pairs[:, 1] - pairs[:, 0]).mean()),
        paired_difference_ci95=np.quantile(samples[:, 1] - samples[:, 0], [0.025, 0.975]).tolist(),
        bootstrap_seed=config["final_seed"],
        resamples=10000,
        panels={
            role: read_json(path / "summary.json")["panels"]["paired"]
            for role, path in matches.items()
        },
        scope="Matched final-endpoint policy-target comparison against configured Qdab. "
        "Opening family is the bootstrap unit. Timed opponent randomness is uncontrolled. "
        "No automatic promotion or published absolute Elo.",
    )


def package(output: Path) -> None:
    """Export compact diagnostics and game receipts, excluding replay and weights."""
    temporary = output / "results.tmp.tgz"
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(output.rglob("*")):
            relative = path.relative_to(output)
            if (
                not path.is_file()
                or path.is_symlink()
                or path.suffix in {".pt", ".npz", ".tgz"}
                or relative.parts[0] == "inputs"
                or "bundles" in relative.parts
            ):
                continue
            archive.add(path, arcname=str(relative), recursive=False)
    temporary.replace(output / "results.tgz")


def run(output: Path, config: dict[str, Any], prepare_only: bool = False) -> None:
    """Prepare once, train both fixed endpoints, then complete the paired comparison."""
    bundles = stage(
        output,
        "prepare",
        lambda attempt: [
            sys.executable,
            "-u",
            str(SCRIPT.with_name("prepare_teacher.py")),
            "augment",
            "--payload",
            str(output / "inputs/payload"),
            "--replay",
            config["replay"],
            "--output",
            str(attempt / "bundles"),
        ],
        1800,
        "bundles",
    )
    shutil.copyfile(bundles / "report.json", output / "preparation-report.json")
    if prepare_only:
        print("Preparation verified. No training or matches were run.", flush=True)
        return
    paths = {}
    for role in ROLES:

        def train_command(attempt: Path) -> list[str]:
            return [
                sys.executable,
                "-u",
                "-m",
                "alpha_go.boxes.frontier_fit",
                "--reference",
                str(output / "inputs/reference.pt"),
                "--prepared",
                str(bundles / role),
                "--output",
                str(attempt / "train"),
                "--arm",
                "control",
                "--steps",
                "2048",
                "--batch-size",
                "256",
                "--lr",
                "0.0001",
                "--seed",
                "20261120",
                "--min-undrawn",
                "24",
                "--snapshot-every",
                "1024",
            ]

        paths[role] = stage(output, "train-" + role, train_command, 5400, "train")
    write_json(output / "matched-training.json", matched_traces(paths, bundles))
    checkpoints = {role: path / "control.pt" for role, path in paths.items()}
    reads_path = output / "teacher-reads.json"
    if not reads_path.exists():
        write_json(
            reads_path,
            teacher_reads(
                output / "inputs/payload",
                {"reference": output / "inputs/reference.pt", **checkpoints},
            ),
        )
    matches = {role: run_match(output, role, checkpoints[role], config) for role in ROLES}
    write_json(output / "summary.json", compare_matches(matches, config))
    print(json.dumps(read_json(output / "summary.json"), indent=2), flush=True)


def main() -> None:
    """Start the fixed experiment or resume the same frozen inputs and game slots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload", type=Path, help="explicit unpacked payload; default bundled data"
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=BASE / "logs/ordinary-four-aqowx5f8/cycle-4-prepare/attempt-tbyadiie/prepared",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=BASE / "logs/qdab-match-1yafbn9k/inputs/checkpoint.pt"
    )
    parser.add_argument("--output-root", type=Path, default=BASE / "logs")
    parser.add_argument("--release", type=Path, default=REPO / "work/qdab-release")
    parser.add_argument(
        "--games", type=int, default=200, help="fixed full games per arm (default200)"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    if args.resume:
        output = args.resume.resolve()
        config = read_json(output / "config.json")
    else:
        output, config = create(args)
    print(
        f"Run: {output}\nResume: bash {output / 'resume.sh'}\n"
        f"Two fixed 2048-update fits, then {config['games_per_arm']} Qdab games per arm.\n"
        "Default full experiment is approximately 8–12 hours; no early score-based stop.\n"
        "Keep the checkout/build unchanged until completion. Every completed game survives resume.",
        flush=True,
    )

    def terminate(_signal: int, _frame: Any) -> None:
        raise KeyboardInterrupt("Stopping safely; use the printed resume command")

    signal.signal(signal.SIGTERM, terminate)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify(output, config)
            run(output, config, args.prepare_only)
            verify(output, config)
        finally:
            package(output)
            print(f"Send this file: {output / 'results.tgz'}", flush=True)


if __name__ == "__main__":
    main()
