"""Reproduce the solver34 overnight result from its original uploaded archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from alpha_go.boxes.rules import BoxesBoard

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARCHIVE_SHA = "166037e4520df66c84ce209c5813d00a433ee349c82762199826c1c371c84532"
BOOTSTRAP_SEED = 0
BOOTSTRAP_SAMPLES = 50_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def result(path: Path) -> dict[str, Any]:
    text = path.read_text()
    assert text.count("===RESULT===") == 1, path
    return json.JSONDecoder().raw_decode(text.split("===RESULT===")[1].lstrip())[0]


def source_provenance(config: dict[str, Any], commit: str) -> dict[str, str]:
    """Compare the recorded source bytes to Git, without importing the old program."""
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    verified = {}
    for name, expected in config["code_sha256"].items():
        if name.endswith(".so"):
            continue  # the compiled extension is not present in this evidence archive
        path = Path(name)
        relative = path.relative_to(config["cwd"]) if path.is_absolute() else path
        source = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO)
        assert hashlib.sha256(source).hexdigest() == expected, relative
        verified[str(relative)] = expected
    # The independent replay implementation must be the same recorded rules implementation.
    assert sha(REPO / "src/alpha_go/boxes/rules.py") == verified["src/alpha_go/boxes/rules.py"]
    return verified


def final_match(folder: Path, config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Replay every edge and independently recompute pair-aware game-score statistics."""
    for name, expected in state["final_sha256"].items():
        assert sha(folder / Path(name).name) == expected, name
    saved = load(folder / "final.json")
    command = load(folder / "final-config.json")
    args = command["command"]
    expected_args = {"--rows": "5", "--cols": "5", "--num_games": "200",
                     "--num_workers": "4", "--seed": str(config["final_seed"]),
                     "--opening_moves": "4", "--sims": "400", "--solver": "34",
                     "--solver_budget": "50000", "--c_puct": "1.5",
                     "--leaf_batch_size": "16", "--policy_temperature": "1.0",
                     "--prove_terminals": "1", "--shape_terminals": "1",
                     "--margin_utility_lambda": "0.5", "--margin_utility_k": "6"}
    for flag, value in expected_args.items():
        assert args.count(flag) == 1 and args[args.index(flag) + 1] == value, flag
    candidate = str(Path(config["checkpoint_dir"]) / f"iter{state['completed_iterations']}.pt")
    assert args[args.index("--a") + 1] == saved["a"] == f"ckpt:{candidate}"
    assert args[args.index("--b") + 1] == saved["b"] == f"ckpt:{config['reference']}"
    assert command["checkpoints"] == {
        candidate: state["checkpoints"][candidate],
        config["reference"]: config["reference_sha256"],
    }
    records = [json.loads(line) for line in (folder / "final-games.jsonl").read_text().splitlines()]
    assert len(records) == saved["games"] == config["final_games"] == 200
    margins = []
    openings = []
    for index, record in enumerate(records):
        assert record["index"] == index and record["a_first"] == (index % 2 == 0)
        assert record["opening"] == 4 and len(record["moves"]) == 60
        board = BoxesBoard(5, 5)
        for row, col in record["moves"]:
            assert board.play_edge(board.edge_index(row, col)), (index, row, col)
        assert board.is_game_over()
        margin = int(board.score()) * (1 if record["a_first"] else -1)
        assert margin == record["a_margin"]
        margins.append(margin)
        if index % 2:
            assert record["moves"][:4] == records[index - 1]["moves"][:4]
            openings.append(tuple(map(tuple, record["moves"][:4])))
    values = np.asarray(margins)
    assert not np.any(values == 0), "This odd-box board cannot draw"
    wins, losses = int((values > 0).sum()), int((values < 0).sum())
    assert (wins, losses, 0) == (saved["a_wins"], saved["b_wins"], saved["draws"])
    pair_wins = (values.reshape(-1, 2) > 0).sum(axis=1)
    sweeps, splits, pair_losses = (int((pair_wins == count).sum()) for count in (2, 1, 0))
    decided = sweeps + pair_losses
    # With no drawn games, whole-pair label swapping reduces to a fair binomial
    # over sweep/loss pairs. Split pairs have zero score advantage and drop out.
    paired_p = min(1.0, 2 * sum(math.comb(decided, k)
                              for k in range(min(sweeps, pair_losses) + 1)) / 2**decided)
    assert math.isclose(paired_p, saved["a_paired_win_pvalue"], abs_tol=1e-15)
    pair_scores = pair_wins / 2
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(pair_scores), (BOOTSTRAP_SAMPLES, len(pair_scores)))
    interval = np.quantile(pair_scores[indices].mean(axis=1), [0.025, 0.975]).tolist()
    pair_margins = values.reshape(-1, 2).sum(axis=1)
    legacy_score = float(((pair_margins > 0) + 0.5 * (pair_margins == 0)).mean())
    assert legacy_score == saved["a_pair_score"]
    assert float(values.mean()) == saved["a_mean_margin"]
    final_log = result(folder / "final-arena.log")
    assert all(saved[key] == value for key, value in final_log.items())
    return dict(games=len(values), a_wins=wins, b_wins=losses, draws=0,
                a_game_score=wins / len(values), paired_game_score_ci95=interval,
                paired_two_sided_pvalue=paired_p, pair_sweeps=sweeps, pair_splits=splits,
                pair_losses=pair_losses, unique_ordered_openings=len(set(openings)),
                mean_score_margin=float(values.mean()),
                first_player_win_rate=float((values[::2] > 0).mean()),
                second_player_win_rate=float((values[1::2] > 0).mean()),
                legacy_margin_pair_score=legacy_score,
                a_seconds_per_move=saved["a_sec_per_move"],
                b_seconds_per_move=saved["b_sec_per_move"],
                final_seed=config["final_seed"], matched_search=expected_args,
                independent_python_replay_verified=True,
                checkpoint_sha256=command["checkpoints"],
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_samples=BOOTSTRAP_SAMPLES)


def iterations(folder: Path, config: dict[str, Any], state: dict[str, Any],
               league: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check learner/collection ancestry and every replay manifest before tabulating."""
    rows, results = [], []
    cumulative_steps = 0
    count = state["completed_iterations"]
    assert len(league["history"]) == count
    checkpoint_dir = Path(config["checkpoint_dir"])
    tag = checkpoint_dir.name
    for i in range(1, count + 1):
        text = (folder / f"train-it{i}.log").read_text()
        trained = result(folder / f"train-it{i}.log")
        assert trained["iteration"] == i
        assert trained["checkpoint"] == str(checkpoint_dir / f"iter{i}.pt")
        assert trained["resumed_from"] == str(checkpoint_dir / f"iter{i - 1}.pt")
        entry, = [x for x in league["history"] if x["iteration"] == i]
        checkpoint_sha = state["checkpoints"][trained["checkpoint"]]
        assert entry["candidate_checkpoint_sha256"] == checkpoint_sha
        iteration_log = (folder / f"iteration-{i}.log").read_text()
        collect = re.search(r"Iter (\d+): self-play with (\S+)", iteration_log)
        assert collect and int(collect[1]) == i - 1
        assert collect[2] == str(checkpoint_dir / f"iter{i - 1}.pt")
        for process in range(4):
            log = (folder / f"collect-it{i - 1}-p{process}.log").read_text()
            assert len(re.findall(r"^Game \d+/25:", log, re.MULTILINE)) == 25
            assert "Game 25/25:" in log and "Traceback" not in log
        manifest = (folder / f"dataset-{tag}-it{i}.txt").read_text().splitlines()
        directories = [line for line in manifest if line and not line.startswith("#")]
        assert all(f"/{tag}/policy-it" in line for line in directories)
        replay = {int(re.search(r"policy-it(\d+)", line)[1]) for line in directories}
        assert replay == set(range(max(0, i - 8), i))
        policy_log = (folder / f"policy-it{i - 1}.log").read_text()
        assert "Policy-labelled dataset ready:" in policy_log
        labelled = re.search(r"Policy labels: 100/100 games ready, (\d+) exact policies",
                             policy_log)
        assert labelled
        timing = load(folder / f"timing/it{i}.json")
        assert timing["iteration"] == i
        cumulative_steps += trained["steps_completed"]
        eligible = re.search(r"dataset: ([\d,]+) positions", text)
        exact_values = re.search(r"([\d,]+) with exact value targets", text)
        assert eligible and exact_values
        row = dict(iteration=i, resumed_from=trained["resumed_from"],
                   checkpoint=trained["checkpoint"], checkpoint_sha256=checkpoint_sha,
                   steps=trained["steps_completed"], cumulative_steps=cumulative_steps,
                   eligible_positions=int(eligible[1].replace(",", "")),
                   exact_value_positions=int(exact_values[1].replace(",", "")),
                   train_positions=trained["positions"], val_positions=trained["val_positions"],
                   train_loss=trained["train_loss"], val_loss=trained["val_loss"],
                   train_policy_accuracy=trained["train_policy_acc"],
                   val_policy_accuracy=trained["val_policy_acc"],
                   train_value_accuracy=trained["train_value_acc"],
                   val_value_accuracy=trained["val_value_acc"],
                   fresh_exact_policies=int(labelled[1]),
                   collection_and_labelling_wall_seconds=timing["collect"],
                   trainer_reported_seconds=trained["elapsed_seconds"],
                   training_stage_wall_seconds=timing["train"],
                   development_arena_wall_seconds=timing["arena"],
                   development_wins=entry["vs_champion"]["a_wins"],
                   development_games=entry["vs_champion"]["games"], promoted=entry["promoted"])
        for split in ("train", "val"):
            for key, value in trained[f"{split}_exact_policy"].items():
                row[f"{split}_exact_policy_{key}"] = value
        rows.append(row)
        results.append(trained)
    return rows, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "data")
    args = parser.parse_args()
    folder = args.results
    archive = HERE / "data/results.tgz"
    assert sha(archive) == ARCHIVE_SHA, "Preserved uploaded archive changed"
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            if member.isfile():
                target = (folder / member.name).resolve()
                assert target.is_relative_to(folder.resolve())
                assert sha(target) == hashlib.sha256(packed.extractfile(member).read()).hexdigest()
    config, state, league = (load(folder / name) for name in
                             ("config.json", "state.json", "league.json"))
    assert sha(folder / "config.json") == (folder / "config.sha256").read_text().strip()
    assert state["phase"] == "complete" and state["completed_iterations"] == 15
    assert config["hours"] == 8 and config["environment"]["GATE"] == "0"
    assert config["environment"]["EXTRA_DATA"] == config["environment"]["START_FROM"] == ""
    commit = (folder / "code-commit.txt").read_text().strip()
    sources = source_provenance(config, commit)
    match = final_match(folder, config, state)
    rows, trained = iterations(folder, config, state, league)
    summary = dict(
        status="verified", declared_code_commit=commit, source_files_verified=len(sources),
        source_archive_sha256=ARCHIVE_SHA, source_archive_bytes=archive.stat().st_size,
        completed_iterations=len(rows), active_work_seconds=state["active_seconds"],
        active_work_hours=state["active_seconds"] / 3600,
        cumulative_optimizer_updates=sum(row["steps"] for row in rows),
        trainer_reported_seconds=sum(row["trainer_reported_seconds"] for row in rows),
        training_stage_wall_seconds=sum(row["training_stage_wall_seconds"] for row in rows),
        collection_and_labelling_wall_seconds=sum(
            row["collection_and_labelling_wall_seconds"] for row in rows),
        development_arena_wall_seconds=sum(row["development_arena_wall_seconds"] for row in rows),
        fresh_games=100 * len(rows), fresh_exact_policies=sum(
            row["fresh_exact_policies"] for row in rows),
        saved_league_champion=league["champion"], final_match=match,
        recommendation="Retain verified iter15 as the next starting incumbent and final baseline",
        checkpoint_bytes_verified=False, compiled_extension_bytes_verified=False,
        raw_training_data_available=False,
        limitations=["Final result is versus iter0 with solver34/50k and four-edge openings",
                     "No Dabble, PRsBoxes, BoxesZero or separate empty-board measurement",
                     "This run predates proof-filter target correction and canonical speedup",
                     "Replay diagnostics use changing populations and are not independent tests",
                     "Checkpoint and native-extension hashes are recorded; their bytes are absent"])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", summary), ("train-results.json", trained),
                        ("verified-source-hashes.json", sources)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    with (args.output / "iterations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
