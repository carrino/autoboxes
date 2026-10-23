"""Independent saved-record verifier, preserved from the inline audit invocation.

No project game/arena code, Torch, native solver, or opponent is imported. The
board is replayed from lattice edges and four-edge box sets. NumPy is used only
to regenerate the frozen opening RNG and the 10,000-draw pair bootstrap.

Run from the repository root with the matching source checkout and pinned Qdab release:
  python verify_benchmark.py INPUT_DIR OUTPUT_DIR [CHECKPOINT [QDAB_RELEASE]]

This standalone form was also run successfully against the uploaded archive.
It performs saved-record verification only, not a new strength match.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_of(move: list[int]) -> int:
    row, col = move
    assert type(row) is int and type(col) is int
    assert 0 <= row <= 10 and 0 <= col <= 10 and (row + col) % 2 == 1
    return (row // 2) * 5 + col // 2 if row % 2 == 0 else 30 + (row // 2) * 6 + col // 2


def move_of(edge: int) -> list[int]:
    if edge < 30:
        return [2 * (edge // 5), 2 * (edge % 5) + 1]
    return [2 * ((edge - 30) // 6) + 1, 2 * ((edge - 30) % 6)]


def hv(edges: set[int] | list[int]) -> tuple[int, int]:
    horizontal = vertical = 0
    for edge in edges:
        if edge < 30:
            vertical |= 1 << ((edge % 5) * 6 + edge // 5)
        else:
            horizontal |= 1 << (edge - 30)
    return horizontal, vertical


def elo(score: float) -> float:
    return 400 * math.log10(score / (1 - score))


def main() -> None:
    started = time.monotonic()
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "work/qdab-gpu-benchmark-1yafbn9k")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else base.with_name(base.name + "-analysis")
    out.mkdir(exist_ok=True)
    config = json.loads((base / "config.json").read_text())
    saved = json.loads((base / "summary.json").read_text())
    config_hash = sha(base / "config.json")
    boxes = [
        frozenset([row * 5 + col, (row + 1) * 5 + col, 30 + row * 6 + col, 30 + row * 6 + col + 1])
        for row in range(5)
        for col in range(5)
    ]

    # Regenerate the exact opening, seed, panel and side schedule independently.
    rng = np.random.default_rng(config["seed"])
    schedule = []
    for panel, count in [("paired", 100), ("empty", 20)]:
        for pair in range(count // 2):
            legal = list(range(60))
            opening = []
            for _ in range(4 if panel == "paired" else 0):
                edge = int(rng.choice(legal))
                legal.remove(edge)
                opening.append(move_of(edge))
            for first in [True, False]:
                index = len(schedule)
                schedule.append(
                    dict(
                        index=index,
                        panel=panel,
                        pair=pair,
                        a_first=first,
                        seed=config["seed"] + index,
                        opening=opening,
                    )
                )
    assert schedule == config["schedule"]
    paths = sorted((base / "games").glob("*.json"))
    assert [path.stem for path in paths] == [f"{i:04d}" for i in range(120)]
    attempt_paths = sorted((base / "attempts").glob("*/attempt.json"))
    assert len(attempt_paths) == 120
    hashes = {str(path.relative_to(base)): sha(path) for path in base.rglob("*") if path.is_file()}
    rows, decision_rows, qdab_log_stats = [], [], []

    for path, slot in zip(paths, schedule):
        row = json.loads(path.read_text())
        assert row["slot"] == slot and row["config_sha256"] == config_hash
        assert len(row["moves"]) == 60
        assert row["moves"][: len(slot["opening"])] == slot["opening"]
        played, claimed = set(), set()
        score, counts = [0, 0], [0, 0]
        mover = turn = 0
        frames = []
        edges = [edge_of(move) for move in row["moves"]]
        for index, edge in enumerate(edges):
            assert edge not in played and 0 <= edge < 60
            h, v = hv(played)
            frames.append(dict(H=h, V=v, S=score.copy(), Now=mover, Turn=turn))
            if index >= len(slot["opening"]):
                counts[mover] += 1
            played.add(edge)
            fresh = [
                box for box, sides in enumerate(boxes) if box not in claimed and sides <= played
            ]
            score[mover] += len(fresh)
            claimed.update(fresh)
            if not fresh:
                mover = 1 - mover
                turn += 1
        assert len(played) == 60 and len(claimed) == 25 and sum(score) == 25
        p1margin = score[0] - score[1]
        amargin = p1margin * (1 if slot["a_first"] else -1)
        assert amargin == row["a_margin"]
        assert row["result"] == f"{'B' if p1margin > 0 else 'W'}+{abs(p1margin):.1f}"
        assert row["termination"] == "all_edges"
        aplayer = 0 if slot["a_first"] else 1
        bplayer = 1 - aplayer
        assert row["a_moves"] == counts[aplayer] and row["b_moves"] == counts[bplayer]
        assert all(
            math.isfinite(row[f"{side}_seconds"]) and row[f"{side}_seconds"] >= 0
            for side in ["a", "b"]
        )
        qindices = {
            i
            for i, frame in enumerate(frames)
            if i >= len(slot["opening"]) and frame["Now"] == bplayer
        }
        covered = set()
        last = -1
        rpc_seconds = wall_seconds = 0.0
        for decision in row["qdab_decisions"]:
            index = decision["move_count"]
            assert index > last and index in qindices
            last = index
            bundle = decision["edges"]
            assert bundle and edges[index : index + len(bundle)] == bundle
            indices = set(range(index, index + len(bundle)))
            assert indices <= qindices and not indices & covered
            covered |= indices
            request, response = decision["rpc_request"], decision["rpc_response"]
            param = request["params"][0]
            assert request["id"] == response["id"] == 1
            assert request["method"] == "Server.MakeMove"
            assert param["Algorithm"] == "quctann" and param["Board"] == frames[index]
            assert param["Timeout"] == 5000 and response["error"] is None
            assert response["result"] == dict(zip(["H", "V"], hv(bundle)))
            assert decision["requested_seconds"] == config["seconds"] == 5
            assert decision["hard_timeout_seconds"] == config["hard_timeout"] == 30
            assert 0 <= decision["decision_seconds"] <= decision["wall_seconds"] <= 30.5
            rpc_seconds += decision["decision_seconds"]
            wall_seconds += decision["wall_seconds"]
            decision_rows.append(
                dict(
                    game=slot["index"],
                    panel=slot["panel"],
                    autoboxes_first=slot["a_first"],
                    a_margin=amargin,
                    move_count=index,
                    undrawn=60 - index,
                    qdab_turn=frames[index]["Turn"],
                    bundle_edges=len(bundle),
                    rpc_seconds=decision["decision_seconds"],
                    wall_seconds=decision["wall_seconds"],
                )
            )
        assert covered == qindices and row["b_seconds"] >= wall_seconds - 1e-6
        attempt_path = base / "attempts" / row["attempt"] / "attempt.json"
        attempt = json.loads(attempt_path.read_text())
        assert attempt["slot"] == slot and attempt["status"] == "complete"
        assert row["attempt"] == f"game-{slot['index']:04d}-1"
        assert math.isfinite(attempt["elapsed_seconds"])
        assert attempt["elapsed_seconds"] >= row["a_seconds"] + row["b_seconds"] - 1e-6
        log = (attempt_path.parent / "qdab.log").read_text()
        assert log.count("Server runing on 127.0.0.1:12345") == len(row["qdab_decisions"])
        for line in log.splitlines():
            match = re.search(
                r"Turn: (\d+) , Elapse: (.*?) , Sim: (\d+) .*?"
                r"Child: (\d+) , MaxDepth: (-?\d+)",
                line,
            )
            if match:
                qdab_log_stats.append(
                    dict(
                        game=slot["index"],
                        turn=int(match[1]),
                        simulations=int(match[3]),
                        children=int(match[4]),
                        max_depth=int(match[5]),
                    )
                )
        rows.append(
            dict(
                game=slot["index"],
                panel=slot["panel"],
                pair=slot["pair"],
                autoboxes_first=slot["a_first"],
                a_margin=amargin,
                p1_score=score[0],
                p2_score=score[1],
                a_moves=counts[aplayer],
                b_moves=counts[bplayer],
                a_seconds=row["a_seconds"],
                b_seconds=row["b_seconds"],
                qdab_rpc_calls=len(row["qdab_decisions"]),
                qdab_rpc_seconds=rpc_seconds,
                qdab_wall_seconds=wall_seconds,
                attempt_elapsed_seconds=attempt["elapsed_seconds"],
                config_sha256=config_hash,
                game_sha256=sha(path),
            )
        )

    # Independent cluster bootstrap and binomial whole-pair sign randomization.
    paired = [row for row in rows if row["panel"] == "paired"]
    pwin = sum(row["a_margin"] > 0 for row in paired) / len(paired)
    pairs = []
    for index in range(0, len(paired), 2):
        a, b = paired[index : index + 2]
        assert a["pair"] == b["pair"] and a["autoboxes_first"] and not b["autoboxes_first"]
        wins = int(a["a_margin"] > 0) + int(b["a_margin"] > 0)
        pairs.append(
            dict(
                pair=a["pair"],
                first_game=a["game"],
                second_game=b["game"],
                first_margin=a["a_margin"],
                second_margin=b["a_margin"],
                wins=wins,
                game_score=wins / 2,
                mean_margin=(a["a_margin"] + b["a_margin"]) / 2,
            )
        )
    pair_scores = np.array([pair["game_score"] for pair in pairs])
    rng = np.random.default_rng(config["seed"])
    samples = rng.integers(0, len(pairs), (10000, len(pairs)))
    ci = np.quantile(pair_scores[samples].mean(axis=1), [0.025, 0.975]).tolist()
    nonzero = [pair for pair in pairs if pair["wins"] != 1]
    positive = sum(pair["wins"] == 2 for pair in nonzero)
    pvalue = min(
        1,
        2
        * sum(math.comb(len(nonzero), k) for k in range(min(positive, len(nonzero) - positive) + 1))
        / (2 ** len(nonzero)),
    )
    assert pwin == saved["panels"]["paired"]["game_score"]
    assert ci == saved["panels"]["paired"]["game_score_ci95"]
    assert abs(elo(pwin) - saved["panels"]["paired"]["relative_elo"]) < 1e-12
    assert abs(pvalue - saved["panels"]["paired"]["paired_win_pvalue"]) < 1e-16
    panels = {}
    for panel in ["paired", "empty"]:
        subset = [row for row in rows if row["panel"] == panel]
        margins = [row["a_margin"] for row in subset]
        wins, losses = [x for x in margins if x > 0], [x for x in margins if x < 0]
        data = dict(
            games=len(subset),
            wins=len(wins),
            losses=len(losses),
            draws=sum(x == 0 for x in margins),
            win_rate=len(wins) / len(margins),
            mean_margin=statistics.mean(margins),
            median_margin=statistics.median(margins),
            margin_histogram=dict(sorted(Counter(margins).items())),
            win_margin_mean=statistics.mean(wins),
            loss_margin_mean=statistics.mean(losses),
            one_box_losses=margins.count(-1),
            three_or_fewer_box_losses=sum(-3 <= x < 0 for x in margins),
            narrow_wins_up_to3=sum(0 < x <= 3 for x in margins),
            wins_at_least7=sum(x >= 7 for x in margins),
            first_wins=sum(row["autoboxes_first"] and row["a_margin"] > 0 for row in subset),
            second_wins=sum(not row["autoboxes_first"] and row["a_margin"] > 0 for row in subset),
            first_games=sum(row["autoboxes_first"] for row in subset),
            second_games=sum(not row["autoboxes_first"] for row in subset),
        )
        for side in ["a", "b"]:
            data[f"{side}_moves"] = sum(row[f"{side}_moves"] for row in subset)
            data[f"{side}_seconds"] = sum(row[f"{side}_seconds"] for row in subset)
            data[f"{side}_seconds_per_edge"] = data[f"{side}_seconds"] / data[f"{side}_moves"]
        assert data["wins"] == saved["panels"][panel]["wins"]
        assert data["mean_margin"] == saved["panels"][panel]["mean_margin"]
        for side in ["a", "b"]:
            assert (
                abs(
                    data[f"{side}_seconds_per_edge"]
                    - saved["panels"][panel][f"{side}_sec_per_move"]
                )
                < 1e-12
            )
        panels[panel] = data

    code_hash_checks = {
        name: sha(Path(name)) == value for name, value in config["code_sha256"].items()
    }
    assert all(code_hash_checks.values())
    checkpoint = Path(sys.argv[3] if len(sys.argv) > 3 else "work/qdab-latest-upload/control.pt")
    assert sha(checkpoint) == config["checkpoint_sha256"]
    release = Path(sys.argv[4] if len(sys.argv) > 4 else "work/qdab-release")
    artifact_checks = {
        name: sha(release / name) == value for name, value in config["qdab_artifacts"].items()
    }
    assert all(artifact_checks.values())
    qstats = {}
    for label, low, high in [
        ("early_non_capture_turns_0_14", 0, 14),
        ("middle_non_capture_turns_15_23", 15, 23),
        ("later_non_capture_turns_24_plus", 24, 60),
    ]:
        subset = [
            row for row in qdab_log_stats if low <= row["turn"] <= high and row["simulations"] > 0
        ]
        qstats[label] = dict(
            logged_searches=len(subset),
            median_simulations=statistics.median(row["simulations"] for row in subset)
            if subset
            else None,
            median_children=statistics.median(row["children"] for row in subset)
            if subset
            else None,
            median_depth=statistics.median(row["max_depth"] for row in subset) if subset else None,
        )
    pair_counts = Counter(pair["wins"] for pair in pairs)
    review = dict(
        schema=1,
        status="verified_saved_records",
        input=str(base),
        config_sha256=config_hash,
        checkpoint_sha256=config["checkpoint_sha256"],
        verification=dict(
            full_games=120,
            edges_replayed=7200,
            first_principles_lattice_replay=True,
            qdab_rpc_calls=len(decision_rows),
            qdab_edges_verified=sum(row["bundle_edges"] for row in decision_rows),
            qdab_request_states_verified=True,
            qdab_responses_match_played_bundles=True,
            frozen_schedule_regenerated=True,
            all_game_config_hashes_match=True,
            attempts=120,
            complete_attempts=120,
            retries=0,
            source_code_hashes_checked=len(code_hash_checks),
            source_code_hashes_all_match_current=True,
            qdab_artifacts_match_local_pinned_release=artifact_checks,
            separate_previously_uploaded_checkpoint_matches=True,
            checkpoint_in_this_upload=False,
        ),
        panels=panels,
        paired_rating=dict(
            opening_pairs=50,
            game_score=pwin,
            pair_cluster_bootstrap_seed=config["seed"],
            resamples=10000,
            score_ci95=ci,
            relative_elo=elo(pwin),
            elo_ci95=[elo(value) for value in ci],
            exact_whole_pair_sign_test_pvalue=pvalue,
            pair_patterns=dict(
                lose_both=pair_counts[0], split=pair_counts[1], win_both=pair_counts[2]
            ),
            saved_summary_reproduced=True,
        ),
        timing=dict(
            total_attempt_seconds=sum(row["attempt_elapsed_seconds"] for row in rows),
            total_a_seconds=sum(row["a_seconds"] for row in rows),
            total_b_seconds=sum(row["b_seconds"] for row in rows),
            qdab_rpc_seconds=sum(row["rpc_seconds"] for row in decision_rows),
            qdab_wall_seconds=sum(row["wall_seconds"] for row in decision_rows),
            max_qdab_decision_seconds=max(row["rpc_seconds"] for row in decision_rows),
            max_qdab_wall_seconds=max(row["wall_seconds"] for row in decision_rows),
            rpc_at_least5s=sum(row["rpc_seconds"] >= 5 for row in decision_rows),
            requested_seconds=5,
            hard_timeout_seconds=30,
            recorded_runtime=config["runtime"],
        ),
        qdab_log_search_diagnostics=qstats,
        limitations=[
            (
                "This verifies uploaded records and consistency, not a re-execution of "
                "GPU inference or wall-clock measurement."
            ),
            (
                "The frozen GPU checkpoint is omitted from this archive; its recorded "
                "hash matches the separate previously uploaded checkpoint."
            ),
            (
                "Remote native extension hash is recorded but its binary is omitted; "
                "local native build may differ."
            ),
            (
                "Relative Elo is conditional on Qdab quctann requested5s versus "
                "autoboxes400sim solver34/50k on the recorded hardware, not an "
                "equal-time or absolute rating."
            ),
            (
                "Opening pairs are bootstrap units; repeated empty-board starts are "
                "descriptive and excluded from Elo."
            ),
            (
                "No exact root solves were performed; final one-box losses do not locate "
                "the decisive error or prove near-equal strength."
            ),
            "Qdab seed is not exposed; timed searches vary.",
        ],
        analysis_elapsed_seconds=time.monotonic() - started,
    )
    # Solver-free deductions added to the initial review after replay/stat verification.
    review["paired_opening_lower_bounds"] = dict(
        autoboxes_initially_winning_positions_eventually_lost_at_least=pair_counts[0],
        qdab_initially_winning_positions_eventually_lost_at_least=pair_counts[2],
        reason=(
            "Each side-swapped pair shares exactly the same post-opening state. "
            "The25-box zero-sum game cannot draw, so one physical side is winning "
            "under perfect play. A player losing both seats must have lost the "
            "initially winning seat in one game."
        ),
        limits=(
            "No exact opening values or mistake locations were computed; these are "
            "pairwise lower bounds, not solved-state labels."
        ),
    )
    review["observed_margin_interpretation"] = dict(
        mean_margin_does_not_rank_match_strength=True,
        evidence=(
            "Autoboxes paired mean margin+0.76 while winning17/100;65of83losses "
            "are−1 but all17wins are+7or larger."
        ),
        hypothesis=(
            "Consistent with Qdab prioritizing securing a win over preserving margin "
            "once losing. Does not by itself establish why our shaping/search fails."
        ),
    )
    for filename, records in [
        ("verified-games.csv", rows),
        ("paired-openings.csv", pairs),
        ("qdab-decisions.csv", decision_rows),
        ("qdab-search-log-stats.csv", qdab_log_stats),
    ]:
        with (out / filename).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (out / "review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")
    (out / "input-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(out),
                paired_rating=review["paired_rating"],
                elapsed=review["analysis_elapsed_seconds"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
