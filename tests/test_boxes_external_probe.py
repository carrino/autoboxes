"""External decision diagnostics preserve ownership, exact labels, and resumability."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from alpha_go.boxes.external_probe import (
    TABLES,
    FrozenRoot,
    aggregate,
    call_worker,
    compare,
    digest,
    read_roots,
    run,
    score_move,
)
from alpha_go.boxes.oracle import Oracle
from alpha_go.boxes.rules import BoxesBoard


def fixture_tables(directory: Path) -> Path:
    """Build independently solved tiny roots, including an already captured box."""
    directory.mkdir()
    positions = [dict(position=i, rows=1, cols=2, prefix=prefix, family=f"f{i}")
                 for i, prefix in enumerate(([], [0, 2, 4, 5]))]
    (directory/TABLES[0]).write_text("".join(json.dumps(p)+"\n" for p in positions))
    roots, children = [], []
    oracle = Oracle(1, 2)
    for p in positions:
        board = BoxesBoard(1, 2)
        for edge in p["prefix"]:
            board.play_edge(edge)
        legal = board.get_legal_moves_flat()
        roots.append(dict(position=p["position"], root_final=oracle.final_margin(board),
                          undrawn=len(legal)))
        children.extend(dict(position=p["position"], edge=edge,
                             final_margin_for_root_mover=board.margin()
                             + oracle.child_value(board.edges, edge)) for edge in legal)
    for filename, rows in zip(TABLES[1:], (roots, children)):
        with (directory/filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return directory


def config(tmp_path: Path, expression: str) -> Path:
    """A test worker that exercises the real fresh-process JSON exchange."""
    worker = tmp_path/"worker.py"
    worker.write_text("import json,sys\nr=json.load(sys.stdin)\n"+expression+"\n")
    path = tmp_path/"config.json"
    path.write_text(json.dumps(dict(name="test-worker", command=[sys.executable, str(worker)],
                                    artifacts={str(worker): digest(worker)},
                                    provenance={"scope": "test only"})))
    return path


class TestPositions:
    def test_replay_capture_retains_player_and_score(self, tmp_path: Path) -> None:
        roots = read_roots(fixture_tables(tmp_path/"tables"))
        request = roots[1].request(.1, 7)
        assert request["to_play"] == 2
        assert request["scores"] == [0, 1]
        assert request["board"][1][1] == 2
        assert request["prefix"] == [0, 2, 4, 5]

    def test_rejects_partial_children(self, tmp_path: Path) -> None:
        tables = fixture_tables(tmp_path/"tables")
        path = tables/TABLES[2]
        path.write_text("\n".join(path.read_text().splitlines()[:-1])+"\n")
        with pytest.raises(ValueError, match="cover legal"):
            read_roots(tables)

    def test_rejects_duplicate_replay_edge(self, tmp_path: Path) -> None:
        tables = fixture_tables(tmp_path/"tables")
        path = tables/TABLES[0]
        positions = [json.loads(line) for line in path.read_text().splitlines()]
        positions[1]["prefix"].append(0)
        path.write_text("".join(json.dumps(p)+"\n" for p in positions))
        with pytest.raises(ValueError, match="illegal replay"):
            read_roots(tables)


class TestMetrics:
    def test_winning_move_can_lose_margin_without_losing_game(self) -> None:
        root = FrozenRoot(0, 5, 5, [], "family", {0: 5, 1: 1, 2: -1}, 5)
        record = score_move(root, 1)
        assert record["margin_regret"] == 4 and record["optimal"] == 0
        assert record["win_preserved"] == 1 and record["win_to_loss"] == 0
        loss = score_move(root, 2)
        assert loss["win_to_loss"] == 1 and loss["margin_regret"] == 6
        assert aggregate([record, loss])["win_preservation_rate"] == .5

    def test_draw_and_losing_roots_are_not_win_preservation_trials(self) -> None:
        a = score_move(FrozenRoot(0, 1, 2, [], "a", {0: 0, 1: -2}, 0), 1)
        b = score_move(FrozenRoot(1, 1, 2, [], "b", {0: 2, 1: 0}, 2), 1)
        assert aggregate([a, b])["winning_roots"] == 1
        assert aggregate([a, b])["win_to_loss"] == 0
        assert aggregate([a, b])["win_to_draw"] == 1

    @pytest.mark.parametrize("edge", [True, -1, 99])
    def test_illegal_moves_are_errors(self, edge: Any) -> None:
        with pytest.raises(ValueError, match="illegal edge"):
            score_move(FrozenRoot(0, 1, 2, [], "a", {0: 0}, 0), edge)

    def test_paired_comparison_uses_only_common_positions(self) -> None:
        root = FrozenRoot(0, 5, 5, [], "a", {0: 1, 1: -1}, 1)
        other = FrozenRoot(1, 5, 5, [], "b", {0: 1, 1: -1}, 1)
        new = [score_move(root, 0)]
        old = [score_move(root, 1), score_move(other, 1)]
        result = compare(new, old, resamples=100)
        assert result["common_positions"] == 1
        assert result["metrics"]["margin_regret"]["ci95"] == [-2, -2]
        assert result["metrics"]["win_preserved"]["delta"] == 1


class TestWorker:
    def test_run_resume_and_artifact_integrity(self, tmp_path: Path) -> None:
        tables = fixture_tables(tmp_path/"tables")
        path = config(tmp_path, "print(json.dumps({'edge': 1}))")
        args = argparse.Namespace(tables=tables, output=tmp_path/"out", engine_config=path,
                                  seconds=.1, timeout_slack=2, seed=7, limit=0, baseline=None)
        report = run(args)
        assert report["complete"] and report["aggregate"]["positions"] == 2
        saved = [(p, p.stat().st_mtime_ns) for p in args.output.glob("position-*.json")]
        assert run(args) == report
        assert all(p.stat().st_mtime_ns == stamp for p, stamp in saved)
        args.seconds = .2
        with pytest.raises(ValueError, match="Resume settings"):
            run(args)
        (tmp_path/"worker.py").write_text("raise RuntimeError('changed')")
        with pytest.raises(ValueError, match="artifact changed"):
            run(args)

    def test_illegal_worker_response_is_not_published(self, tmp_path: Path) -> None:
        args = argparse.Namespace(tables=fixture_tables(tmp_path/"tables"),
                                  output=tmp_path/"out",
                                  engine_config=config(tmp_path, "print('{\"edge\": 999}')"),
                                  seconds=.1, timeout_slack=2, seed=7, limit=0, baseline=None)
        with pytest.raises(ValueError, match="illegal edge"):
            run(args)
        assert not list(args.output.glob("position-*.json"))

    def test_timeout_reaps_worker(self, tmp_path: Path) -> None:
        with pytest.raises(subprocess.TimeoutExpired):
            call_worker([sys.executable, "-c", "import time; time.sleep(10)"], {}, tmp_path, .05)
