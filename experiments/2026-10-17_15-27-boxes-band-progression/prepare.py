"""Freeze development roots and cost bounded root/played-child labels by edge count.

Replays existing arena games; generates no new games and uses no neural inference.
The frozen sample is selected before solving. Unknowns stay unknown, never mistakes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import alpha_go_cpp as cpp
import numpy as np

from alpha_go.boxes.rules import BoxesBoard, geometry
from alpha_go.boxes.search_probe import Position, load_positions, sha256, write_json
from alpha_go.boxes.symmetry import edge_permutation, transforms
from alpha_go.boxes.validation import replay_group


def freeze(games: Path, output: Path, count: int, seed: int) -> list[dict[str, Any]]:
    """Select equal counts at 35..40, one root per canonical opening family per band."""
    geo = geometry(5, 5)
    permutations = [edge_permutation(geo, k) for k in transforms(5, 5)]
    source_hash = sha256(games)
    pool: dict[int, list[dict[str, Any]]] = {n: [] for n in range(35, 41)}
    capture_counts: Counter[int] = Counter()
    rows = [json.loads(line) for line in games.read_text().splitlines() if line.strip()]
    assert len(rows) % 2 == 0
    for index, row in enumerate(rows):
        assert row['index'] == index and row['a_first'] == (index % 2 == 0)
        if index % 2:
            assert row['moves'][:4] == rows[index - 1]['moves'][:4]
        assert row['opening'] == 4 and len(row['moves']) == 60
        moves = [int(geo.lattice_edge[r, c]) for r, c in row['moves']]
        assert len(set(moves)) == 60 and min(moves) >= 0
        canonical_opening = min(tuple(int(p[e]) for e in moves[:4]) for p in permutations)
        family = 'arena-opening4:' + hashlib.sha256(bytes(canonical_opening)).hexdigest()
        training_family = replay_group({'moves': np.asarray(row['moves'])}, geo).hex()
        native, python = cpp.BoxesBoard(5, 5), BoxesBoard(5, 5)
        for move_index, edge in enumerate(moves):
            undrawn = 60 - move_index
            assert np.array_equal(native.to_numpy(), python.to_numpy())
            assert native.to_play() == python.to_play()
            if undrawn in pool:
                if cpp.BoxesSearchState(native).prefix():
                    capture_counts[undrawn] += 1
                else:
                    player_a = (native.to_play() == 1) == row['a_first']
                    position = Position(5, 5, moves[:move_index],
                                        f'final-games.jsonl:game{index}', source_hash, family)
                    pool[undrawn].append(dict(
                        position=asdict(position), undrawn=undrawn, game=index,
                        played_edge=edge, mover='iter15' if player_a else 'iter0',
                        training_family=training_family,
                        canonical_state=[int(min(sum(1 << int(p[e]) for e in moves[:move_index])
                                                 for p in permutations)),
                                         int(native.to_play()), int(native.margin())]))
            assert python.play_edge(edge)
            native.play_edge(edge)
        assert int(python.score()) * (1 if row['a_first'] else -1) == row['a_margin']
    selected = []
    counts = {}
    for lo, hi in [(35, 36), (37, 38), (39, 40)]:
        families: set[str] = set()
        states: set[tuple[int, ...]] = set()
        band = []
        for n in [lo, hi]:
            candidates = pool[n].copy()
            random.Random(seed + n).shuffle(candidates)
            chosen = []
            for row in candidates:
                family = row['position']['family']
                key = tuple(row['canonical_state'])
                if family not in families and key not in states:
                    chosen.append(row)
                    families.add(family)
                    states.add(key)
                if len(chosen) == count:
                    break
            assert len(chosen) == count, (n, len(chosen), count)
            for sample_index, row in enumerate(chosen):
                row.update(id=f'u{n}-{sample_index:03d}', band=f'{lo}..{hi}',
                           sample_index=sample_index)
            band.extend(chosen)
            counts[str(n)] = dict(quiet_eligible=len(candidates),
                                  forced_excluded=capture_counts[n], frozen=len(chosen))
        path = output / f'positions-{lo}-{hi}.jsonl'
        path.write_text(''.join(json.dumps(row['position']) + '\n' for row in band))
        assert len(load_positions(path, 5, 5, lo, hi)) == 2 * count
        selected.extend(band)
    write_json(output / 'sample.json', dict(
        source_sha256=source_hash, seed=seed, bands=counts, rows=selected,
        scope='Development monitor from completed iter15-vs-iter0 arena; never training data',
        selection='Decision roots without a forced prefix; equal samples per undrawn count; '
                  'unique opening families per band',
        family_note='Both sides of an opening pair share a symmetry-canonical four-edge family; '
                    'eight-edge training family is also retained for exclusion.'))
    return selected


def label(task: tuple[dict[str, Any], int, int]) -> dict[str, Any]:
    """Cold table per root; played child reuses it, preserving extra-turn perspective."""
    row, budget, entries = task
    board = Position(**row['position']).board()
    solver = cpp.BoxesSolver(5, 5, entries)
    result = {key: row[key] for key in ('id', 'band', 'undrawn', 'game', 'mover', 'played_edge')}
    result['family'] = row['position']['family']
    values: dict[str, int | None] = {}
    seconds, nodes = {}, {}
    root_player = int(board.to_play())
    for name in ['root', 'played_child']:
        mask = int(board.edges())
        start = time.monotonic()
        remaining = solver.value_within(mask, budget)
        seconds[name] = time.monotonic() - start
        nodes[name] = int(solver.nodes())
        if remaining is None:
            values[name] = None
        else:
            remaining = int(remaining)
            margin = int(board.margin()) + remaining
            values[name] = margin if int(board.to_play()) == root_player else -margin
            result.setdefault('remaining_values', {})[str(mask)] = remaining
        if name == 'root':
            board.play_edge(row['played_edge'])
    root_value, child_value = values['root'], values['played_child']
    regret = (root_value - child_value
              if root_value is not None and child_value is not None else None)
    assert regret is None or regret >= 0
    result.update(values=values, regret=regret, seconds=seconds, nodes=nodes)
    return result


def main() -> None:
    """Preserve positions, per-solve records, exact-only reusable cache and coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-edge', type=int, default=40)
    parser.add_argument('--pilot-per-edge', type=int, default=6)
    parser.add_argument('--node-budget', type=int, default=2_000_000)
    parser.add_argument('--table-entries', type=int, default=1 << 22)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261017)
    args = parser.parse_args()
    assert 0 < args.pilot_per_edge <= args.per_edge
    assert min(args.node_budget, args.table_entries, args.workers) > 0
    args.output.mkdir(parents=True, exist_ok=False)
    source_copy = args.output / 'final-games.jsonl'
    source_copy.write_bytes(args.games.read_bytes())
    (args.output / 'prepare-snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output / 'config.json', dict(
        arguments=vars(args), code_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=sha256(Path(__file__)), extension_sha256=sha256(Path(cpp.__file__))))
    selected = freeze(source_copy, args.output, args.per_edge, args.seed)
    pilot = [row for row in selected if row['sample_index'] < args.pilot_per_edge]
    print(f'Frozen {len(selected)} roots; bounded cost pilot on {len(pilot)}', flush=True)
    started = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(label, (row, args.node_budget, args.table_entries)) for row in pilot]
        for job in as_completed(jobs):
            records.append(job.result())
            write_json(args.output / 'labels-progress.json', records)
            print(f'{len(records)}/{len(pilot)} labelled; '
                  f'{time.monotonic() - started:.1f}s elapsed', flush=True)
    records.sort(key=lambda row: row['id'])
    write_json(args.output / 'records.json', records)
    cache: dict[str, int] = {}
    for row in records:
        for key, value in row.get('remaining_values', {}).items():
            assert key not in cache or cache[key] == value
            cache[key] = value
    write_json(args.output / 'labels.json', dict(
        schema=1, rows=5, cols=5, remaining_values=cache, node_budget=args.node_budget))
    summary = {}
    for band in ['35..36', '37..38', '39..40']:
        rows = [row for row in records if row['band'] == band]
        known = [row for row in rows if row['regret'] is not None]
        summary[band] = dict(
            attempted=len(rows), root_known=sum(row['values']['root'] is not None for row in rows),
            root_and_child_known=len(known), unknown=len(rows) - len(known),
            total_solve_seconds=sum(sum(row['seconds'].values()) for row in rows),
            total_nodes=sum(sum(row['nodes'].values()) for row in rows),
            observed_move_mistakes=(sum(row['regret'] > 0 for row in known) if known else None),
            observed_move_mean_regret=(sum(row['regret'] for row in known) / len(known)
                                       if known else None))
    write_json(args.output / 'summary.json', dict(
        scope='Tiny solve-cost pilot; observed moves belong to BOTH nets on different positions. '
              'No checkpoint strength comparison, no full optimal-policy labels.',
        regret_units='Final score-margin points, not boxes captured',
        node_budget=args.node_budget, wall_seconds=time.monotonic() - started, bands=summary))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
