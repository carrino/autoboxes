"""Reproduce the completed proof-target overnight result and target-audit totals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARCHIVE_SHA = '8eef8346a33d2864db4f2d38af7f9d1391fb6f33d6dc491fce53ca7d585316b6'
REFERENCE_SHA = '3f17b19f3caa66905be1f33a960c038fec01fa499cda483ed00367b99e164dac'
CANDIDATE_SHA = '73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d'


def helpers() -> Any:
    """Reuse the earlier independent replay/ancestry audit without its run constants."""
    path = REPO / 'experiments/2026-10-17_15-12-boxes-solver34-learning-result/analyze.py'
    spec = importlib.util.spec_from_file_location('earlier_result_audit', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proof_audits(folder: Path, count: int) -> dict[str, Any]:
    """Verify saved collection reports and weight their means by searched rows."""
    totals: Counter[str] = Counter()
    bands: dict[str, Counter[str]] = {}
    mass = 0.0
    band_mass: Counter[str] = Counter()
    per_iteration = []
    fields = ('positions', 'searched_positions', 'changed_positions', 'unsearched_positions')
    for iteration in range(count):
        report = json.loads((folder / f'targets-it{iteration}.json').read_text())
        assert report['status'] == 'complete'
        assert report['policy_target_semantics'] == 'proof_filtered_visits_v1'
        assert report['games'] == len(report['files_sha256']) == 100
        assert all(f'/selfplay-it{iteration}/' in p for p in report['files_sha256'])
        row = report['totals']
        assert row['positions'] == row['searched_positions'] + row['unsearched_positions']
        assert 0 <= row['changed_positions'] <= row['searched_positions']
        removed = row['mean_raw_mass_removed'] * row['searched_positions']
        per_iteration.append(dict(iteration=iteration, **row))
        mass += removed
        totals.update({key: row[key] for key in fields})
        for key in fields:
            assert sum(b[key] for b in report['bands'].values()) == row[key]
        band_total = 0.0
        for band, values in report['bands'].items():
            bands.setdefault(band, Counter()).update({key: values[key] for key in fields})
            if values['searched_positions']:
                assert 0 <= values['mean_raw_mass_removed'] <= 1
                weighted = values['mean_raw_mass_removed'] * values['searched_positions']
                band_mass[band] += weighted
                band_total += weighted
            else:
                assert values['mean_raw_mass_removed'] is None
        assert math.isclose(band_total, removed, abs_tol=1e-10)
    return dict(
        status='saved reports internally consistent; raw NPZ bytes are absent',
        total=dict(totals, mean_raw_mass_removed=mass / totals['searched_positions']),
        bands={band: dict(values, mean_raw_mass_removed=(band_mass[band] /
               values['searched_positions'] if values['searched_positions'] else None))
               for band, values in bands.items()}, iterations=per_iteration)


def main() -> None:
    """Validate original archive bytes, then save the independently recomputed report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'data')
    args = parser.parse_args()
    old = helpers()
    folder = args.results
    archive = HERE / 'data/results.tgz'
    assert old.sha(archive) == ARCHIVE_SHA
    with tarfile.open(archive) as packed:
        for member in packed.getmembers():
            assert member.isfile() or member.isdir()
            target = (folder / member.name).resolve()
            assert target.is_relative_to(folder.resolve())
            if member.isfile():
                stream = packed.extractfile(member)
                assert stream is not None
                assert old.sha(target) == hashlib.sha256(stream.read()).hexdigest()
    config, state, league = (old.load(folder / name) for name in
                             ('config.json', 'state.json', 'league.json'))
    assert old.sha(folder / 'config.json') == (folder / 'config.sha256').read_text().strip()
    assert state['phase'] == 'complete' and state['completed_iterations'] == 24
    assert config['reference_sha256'] == REFERENCE_SHA
    assert state['checkpoints'][str(Path(config['checkpoint_dir']) / 'iter24.pt')] == CANDIDATE_SHA
    assert config['hours'] == 8 and config['final_seed'] == 20261029
    expected_settings = dict(GATE='0', PROOF_TARGET_AUDIT='1', SP_GAMES='100', SP_SIMS='400',
                             SOLVER_N='34', SOLVER_BUDGET='50000', POLICY_MIN_UNDRAWN='35',
                             POLICY_MAX_UNDRAWN='36', POLICY_MAX_PER_GAME='2', TRAIN_EPOCHS='4',
                             TRAIN_ARGS='--lr 0.0001', START_FRACTION='0', START_FROM='',
                             EXTRA_DATA='', WINDOW='8')
    assert all(config['environment'][key] == value for key, value in expected_settings.items())
    commit = (folder / 'code-commit.txt').read_text().strip()
    sources = old.source_provenance(config, commit)
    preflight = old.load(folder / 'preflight/preflight.json')
    assert config['preflight'] == preflight
    assert preflight['policy_target_semantics'] == 'proof_filtered_visits_v1'
    assert '17 passed' in (folder / 'preflight/checks.log').read_text()
    runtime = preflight['runtime']
    assert config['code_sha256'][runtime['extension']] == runtime['extension_sha256']
    build_inputs = {}
    for absolute, expected in preflight['build_inputs_sha256'].items():
        relative = Path(absolute).relative_to(config['cwd'])
        raw = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=REPO)
        assert hashlib.sha256(raw).hexdigest() == expected
        build_inputs[str(relative)] = expected
    match = old.final_match(folder, config, state)
    rows, trained = old.iterations(folder, config, state, league)
    audits = proof_audits(folder, len(rows))
    # Reproduce the archived interval's exact seed/count, separate from the
    # earlier helper's independent seed-0 / 50,000-draw bootstrap.
    games = [json.loads(line) for line in (folder / 'final-games.jsonl').read_text().splitlines()]
    margins = np.asarray([row['a_margin'] for row in games])
    pair_scores = (margins.reshape(-1, 2) > 0).mean(axis=1)
    indices = np.random.default_rng(20261018).integers(0, 100, (10_000, 100))
    interval = np.quantile(pair_scores[indices].mean(axis=1), [.025, .975]).tolist()
    saved = old.load(folder / 'final.json')
    assert interval == saved['paired_game_score_ci95']
    match.update(archived_paired_game_score_ci95=interval,
                 archived_bootstrap_seed=20261018, archived_bootstrap_samples=10_000,
                 first_mean_margin=float(margins[::2].mean()),
                 second_mean_margin=float(margins[1::2].mean()),
                 recorded_a_seconds_per_move=saved['a_sec_per_move'],
                 recorded_b_seconds_per_move=saved['b_sec_per_move'])
    summary = dict(
        status='verified', declared_code_commit=commit, source_files_verified=len(sources),
        build_inputs_verified=len(build_inputs), source_archive_sha256=ARCHIVE_SHA,
        source_archive_bytes=archive.stat().st_size, completed_iterations=len(rows),
        active_work_seconds=state['active_seconds'],
        active_work_hours=state['active_seconds'] / 3600,
        cumulative_optimizer_updates=sum(r['steps'] for r in rows),
        trainer_reported_seconds=sum(r['trainer_reported_seconds'] for r in rows),
        training_stage_wall_seconds=sum(r['training_stage_wall_seconds'] for r in rows),
        collection_and_labelling_wall_seconds=sum(
            r['collection_and_labelling_wall_seconds'] for r in rows),
        development_arena_wall_seconds=sum(r['development_arena_wall_seconds'] for r in rows),
        fresh_games=100 * len(rows),
        fresh_exact_policies=sum(r['fresh_exact_policies'] for r in rows),
        saved_league_champion=league['champion'],
        promoted_iterations=[r['iteration'] for r in rows if r['promoted']],
        final_match=match, proof_target_audit=audits,
        recommendation=('Use verified iter24 as the new incumbent; '
                        'measure bands before extending the curriculum'),
        checkpoint_bytes_verified=False, compiled_extension_bytes_verified=False,
        raw_training_data_available=False,
        limitations=['Matched-search result versus old solver34 iter15 on four-edge openings',
                     'No strong-external-bot or separate empty-board measurement',
                     'Cannot isolate corrected targets from additional training',
                     'First/second seat performance differs substantially',
                     'Replay metrics use changing populations and are not a held-out band test',
                     'Weights, extension and raw NPZ hashes are recorded but bytes are absent'])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary.json', summary), ('train-results.json', trained),
                        ('verified-source-hashes.json', sources),
                        ('verified-build-inputs.json', build_inputs)]:
        (args.output / name).write_text(json.dumps(value, indent=2) + '\n')
    with (args.output / 'iterations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != 'proof_target_audit'}, indent=2))
    print('Proof targets:', json.dumps(audits['total']))


if __name__ == '__main__':
    main()
