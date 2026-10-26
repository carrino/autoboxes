"""Fit saved frontier37 labels, then evaluate both fixed endpoints automatically."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
BASE = Path('experiments/2026-10-08_14-42-boxes-3x3-loop')
SOURCE = BASE / 'logs/5x5-frontier37-ezpgdpxy'
SOURCE_SHA = '72c2dc043ae6d71d3affb772867e609b5ada35a42acd8023c01adfa7cb90900d'
REFERENCE_SHA = '73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d'
HELPER = REPO / 'experiments/2026-10-20_08-13-boxes-balanced-policy/run_balanced.py'
spec = importlib.util.spec_from_file_location('frontier_fit_helpers', HELPER)
assert spec and spec.loader
balanced = importlib.util.module_from_spec(spec)
spec.loader.exec_module(balanced)
stages, helpers = balanced.pilot, balanced.helpers
checked, digest, write_json = helpers.checked, helpers.digest, helpers.write_json


def create(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Freeze the completed run's verified replay and the untrained-on-it incumbent."""
    from alpha_go.boxes.frontier_fit_data import verify_sources

    source = args.source_run.resolve()
    checked(source / 'config.json', SOURCE_SHA)
    original = json.loads((source / 'config.json').read_text())
    state = json.loads((source / 'state.json').read_text())
    if state['phase'] != 'complete' or state['completed_iterations'] != 10:
        raise ValueError('Expected the completed ten-iteration frontier37 run')
    reference = Path(original['reference'])
    checked(reference, REFERENCE_SHA)
    collection = source / 'inputs/collection-config.json'
    checked(collection, original['input_sha256']['collection-config.json'])
    runtime = balanced.runtime_identity(json.loads(collection.read_text()), args.cpu)
    directories = [Path(original['environment']['GAME_DATA_DIR']) / BASE / source.name /
                   f'policy-it{i}' for i in range(10)]
    print('Checking saved replay receipts and NPZs; no new games or solves...', flush=True)
    sources = verify_sources(directories)
    for name, expected in original['input_sha256'].items():
        checked(source / 'inputs' / name, expected)
    output = Path(tempfile.mkdtemp(prefix='fit37-', dir=REPO / BASE / 'logs'))
    inputs = output / 'inputs'
    inputs.mkdir()
    shutil.copyfile(reference, inputs / 'iter24.pt')
    shutil.copyfile(source / 'config.json', inputs / 'source-config.json')
    shutil.copyfile(collection, inputs / 'collection-config.json')
    for name in ('positions-35-36.jsonl', 'labels.json'):
        shutil.copyfile(source / 'inputs' / name, inputs / name)
    # Include the previous successful solves without treating failed solves as labels.
    stages.merge_caches([inputs / 'labels.json',
        *sorted(source.glob('diagnostic-*/attempt-*/probe/labels.json'))], inputs / 'labels.json')
    write_json(inputs / 'source-hashes.json', sources)
    paths = [SCRIPT, SCRIPT.with_name('run_overnight.sh'), HELPER, stages.SCRIPT,
             stages.HELPER, *(REPO / 'src/alpha_go').rglob('*.py')]
    config = dict(schema=1, cwd=str(REPO), source=str(source), cpu=args.cpu, runtime=runtime,
        directories=[str(p) for p in directories], steps=8192, batch_size=256, lr=.0001,
        canary_steps=2048, canary_batch_size=64, canary_lr=.001,
        seed=20261105, final_seed=20261106, final_games=400,
        caps=dict(prepare=1800, canary=1800, train=5400, arena=10800, probe=7200),
        input_sha256={p.name: digest(p) for p in inputs.iterdir()},
        code_sha256={str(p.relative_to(REPO)): digest(p) for p in paths},
        scope='Fixed data and endpoints; both arms play the same 400 fresh paired games. '
              'Canary is diagnostic only. No automatic promotion or checkpoint selection.')
    write_json(output / 'config.json', config)
    (output / 'config.sha256').write_text(digest(output / 'config.json')+'\n')
    (output / 'code-commit.txt').write_text(subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True))
    (output / 'resume.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' +
        shlex.quote(str(REPO))+'\n'+shlex.join(['bash', str(SCRIPT.with_name('run_overnight.sh')),
                                              '--resume', str(output)])+'\n')
    return output, config


def verify(output: Path, config: dict[str, Any]) -> None:
    """Reject changed inputs/code/runtime; completed stages verify their own receipts."""
    checked(output / 'config.json', (output / 'config.sha256').read_text().strip())
    if config['cwd'] != str(REPO):
        raise ValueError('Resume in the original checkout')
    for name, expected in config['input_sha256'].items():
        checked(output / 'inputs' / name, expected)
    for name, expected in config['code_sha256'].items():
        checked(REPO / name, expected)
    source = json.loads((output / 'inputs/collection-config.json').read_text())
    if balanced.runtime_identity(source, config['cpu']) != config['runtime']:
        raise ValueError('Restore the original Python/native runtime before resuming')
    for name, expected in json.loads((output / 'inputs/source-hashes.json').read_text()).items():
        checked(Path(name), expected)


def train_command(output: Path, prepared: Path, attempt: Path, arm: str,
                  config: dict[str, Any]) -> list[str]:
    """Every independent arm starts from the same frozen iter24."""
    canary = arm == 'canary'
    command = [sys.executable, '-u', '-m', 'alpha_go.boxes.frontier_fit',
        '--reference', str(output / 'inputs/iter24.pt'), '--prepared', str(prepared),
        '--output', str(attempt / 'train'), '--arm', arm, '--steps',
        str(config['canary_steps'] if canary else config['steps']), '--batch-size',
        str(config['canary_batch_size'] if canary else config['batch_size']), '--lr',
        str(config['canary_lr'] if canary else config['lr']), '--seed', str(config['seed'])]
    return command + (['--cpu'] if config['cpu'] else [])


def arena_command(output: Path, candidate: Path, attempt: Path,
                  config: dict[str, Any]) -> list[str]:
    """Use identical search and the same new side-swapped openings for both arms."""
    command = [sys.executable, '-u', '-m', 'alpha_go.boxes.arena', '--rows', '5', '--cols', '5',
        '--a', 'ckpt:'+str(candidate), '--b', 'ckpt:'+str(output / 'inputs/iter24.pt'),
        '--num_games', str(config['final_games']), '--num_workers', '2', '--seed',
        str(config['final_seed']), '--opening_moves', '4', '--sims', '400', '--solver', '34',
        '--solver_budget', '50000', '--c_puct', '1.5', '--leaf_batch_size', '16',
        '--policy_temperature', '1', '--prove_terminals', '1', '--shape_terminals', '1',
        '--margin_utility_lambda', '.5', '--margin_utility_k', '6',
        '--save_games', str(attempt / 'games.jsonl')]
    return command + (['--cpu'] if config['cpu'] else [])


def probe_command(output: Path, trained: Path, attempt: Path, candidate: Path, band: str,
                  config: dict[str, Any]) -> list[str]:
    """Use exact cached held-out37 children, plus the existing35–36 regression roots."""
    positions = (trained / 'validation-positions.jsonl' if band == '37'
                 else output / 'inputs/positions-35-36.jsonl')
    cache = attempt / 'labels-input.json'
    stages.merge_caches([output / 'inputs/labels.json', trained / 'validation-labels.json',
                         *sorted(output.glob('probe-*/attempt-*/probe/labels.json'))], cache)
    command = [sys.executable, '-u', '-m', 'alpha_go.boxes.search_probe',
        '--reference', str(output / 'inputs/iter24.pt'), '--candidate', str(candidate),
        '--positions-file', str(positions), '--label-cache', str(cache),
        '--output', str(attempt / 'probe'), '--rows', '5', '--cols', '5',
        '--min-undrawn', '37' if band == '37' else '35',
        '--max-undrawn', '37' if band == '37' else '36', '--mixes', 'reference', 'candidate',
        '--simulations', '400', '--lambdas', '.5', '--solver', '34', '--solver-budget', '50000',
        '--label-budget', '20000000', '--c-puct', '1.5', '--leaf-batch-size', '16',
        '--policy-temperature', '1', '--prove-terminals', '1', '--shape-terminals', '1',
        '--margin-utility-k', '6', '--seed', str(config['seed'])]
    return command + (['--cpu'] if config['cpu'] else [])


def run(output: Path, config: dict[str, Any]) -> bool:
    """Preserve every completed stage; diagnostic failures never select or block an arm."""
    from alpha_go.boxes.curriculum import verify_match
    from alpha_go.boxes.probe_report import build_report, write_report

    summary: dict[str, Any] = dict(complete=False, stages={}, scope=config['scope'])

    def stage(name: str, cap: int, command: Any, finish: Any = None) -> Path | None:
        try:
            result: Path | None = stages.stage(output, name, cap, command, finish)
            assert result is not None
            summary['stages'][name] = dict(complete=True, attempt=str(result.relative_to(output)))
        except (TimeoutError, subprocess.CalledProcessError) as error:
            summary['stages'][name] = dict(complete=False, error=str(error))
            print(f'{name} incomplete: {error}; proceeding with independent work', flush=True)
            result = None
        write_json(output / 'summary.json', summary)
        return result

    caps = config['caps']
    prepared = stage('prepare', caps['prepare'], lambda a: [sys.executable, '-u', '-m',
        'alpha_go.boxes.frontier_fit_data', '--directories', *config['directories'],
        '--output', str(a / 'prepared')])
    if prepared is None:
        return False
    prepared = prepared / 'prepared'
    trained = {}
    for arm in ('canary', 'control', 'balanced'):
        attempt = stage('train-'+arm, caps['canary'] if arm == 'canary' else caps['train'],
                        lambda a, arm=arm: train_command(output, prepared, a, arm, config))
        if attempt is not None:
            folder = attempt / 'train'
            info = json.loads((folder / 'summary.json').read_text())
            if info['status'] != 'complete':
                raise ValueError(f'Training returned incomplete: {arm}')
            if arm != 'canary':
                trained[arm] = folder
    # Final arenas happen before optional costly band diagnostics, for BOTH fixed arms.
    for arm, folder in trained.items():
        candidate = folder / ('balanced.pt' if arm == 'balanced' else 'control.pt')

        def finish_arena(attempt: Path) -> None:
            text = (attempt / 'run.log').read_text().rsplit('===RESULT===', 1)[1]
            final = json.JSONDecoder().raw_decode(text.lstrip())[0]
            verify_match(attempt / 'games.jsonl', final, config['final_games'])
            write_json(attempt / 'final.json', final)
        stage('arena-'+arm, caps['arena'],
              lambda a, candidate=candidate: arena_command(output, candidate, a, config),
              finish_arena)
    for arm, folder in trained.items():
        candidate = folder / ('balanced.pt' if arm == 'balanced' else 'control.pt')
        for band in ('37', '35-36'):
            def finish_probe(attempt: Path) -> None:
                report = build_report(attempt / 'probe', seed=config['seed'], resamples=10000)
                write_report(report, attempt / 'report')
            stage('probe-'+arm+'-'+band, caps['probe'],
                lambda a, folder=folder, candidate=candidate, band=band:
                    probe_command(output, folder, a, candidate, band, config), finish_probe)
    summary['complete'] = len(summary['stages']) == 10 and all(
        s['complete'] for s in summary['stages'].values())
    write_json(output / 'summary.json', summary)
    return bool(summary['complete'])


def main() -> None:
    """Run the full fixed overnight protocol or reuse completed stages after interruption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=REPO / SOURCE)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    helpers.pin_runtime()
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / 'config.json').read_text())
    else:
        output, config = create(args)
    print(f"Run: {output}\nResume: bash {output / 'resume.sh'}\n"
          'Reuses saved labels; canary + two 8,192-update arms, then 400 games EACH vs iter24.\n'
          'Both fixed endpoints are tested; no small-probe checkpoint selection.\n'
          'Keep checkout and native build unchanged until this run finishes.', flush=True)
    def terminate(_signal: int, _frame: Any) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify(output, config)
            complete = run(output, config)
            verify(output, config)
        finally:
            stages.package(output)
            print(f"Send this file: {output / 'results.tgz'}", flush=True)
    if not complete:
        raise SystemExit('Some stages are incomplete; use the printed resume command.')


if __name__ == '__main__':
    main()
