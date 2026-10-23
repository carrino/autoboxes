"""Confirm the ordinary endpoint, then run two fixed fresh-data learning cycles."""
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
SOURCE = BASE / 'logs/fit37-wau7ku0x'
SOURCE_SHA = '19109763407d8edce1169acf0b03e899c542ca85263be78332718f4829527da6'
START_SHA = 'e913bc0af1b6b0df4ecd44b55f2f41af5d5c7e823af7e0a12f912773408fc153'
FIT = REPO / 'experiments/2026-10-21_21-13-boxes-frontier-fit/overnight.py'
spec = importlib.util.spec_from_file_location('ordinary_fit_helpers', FIT)
assert spec and spec.loader
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)
stages, helpers = fit.stages, fit.helpers
checked, digest, write_json = helpers.checked, helpers.digest, helpers.write_json
BASE_COLLECTION_COMMAND = stages.collection_command


def create(source: Path, cpu: bool) -> tuple[Path, dict[str, Any]]:
    """Freeze the verified ordinary checkpoint, incumbent, replay and diagnostic inputs."""
    from alpha_go.boxes.frontier_fit_data import verify_sources

    source = source.resolve()
    checked(source / 'config.json', SOURCE_SHA)
    original = json.loads((source / 'config.json').read_text())
    if not json.loads((source / 'summary.json').read_text())['complete']:
        raise ValueError('The source fitting run must be complete')
    for name, expected in original['input_sha256'].items():
        checked(source / 'inputs' / name, expected)

    def never_train(_attempt: Path) -> list[str]:
        raise ValueError('Missing source training receipt; do not repeat the fitting run')

    trained = stages.stage(source, 'train-control', 0, never_train) / 'train'
    checked(trained / 'control.pt', START_SHA)
    checked(source / 'inputs/iter24.pt', fit.REFERENCE_SHA)
    runtime = fit.balanced.runtime_identity(
        json.loads((source / 'inputs/collection-config.json').read_text()), cpu)
    directories = [Path(p) for p in original['directories']]
    print('Verifying the ten saved replay preparations; no relabelling...', flush=True)
    sources = verify_sources(directories)
    frontier = json.loads((source / 'inputs/source-config.json').read_text())
    excluded = Path(original['source']) / 'inputs/exclude.jsonl'
    checked(excluded, frontier['input_sha256']['exclude.jsonl'])
    settings = json.loads((directories[0] / 'preparation.json').read_text())['identity']['settings']
    if (settings['rows'], settings['cols'], settings['min_undrawn'],
            settings['max_undrawn'], settings['max_per_game']) != (5, 5, 35, 37, 3):
        raise ValueError('Expected the saved 35..37 mistake-policy recipe')
    output = Path(tempfile.mkdtemp(prefix='ordinary-next-', dir=REPO / BASE / 'logs'))
    inputs = output / 'inputs'
    inputs.mkdir()
    shutil.copyfile(trained / 'control.pt', inputs / 'start.pt')
    for name in ('iter24.pt', 'collection-config.json', 'positions-35-36.jsonl'):
        shutil.copyfile(source / 'inputs' / name, inputs / name)
    shutil.copyfile(trained / 'validation-positions.jsonl', inputs / 'positions-37.jsonl')
    shutil.copyfile(trained / 'validation-labels.json', inputs / 'validation-labels.json')
    # Also exclude the fitting run's held-out decision positions from fresh source games.
    (inputs / 'exclude.jsonl').write_text(excluded.read_text() + '\n' +
                                         (inputs / 'positions-37.jsonl').read_text())
    stages.merge_caches([source / 'inputs/labels.json', trained / 'validation-labels.json',
                        *sorted(source.glob('probe-*/attempt-*/probe/labels.json'))],
                       inputs / 'labels.json')
    write_json(inputs / 'replay-hashes.json', sources)
    paths = [SCRIPT, SCRIPT.with_name('run_overnight.sh'), FIT, fit.HELPER,
             stages.SCRIPT, stages.HELPER, REPO / BASE / 'run_games.py',
             *(REPO / 'src/alpha_go').rglob('*.py')]
    config = dict(schema=1, cwd=str(REPO), cpu=cpu, runtime=runtime, source=str(source),
        directories=[str(p) for p in directories], label_settings=settings,
        cycles=2, games=200, steps=8192, batch_size=256, lr=.0001, snapshot_every=2048,
        seed=20261201, confirmation_seed=20261108, final_seed=20261109, final_games=400,
        caps=dict(arena=14400, collect=7200, label=7200, prepare=1800, train=5400, probe=7200),
        input_sha256={p.name: digest(p) for p in inputs.iterdir()},
        code_sha256={str(p.relative_to(REPO)): digest(p) for p in paths},
        scope='Fixed two-cycle ordinary-replay continuation from the fitting endpoint. '
              'Confirmation does not select, promote or gate training. Final LAST vs frozen '
              'starting learner is primary for continuation; no intermediate selection.')
    write_json(output / 'config.json', config)
    (output / 'config.sha256').write_text(digest(output / 'config.json') + '\n')
    (output / 'code-commit.txt').write_text(subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True))
    (output / 'resume.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' +
        shlex.quote(str(REPO)) + '\n' + shlex.join(['bash', str(SCRIPT.with_name(
            'run_overnight.sh')), '--resume', str(output)]) + '\n')
    return output, config


def verify(output: Path, config: dict[str, Any]) -> None:
    """Reject changed code, runtime or immutable inputs before starting or resuming."""
    checked(output / 'config.json', (output / 'config.sha256').read_text().strip())
    if config['cwd'] != str(REPO):
        raise ValueError('Resume in the original checkout')
    for name, expected in config['input_sha256'].items():
        checked(output / 'inputs' / name, expected)
    for name, expected in config['code_sha256'].items():
        checked(REPO / name, expected)
    checked(output / 'inputs/start.pt', config.get('start_sha256', START_SHA))
    checked(output / 'inputs/iter24.pt', fit.REFERENCE_SHA)
    source = json.loads((output / 'inputs/collection-config.json').read_text())
    if fit.balanced.runtime_identity(source, config['cpu']) != config['runtime']:
        raise ValueError('Restore the original Python/native runtime')
    for name, expected in json.loads((output / 'inputs/replay-hashes.json').read_text()).items():
        checked(Path(name), expected)


def collection_command(output: Path, attempt: Path, checkpoint: Path, seed: int,
                       config: dict[str, Any], worker: int) -> list[str]:
    """Use the existing full-start, proof-filtered solver34 collector with fresh seeds."""
    command: list[str] = BASE_COLLECTION_COMMAND(output, attempt, config['games'], worker)
    command[command.index('--checkpoint') + 1] = str(checkpoint)
    command[command.index('--seed') + 1] = str(seed + worker * 1000)
    return command + (['--cpu'] if config['cpu'] else [])


def collect(output: Path, attempt: Path, checkpoint: Path, seed: int) -> None:
    """Reuse process cleanup and proof-target validation in an isolated collector child."""
    config = json.loads((output / 'config.json').read_text())
    original = stages.collection_command

    def command(o: Path, a: Path, games: int, worker: int) -> list[str]:
        assert games == config['games']
        return collection_command(o, a, checkpoint, seed, config, worker)

    stages.collection_command = command
    try:
        stages.collect(output, attempt)
    finally:
        stages.collection_command = original


def label_command(output: Path, games: Path, prepared: Path,
                  config: dict[str, Any]) -> list[str]:
    """Retain the exact saved mistake-selection and solver settings."""
    command = [sys.executable, '-u', '-m', 'alpha_go.boxes.policy_loop_data',
        '--games', str(games), '--output', str(prepared), '--workers', '4',
        '--exclude-positions', str(output / 'inputs/exclude.jsonl')]
    for name, value in config['label_settings'].items():
        command.extend(['--' + name.replace('_', '-'), str(value)])
    return command


def train_command(checkpoint: Path, prepared: Path, attempt: Path, cycle: int,
                  config: dict[str, Any]) -> list[str]:
    """Continue the latest weights using the successful fixed ordinary-training recipe."""
    command = [sys.executable, '-u', '-m', 'alpha_go.boxes.frontier_fit',
        '--reference', str(checkpoint), '--prepared', str(prepared),
        '--output', str(attempt / 'train'), '--arm', 'control',
        '--steps', str(config['steps']), '--batch-size', str(config['batch_size']),
        '--lr', str(config['lr']), '--seed', str(config['seed'] + cycle),
        '--snapshot-every', str(config['snapshot_every'])]
    return command + (['--cpu'] if config['cpu'] else [])


def arena_command(output: Path, candidate: Path, opponent: Path, attempt: Path,
                  seed: int, config: dict[str, Any]) -> list[str]:
    """Use the same fixed search for both players, with distinct new paired openings."""
    command: list[str] = fit.arena_command(output, candidate, attempt,
                                         dict(config, final_seed=seed))
    command[command.index('--b') + 1] = 'ckpt:' + str(opponent)
    return command


def probe_command(output: Path, candidate: Path, attempt: Path, band: str,
                  config: dict[str, Any]) -> list[str]:
    """Keep fixed development roots and compare against the frozen starting learner."""
    command: list[str] = fit.probe_command(output, output / 'inputs', attempt, candidate,
                                          band, config)
    command[command.index('--reference') + 1] = str(output / 'inputs/start.pt')
    command[command.index('--positions-file') + 1] = str(
        output / 'inputs' / f'positions-{band}.jsonl')
    return command


def run(output: Path, config: dict[str, Any]) -> bool:
    """Run fixed cycles; optional confirmation never gates or selects training endpoints."""
    from alpha_go.boxes.curriculum import verify_match
    from alpha_go.boxes.probe_report import build_report, write_report

    summary: dict[str, Any] = dict(complete=False, scope=config['scope'], stages={})
    write_json(output / 'summary.json', summary)

    def optional(name: str, seconds: int, command: Any, finish: Any) -> None:
        try:
            attempt = stages.stage(output, name, seconds, command, finish)
            summary['stages'][name] = dict(complete=True, attempt=str(attempt.relative_to(output)))
        except (TimeoutError, subprocess.CalledProcessError) as error:
            summary['stages'][name] = dict(complete=False, error=str(error))
            print(f'{name} incomplete: {error}; continuing independent work', flush=True)
        write_json(output / 'summary.json', summary)

    def finish_arena(attempt: Path) -> None:
        text = (attempt / 'run.log').read_text().rsplit('===RESULT===', 1)[1]
        result = json.JSONDecoder().raw_decode(text.lstrip())[0]
        verify_match(attempt / 'games.jsonl', result, config['final_games'])
        write_json(attempt / 'final.json', result)
        print('Completed match: ' + json.dumps(result), flush=True)

    caps = config['caps']
    checkpoint = output / 'inputs/start.pt'
    if config.get('confirmation', True):
        optional('confirmation', caps['arena'], lambda a: arena_command(output, checkpoint,
            output / 'inputs/iter24.pt', a, config['confirmation_seed'], config), finish_arena)
    directories = list(config['directories'])
    for cycle in range(1, config['cycles'] + 1):
        prefix = f'cycle-{cycle}'
        collected = stages.stage(output, prefix + '-collect', caps['collect'], lambda a: [
            sys.executable, '-u', str(SCRIPT), '--collect-worker', str(output), str(a),
            str(checkpoint), str(config['seed'] + cycle * 10000)])
        # Stable output lets the labeller reuse each completed game after interruption.
        labelled = output / (prefix + '-labelled')
        helpers.job(label_command(output, collected / 'games', labelled, config),
                    output / (prefix + '-labels.log'), caps['label'])
        directories.append(str(labelled))
        prepared = stages.stage(output, prefix + '-prepare', caps['prepare'], lambda a: [
            sys.executable, '-u', '-m', 'alpha_go.boxes.frontier_fit_data',
            '--directories', *directories, '--output', str(a / 'prepared')])
        trained = stages.stage(output, prefix + '-train', caps['train'], lambda a: train_command(
            checkpoint, prepared / 'prepared', a, cycle, config)) / 'train'
        info = json.loads((trained / 'summary.json').read_text())
        if info['status'] != 'complete' or info['result']['steps'] != config['steps']:
            raise ValueError('Incomplete fixed training endpoint')
        checkpoint = trained / 'control.pt'
        checked(checkpoint, info['result']['checkpoint_sha256'])
        summary['stages'][prefix] = dict(complete=True, checkpoint=str(checkpoint),
                                        sha256=digest(checkpoint))
        write_json(output / 'summary.json', summary)
    optional('final-vs-start', caps['arena'], lambda a: arena_command(output, checkpoint,
        output / 'inputs/start.pt', a, config['final_seed'], config), finish_arena)
    for band in ('35-36', '37'):
        def finish_probe(attempt: Path) -> None:
            report = build_report(attempt / 'probe', seed=config['seed'], resamples=10000)
            write_report(report, attempt / 'report')
        optional('probe-' + band, caps['probe'], lambda a, band=band:
            probe_command(output, checkpoint, a, band, config), finish_probe)
    summary['complete'] = all(s['complete'] for s in summary['stages'].values())
    write_json(output / 'summary.json', summary)
    return bool(summary['complete'])


def main() -> None:
    """Start a bounded overnight recipe or resume its completed stages without retraining."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=REPO / SOURCE)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--cpu', action='store_true', help='CPU smoke tests only')
    parser.add_argument('--collect-worker', nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    helpers.pin_runtime()
    if args.collect_worker:
        output, attempt, checkpoint, seed = args.collect_worker
        collect(Path(output), Path(attempt), Path(checkpoint), int(seed))
        return
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / 'config.json').read_text())
    else:
        output, config = create(args.source_run, args.cpu)
    print(f'Run: {output}\nResume: bash {output / "resume.sh"}\n'
          '400 confirmation games; TWO cycles of 200 fresh games + 8192 ordinary updates;\n'
          'then 400 LAST-vs-start games and bounded 35–36/37 diagnostics.\n'
          'Expected 6–9 hours; stage caps are failure limits, not a total deadline.\n'
          'Training continues regardless of confirmation score. No automatic promotion.\n'
          'Keep this checkout and native build unchanged until completion.', flush=True)

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
            print(f'Send this file: {output / "results.tgz"}', flush=True)
    if not complete:
        raise SystemExit('Some evaluations are incomplete; use the printed resume command.')


if __name__ == '__main__':
    main()
