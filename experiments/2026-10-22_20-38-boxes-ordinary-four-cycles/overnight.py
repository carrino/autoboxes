"""Continue the verified 61% winner for four fixed ordinary-replay cycles overnight."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import shlex
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
BASE = Path('experiments/2026-10-08_14-42-boxes-3x3-loop')
SOURCE = BASE / 'logs/ordinary-next-vm_dguq5'
SOURCE_SHA = 'c7b69dcbf3476720a5220dab3a1cc25dff981fa0a16b226d4069c57bd273f152'
START_SHA = '674b27ff345c705a6ee144a305f7fcb69d812fbc7393da3133523d7d9da57108'
RUNNER = REPO / 'experiments/2026-10-22_13-11-boxes-ordinary-continuation/overnight.py'
spec = importlib.util.spec_from_file_location('ordinary_four_cycle_runner', RUNNER)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
checked, digest, write_json = runner.checked, runner.digest, runner.write_json


def create(source: Path, cpu: bool) -> tuple[Path, dict[str, Any]]:
    """Freeze the completed cycle-2 endpoint and all 1,400 previous source games."""
    from alpha_go.boxes.frontier_fit_data import verify_sources

    source = source.resolve()
    checked(source / 'config.json', SOURCE_SHA)
    original = json.loads((source / 'config.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    if not summary['complete'] or original['cycles'] != 2:
        raise ValueError('Expected the completed two-cycle ordinary run')
    for name, expected in original['input_sha256'].items():
        checked(source / 'inputs' / name, expected)

    def never_train(_attempt: Path) -> list[str]:
        raise ValueError('Missing cycle-2 training receipt; do not retrain the source run')

    trained = runner.stages.stage(source, 'cycle-2-train', 0, never_train) / 'train'
    checked(trained / 'control.pt', START_SHA)
    if summary['stages']['cycle-2']['sha256'] != START_SHA:
        raise ValueError('Source summary disagrees with verified cycle-2 weights')
    runtime = runner.fit.balanced.runtime_identity(
        json.loads((source / 'inputs/collection-config.json').read_text()), cpu)
    directories = [Path(p) for p in original['directories']]
    directories.extend(source / f'cycle-{cycle}-labelled' for cycle in (1, 2))
    print('Verifying saved replay and both completed label passes; no relabelling...', flush=True)
    sources = verify_sources(directories)
    output = Path(tempfile.mkdtemp(prefix='ordinary-four-', dir=REPO / BASE / 'logs'))
    inputs = output / 'inputs'
    inputs.mkdir()
    for name in original['input_sha256']:
        if name not in ('start.pt', 'replay-hashes.json', 'labels.json'):
            shutil.copyfile(source / 'inputs' / name, inputs / name)
    shutil.copyfile(trained / 'control.pt', inputs / 'start.pt')
    runner.stages.merge_caches([source / 'inputs/labels.json',
        *sorted(source.glob('probe-*/attempt-*/probe/labels.json'))], inputs / 'labels.json')
    write_json(inputs / 'replay-hashes.json', sources)
    shutil.copyfile(source / 'config.json', inputs / 'parent-config.json')
    paths = [SCRIPT, SCRIPT.with_name('run_overnight.sh'), RUNNER, runner.FIT,
             runner.fit.HELPER, runner.stages.SCRIPT, runner.stages.HELPER,
             REPO / BASE / 'run_games.py', *(REPO / 'src/alpha_go').rglob('*.py')]
    config = dict(schema=1, cwd=str(REPO), cpu=cpu, runtime=runtime, source=str(source),
        source_config_sha256=SOURCE_SHA, start_sha256=START_SHA,
        directories=[str(p) for p in directories], label_settings=original['label_settings'],
        cycles=4, games=200, steps=8192, batch_size=256, lr=.0001, snapshot_every=2048,
        seed=20261023, confirmation=False, final_seed=20261110, final_games=400,
        caps=dict(arena=14400, collect=7200, label=7200, prepare=1800, train=5400, probe=7200),
        input_sha256={p.name: digest(p) for p in inputs.iterdir()},
        code_sha256={str(p.relative_to(REPO)): digest(p) for p in paths},
        scope='Four fixed ordinary-replay cycles from verified ordinary cycle 2. '
              'No repeated confirmation. LAST vs frozen starting learner on 400 new paired '
              'games is primary. Fixed development regression probes; no automatic promotion.')
    write_json(output / 'config.json', config)
    (output / 'config.sha256').write_text(digest(output / 'config.json') + '\n')
    (output / 'code-commit.txt').write_text(subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True))
    (output / 'resume.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' +
        shlex.quote(str(REPO)) + '\n' + shlex.join(['bash', str(SCRIPT.with_name(
            'run_overnight.sh')), '--resume', str(output)]) + '\n')
    return output, config


def verify(output: Path, config: dict[str, Any]) -> None:
    """Require the selected winner and fixed protocol, then check all frozen artifacts."""
    if (config['start_sha256'] != START_SHA or config['source_config_sha256'] != SOURCE_SHA
            or config['cycles'] != 4 or config['confirmation'] is not False):
        raise ValueError('This experiment requires the verified cycle-2 winner and four cycles')
    checked(output / 'inputs/parent-config.json', SOURCE_SHA)
    runner.verify(output, config)


def main() -> None:
    """Start four fixed cycles or reuse completed stages of the same experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=REPO / SOURCE)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--cpu', action='store_true', help='explicit CPU smoke tests only')
    args = parser.parse_args()
    runner.helpers.pin_runtime()
    if args.resume:
        output = args.resume.resolve()
        config = json.loads((output / 'config.json').read_text())
    else:
        output, config = create(args.source_run, args.cpu)
    print(f'Run: {output}\nResume: bash {output / "resume.sh"}\n'
          'Starts from verified ordinary cycle 2 (244–156 winner).\n'
          'FOUR cycles: 200 fresh full-start games + 8192 ordinary updates each.\n'
          'Then 400 LAST-vs-start games and fixed 35–36/37 regression probes.\n'
          'Expected 6–9 hours; fixed workload, with per-stage failure caps.\n'
          'Keep checkout and native build unchanged until completion.', flush=True)

    def terminate(_signal: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify(output, config)
            complete = runner.run(output, config)
            verify(output, config)
        finally:
            runner.stages.package(output)
            print(f'Send this file: {output / "results.tgz"}', flush=True)
    if not complete:
        raise SystemExit('Some evaluations are incomplete; use the printed resume command.')


if __name__ == '__main__':
    main()
