"""Eight-hour resumable collection/label job; no training or promotion."""
import fcntl
import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import sys
import time
from pathlib import Path

root = Path.cwd().resolve()
run = Path(__file__).resolve().parent
exp = root / 'experiments/2026-10-08_14-42-boxes-3x3-loop'
source = exp / 'logs/all-policy-recovered-fgo6edja'
script = root / 'experiments/2026-10-18_20-07-boxes-all-policy-pilot/pilot.py'
spec = importlib.util.spec_from_file_location('pilot', script)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

if not (run / 'config.json').exists():
    import alpha_go_cpp
    import torch
    assert torch.cuda.is_available(), 'CUDA is required'
    checkpoint = source / 'inputs/iter24.pt'
    assert pilot.digest(checkpoint) == '73e9aa25856ebc1f01529013b215fcd7a5935909da2920a1b7b454aecaafbe8d'
    exclusions = (source / 'inputs/exclude.jsonl').read_text()
    assert all(json.loads(s)['rows'] == json.loads(s)['cols'] == 5
               for s in exclusions.splitlines() if s.strip())
    (run / 'inputs').mkdir()
    shutil.copyfile(checkpoint, run / 'inputs/iter24.pt')
    (run / 'inputs/exclude.jsonl').write_text(exclusions)
    files = [script, pilot.HELPER, exp / 'run_games.py', Path(__file__).resolve(),
             Path(alpha_go_cpp.__file__), *(root / 'src/alpha_go').rglob('*.py'),
             *(run / 'inputs').iterdir()]
    config = dict(cwd=str(root), games=200, chunks=10,
                  seed=secrets.randbelow(1_000_000_000),
                  hashes={str(p): pilot.digest(p) for p in files},
                  future_validation=dict(family_fraction=0.1, split_seed=0,
                                         require_D4_transposition_exclusion=True))
    pilot.write_json(run / 'config.json', config)
config = json.loads((run / 'config.json').read_text())
assert config['cwd'] == str(root), 'Resume from the original repository'
for name, digest in config['hashes'].items():
    pilot.checked(Path(name), digest)

if len(sys.argv) > 1:
    assert sys.argv[1] == '--collect' and len(sys.argv) == 4
    pilot.SEED = config['seed'] + int(sys.argv[3]) * 10000
    pilot.collect(run, Path(sys.argv[2]))
    raise SystemExit

lock = (run / '.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
deadline = time.monotonic() + 8 * 3600

def remaining():
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError('Eight-hour budget reached; use resume.sh to continue')
    return seconds

try:
    folders = []
    for i in range(config['chunks']):
        attempt = pilot.stage(run, f'collect-{i:02d}', min(5400, remaining()),
            lambda a, i=i: [sys.executable, '-u', str(Path(__file__).resolve()),
                            '--collect', str(a), str(i)])
        folders.append(str(attempt / 'games'))
    pilot.helpers.job([sys.executable, '-u', '-m', 'alpha_go.boxes.all_policy_data',
        '--games', *folders, '--output', str(run / 'prepared'),
        '--exclude-positions', str(run / 'inputs/exclude.jsonl'),
        '--rows', '5', '--min-undrawn', '35', '--max-undrawn', '36',
        '--workers', '4', '--node-budget', '20000000',
        '--child-node-budget', '20000000'], run / 'prepare.log', remaining())
    print('Collection and exact labelling complete. Weights unchanged.', flush=True)
finally:
    pilot.package(run)
    print(f'Send this file: {run}/results.tgz', flush=True)
