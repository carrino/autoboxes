#!/usr/bin/env bash
# Executed with: bash -ex experiments/2026-10-12_10-11-boxes-cpu-solver-work/logs/finalize.sh
# Run from /workspace/work/cpu-solver/autoboxes with supported network permission.
set -euo pipefail

REPORT_DIR=experiments/2026-10-12_10-11-boxes-cpu-solver-work
DELIVERABLE_ROOT=/codex/browser/projectless/0001-files-mentioned-by-the-user-uploaded-file-pointe/output

test "$(git branch --show-current)" = astra/cpu-solver-work
git diff --exit-code -- src/ tests/ experiments/2026-10-08_14-42-boxes-3x3-loop/
git diff --check
python3 - <<'PY'
import json
from pathlib import Path

report = Path('experiments/2026-10-12_10-11-boxes-cpu-solver-work')
machine = json.loads((report / 'data/machine.json').read_text())
diagnosis = json.loads((report / 'data/python-build-diagnosis.json').read_text())
assert machine['solver_processes'] == 4
assert diagnosis['candidate_libraries']['/install/lib/libpython3.12.so'] is False
assert diagnosis['candidate_libraries'][diagnosis['base_prefix'] + '/lib/libpython3.12.so']
assert 'Configuring incomplete' in (report / 'logs/setup.txt').read_text()
assert not list(report.rglob('*.npz'))
assert not list(report.rglob('*.tgz'))
assert not list(report.rglob('*.csv'))
print('Report evidence consistent; no game records or solver measurements in report.')
PY
git add -- "$REPORT_DIR"
git diff --cached --check
python3 - <<'PY'
import subprocess

paths = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).splitlines()
assert paths
assert all(p.startswith('experiments/2026-10-12_10-11-boxes-cpu-solver-work/') for p in paths)
print('Staged files:', len(paths), '(all inside this report directory)')
PY
git diff --cached --stat
git commit -m "Document CPU solver work blocked by relocated Python library path"
git push -u origin HEAD:refs/heads/astra/cpu-solver-work
git status --short --branch
git log -1 --format='%H %s'
mkdir -p "$DELIVERABLE_ROOT"
cp -a "$REPORT_DIR" "$DELIVERABLE_ROOT/"

# The PR is created afterward via the GitHub connector, base=main,
# head=astra/cpu-solver-work. No gh shell command is used.
