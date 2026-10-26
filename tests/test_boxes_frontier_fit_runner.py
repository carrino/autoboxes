"""The overnight fit runner evaluates both fixed endpoints and preserves completed work."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def launcher() -> Any:
    path = Path(__file__).resolve().parents[1] / (
        'experiments/2026-10-21_21-13-boxes-frontier-fit/overnight.py')
    spec = importlib.util.spec_from_file_location('test_frontier_fit_runner', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> dict[str, Any]:
    return dict(scope='fixed endpoints', directories=['policy-it0'], cpu=True, steps=8192,
        batch_size=256, lr=.0001, canary_steps=2048, canary_batch_size=64, canary_lr=.001,
        seed=20261105, final_seed=20261106, final_games=400,
        caps=dict(prepare=1800, canary=1800, train=5400, arena=10800, probe=7200))


def test_both_fixed_endpoints_use_same_reference_search_and_openings(
    launcher: Any, tmp_path: Path,
) -> None:
    commands = [launcher.arena_command(tmp_path, tmp_path / f'{arm}.pt', tmp_path / arm,
                                      config()) for arm in ('control', 'balanced')]
    for command in commands:
        assert command[command.index('--b')+1] == 'ckpt:'+str(tmp_path/'inputs/iter24.pt')
        assert command[command.index('--num_games')+1] == '400'
        assert command[command.index('--seed')+1] == '20261106'
        assert command[command.index('--solver')+1] == '34'
        assert command[command.index('--solver_budget')+1] == '50000'
    for arm in ('control', 'balanced', 'canary'):
        command = launcher.train_command(tmp_path, tmp_path, tmp_path, arm, config())
        assert command[command.index('--reference')+1] == str(tmp_path/'inputs/iter24.pt')
        assert command[command.index('--steps')+1] == ('2048' if arm=='canary' else '8192')


@pytest.mark.parametrize('failed', [None, 'canary', 'control'])
def test_stage_recovery_and_independent_work(
    launcher: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: str | None,
) -> None:
    from alpha_go.boxes import curriculum, probe_report

    (tmp_path/'inputs').mkdir()
    launcher.write_json(tmp_path/'inputs/labels.json',
                        dict(schema=1, rows=5, cols=5, remaining_values={}))
    calls = []
    broken = [failed]

    def job(command: list[str], log: Path, seconds: int) -> None:
        calls.append(command)
        if '--arm' in command and command[command.index('--arm')+1] == broken[0]:
            raise subprocess.CalledProcessError(1, command)
        log.write_text('===RESULT===\n{"games": 400}\n')
        if '--output' in command:
            output = Path(command[command.index('--output')+1])
            output.mkdir()
            if '--arm' in command:
                launcher.write_json(output/'summary.json', dict(status='complete'))
                (output/'control.pt').write_bytes(b'control')
                (output/'balanced.pt').write_bytes(b'balanced')
                launcher.write_json(output/'validation-labels.json',
                                    dict(schema=1, rows=5, cols=5, remaining_values={}))
            if 'alpha_go.boxes.search_probe' in command:
                launcher.write_json(output/'labels.json',
                                    dict(schema=1, rows=5, cols=5, remaining_values={}))
        if '--save_games' in command:
            Path(command[command.index('--save_games')+1]).write_text('mocked games')

    monkeypatch.setattr(launcher.helpers, 'job', job)
    monkeypatch.setattr(curriculum, 'verify_match', lambda *a: None)
    monkeypatch.setattr(probe_report, 'build_report', lambda *a, **kw: {})
    monkeypatch.setattr(probe_report, 'write_report', lambda *a: None)
    assert launcher.run(tmp_path, config()) == (failed is None)
    arena = [c for c in calls if 'alpha_go.boxes.arena' in c]
    assert len(arena) == (1 if failed == 'control' else 2)
    assert all('canary' not in c[c.index('--a')+1] for c in arena)
    # Arenas precede probe work, regardless of canary status.
    assert max(i for i,c in enumerate(calls) if 'alpha_go.boxes.arena' in c) < min(
        i for i,c in enumerate(calls) if 'alpha_go.boxes.search_probe' in c)
    before = len(calls)
    broken[0] = None
    assert launcher.run(tmp_path, config())
    assert len(calls)-before == (0 if failed is None else 1 if failed=='canary' else 4)
    before = len(calls)
    assert launcher.run(tmp_path, config()) and len(calls)==before


def test_resume_rejects_changed_code_before_jobs(
    launcher: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, 'REPO', tmp_path)
    code = tmp_path/'trainer.py'
    code.write_text('original')
    c = dict(cwd=str(tmp_path), input_sha256={}, code_sha256={'trainer.py':launcher.digest(code)})
    launcher.write_json(tmp_path/'config.json', c)
    (tmp_path/'config.sha256').write_text(launcher.digest(tmp_path/'config.json'))
    code.write_text('changed')
    with pytest.raises(ValueError, match='Missing or changed'):
        launcher.verify(tmp_path, c)
