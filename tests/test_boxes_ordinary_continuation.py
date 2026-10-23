"""The ordinary continuation preserves lineage, completed work and fixed comparisons."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def launcher() -> Any:
    path = Path(__file__).resolve().parents[1] / (
        'experiments/2026-10-22_13-11-boxes-ordinary-continuation/overnight.py')
    spec = importlib.util.spec_from_file_location('test_ordinary_continuation', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> dict[str, Any]:
    return dict(scope='two fixed cycles', cpu=True, directories=['old-0', 'old-1'],
        cycles=2, games=200, steps=8192, batch_size=256, lr=.0001, snapshot_every=2048,
        seed=20261201, confirmation_seed=20261108, final_seed=20261109, final_games=400,
        label_settings=dict(rows=5, cols=5, min_undrawn=35, max_undrawn=37,
            node_budget=20000000, child_node_budget=20000000, table_entries=1 << 22,
            max_per_game=3, min_regret=2),
        caps=dict(arena=14400, collect=7200, label=7200, prepare=1800, train=5400, probe=7200))


def option(command: list[str], key: str) -> str:
    return command[command.index(key) + 1]


def test_runtime_settings_and_fresh_collection_seeds(launcher: Any, tmp_path: Path) -> None:
    c = config()
    source = tmp_path / 'cycle1.pt'
    commands = [launcher.collection_command(tmp_path, tmp_path, source, 20271201, c, worker)
                for worker in range(4)]
    assert {option(cmd, '--seed') for cmd in commands} == {
        str(20271201 + i * 1000) for i in range(4)}
    for cmd in commands:
        assert option(cmd, '--checkpoint') == str(source)
        assert option(cmd, '--num_games') == '50'
        assert option(cmd, '--num_simulations') == '400'
        assert option(cmd, '--solver_max_undrawn') == '34'
        assert '--stop-when-solved' in cmd and '--cpu' in cmd
    cmd = launcher.label_command(tmp_path, tmp_path, tmp_path, c)
    assert option(cmd, '--max-undrawn') == '37'
    assert option(cmd, '--max-per-game') == '3'
    cmd = launcher.train_command(source, tmp_path, tmp_path, 2, c)
    assert option(cmd, '--reference') == str(source)
    assert option(cmd, '--arm') == 'control' and option(cmd, '--steps') == '8192'
    assert option(cmd, '--snapshot-every') == '2048'


@pytest.mark.parametrize('cycles,confirmation,fail_confirmation', [
    (2, True, False), (2, True, True), (4, False, False)])
def test_fixed_endpoints_latest_weight_lineage_and_resume(
    launcher: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_confirmation: bool,
    cycles: int, confirmation: bool,
) -> None:
    from alpha_go.boxes import curriculum, probe_report

    c = config()
    c.update(cycles=cycles, confirmation=confirmation)
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    for name in ('start.pt', 'iter24.pt'):
        (inputs / name).write_bytes(name.encode())
    for name in ('labels.json', 'validation-labels.json'):
        launcher.write_json(inputs / name,
                            dict(schema=1, rows=5, cols=5, remaining_values={}))
    calls: list[list[str]] = []
    fail = [fail_confirmation]

    def job(command: list[str], log: Path, seconds: int) -> None:
        calls.append(command)
        if 'alpha_go.boxes.arena' in command:
            if option(command, '--seed') == str(c['confirmation_seed']) and fail[0]:
                raise TimeoutError('simulated interrupted confirmation')
            # Even a losing confirmation must continue the declared training recipe.
            log.write_text('===RESULT===\n{"games": 400, "a_wins": 170}\n')
            Path(option(command, '--save_games')).write_text('games')
        else:
            log.write_text('complete\n')
        if '--output' in command:
            output = Path(option(command, '--output'))
            output.mkdir(exist_ok=True)
            if 'alpha_go.boxes.frontier_fit' in command:
                checkpoint = output / 'control.pt'
                checkpoint.write_bytes(str(output).encode())
                launcher.write_json(output / 'summary.json', dict(status='complete', result=dict(
                    steps=8192, checkpoint_sha256=launcher.digest(checkpoint))))
            if 'alpha_go.boxes.search_probe' in command:
                launcher.write_json(output / 'labels.json',
                                    dict(schema=1, rows=5, cols=5, remaining_values={}))

    monkeypatch.setattr(launcher.helpers, 'job', job)
    monkeypatch.setattr(curriculum, 'verify_match', lambda *a: None)
    monkeypatch.setattr(probe_report, 'build_report', lambda *a, **kw: {})
    monkeypatch.setattr(probe_report, 'write_report', lambda *a: None)
    assert launcher.run(tmp_path, c) == (not fail_confirmation)
    trains = [cmd for cmd in calls if 'alpha_go.boxes.frontier_fit' in cmd]
    assert len(trains) == cycles
    endpoints = [Path(option(cmd, '--output')) / 'control.pt' for cmd in trains]
    parents = [inputs / 'start.pt', *endpoints[:-1]]
    last = endpoints[-1]
    assert [option(cmd, '--reference') for cmd in trains] == [str(p) for p in parents]
    collectors = [cmd for cmd in calls if '--collect-worker' in cmd]
    assert [cmd[-2] for cmd in collectors] == [str(p) for p in parents]
    preparation = [cmd for cmd in calls if 'alpha_go.boxes.frontier_fit_data' in cmd]
    assert all(cmd[cmd.index('--directories')+1:cmd.index('--directories')+3] == c['directories']
               for cmd in preparation)
    assert str(tmp_path / 'cycle-1-labelled') in preparation[1]
    assert str(tmp_path / 'cycle-2-labelled') in preparation[1]
    matches = [cmd for cmd in calls if 'alpha_go.boxes.arena' in cmd]
    assert len(matches) == 1 + int(confirmation)
    if confirmation:
        assert option(matches[0], '--a') == 'ckpt:' + str(inputs / 'start.pt')
        assert option(matches[0], '--b') == 'ckpt:' + str(inputs / 'iter24.pt')
        assert option(matches[0], '--seed') != option(matches[-1], '--seed')
    assert option(matches[-1], '--a') == 'ckpt:' + str(last)
    assert option(matches[-1], '--b') == 'ckpt:' + str(inputs / 'start.pt')
    assert all(option(cmd, '--num_games') == '400' for cmd in matches)
    probes = [cmd for cmd in calls if 'alpha_go.boxes.search_probe' in cmd]
    assert len(probes) == 2
    assert all(option(cmd, '--reference') == str(inputs / 'start.pt') for cmd in probes)
    assert max(i for i, cmd in enumerate(calls) if 'alpha_go.boxes.arena' in cmd) < min(
        i for i, cmd in enumerate(calls) if 'alpha_go.boxes.search_probe' in cmd)
    before = len(calls)
    fail[0] = False
    assert launcher.run(tmp_path, c)
    # Completed arenas, collection and training are never repeated. The labeller re-verifies
    # its complete per-game receipts; the interrupted confirmation alone needs new work.
    new = calls[before:]
    assert len(new) == cycles + int(fail_confirmation)
    assert all('alpha_go.boxes.policy_loop_data' in cmd or 'alpha_go.boxes.arena' in cmd
               for cmd in new)
    assert json.loads((tmp_path / 'summary.json').read_text())['complete']
    artifact = last
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Missing or changed'):
        launcher.run(tmp_path, c)


def test_changed_code_is_rejected_before_work(
    launcher: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, 'REPO', tmp_path)
    path = tmp_path / 'code.py'
    path.write_text('original')
    c = dict(cwd=str(tmp_path), input_sha256={}, code_sha256={'code.py': launcher.digest(path)})
    launcher.write_json(tmp_path / 'config.json', c)
    (tmp_path / 'config.sha256').write_text(launcher.digest(tmp_path / 'config.json'))
    path.write_text('changed')
    with pytest.raises(ValueError, match='Missing or changed'):
        launcher.verify(tmp_path, c)
