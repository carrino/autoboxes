"""The next overnight must preserve the verified winner and all prior replay."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def launcher() -> Any:
    path = Path(__file__).resolve().parents[1] / (
        'experiments/2026-10-22_20-38-boxes-ordinary-four-cycles/overnight.py')
    spec = importlib.util.spec_from_file_location('test_ordinary_four_cycles', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(launcher: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from alpha_go.boxes import frontier_fit_data

    source = tmp_path / 'parent'
    inputs = source / 'inputs'
    inputs.mkdir(parents=True)
    checkpoint = source / 'cycle-2-train/attempt-original/train/control.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'verified-cycle-2')
    monkeypatch.setattr(launcher, 'START_SHA', launcher.digest(checkpoint))
    (inputs / 'iter24.pt').write_bytes(b'iter24')
    monkeypatch.setattr(launcher.runner.fit, 'REFERENCE_SHA',
                        launcher.digest(inputs / 'iter24.pt'))
    (inputs / 'start.pt').write_bytes(b'previous-start')
    (inputs / 'collection-config.json').write_text('{}')
    (inputs / 'replay-hashes.json').write_text('{}')
    for name in ('labels.json', 'validation-labels.json'):
        launcher.write_json(inputs / name,
                            dict(schema=1, rows=5, cols=5, remaining_values={'42': 3}))
    for name in ('exclude.jsonl', 'positions-35-36.jsonl', 'positions-37.jsonl'):
        (inputs / name).write_text('{}\n')
    launcher.write_json(source / 'cycle-2-train/completed.json', dict(
        attempt='cycle-2-train/attempt-original', artifacts_sha256={
            str(checkpoint.relative_to(source)): launcher.digest(checkpoint)}))
    launcher.write_json(source / 'summary.json', dict(complete=True, stages={
        'cycle-2': dict(sha256=launcher.START_SHA)}))
    old_replay = tmp_path / 'old-replay'
    old_replay.write_text('saved-source')
    original = dict(cycles=2, directories=[str(old_replay)], label_settings={'max_undrawn': 37},
        input_sha256={p.name: launcher.digest(p) for p in inputs.iterdir()})
    launcher.write_json(source / 'config.json', original)
    monkeypatch.setattr(launcher, 'SOURCE_SHA', launcher.digest(source / 'config.json'))
    monkeypatch.setattr(launcher.runner.fit.balanced, 'runtime_identity', lambda *a: {})

    def verify_sources(directories: list[Path]) -> dict[str, str]:
        assert directories == [old_replay, source / 'cycle-1-labelled', source / 'cycle-2-labelled']
        return {str(old_replay): launcher.digest(old_replay)}

    monkeypatch.setattr(frontier_fit_data, 'verify_sources', verify_sources)

    def destination(**_kwargs: Any) -> str:
        output = tmp_path / 'created'
        output.mkdir()
        return str(output)

    monkeypatch.setattr(launcher.tempfile, 'mkdtemp', destination)
    return source


def test_create_retains_parent_replay_and_freezes_winner(
    launcher: Any, source: Path,
) -> None:
    before = {str(p): launcher.digest(p) for p in source.rglob('*') if p.is_file()}
    output, config = launcher.create(source, cpu=True)
    assert config['cycles'] == 4 and config['games'] == 200 and config['steps'] == 8192
    assert config['confirmation'] is False
    assert config['start_sha256'] == launcher.START_SHA
    assert launcher.digest(output / 'inputs/start.pt') == launcher.START_SHA
    assert config['directories'][-2:] == [str(source / 'cycle-1-labelled'),
                                         str(source / 'cycle-2-labelled')]
    assert (output / 'inputs/positions-37.jsonl').read_bytes() == (
        source / 'inputs/positions-37.jsonl').read_bytes()
    assert str(launcher.SCRIPT.with_name('run_overnight.sh')) in (
        output / 'resume.sh').read_text()
    assert config['final_seed'] not in (20261108, 20261109)
    launcher.verify(output, config)
    assert before == {str(p): launcher.digest(p) for p in source.rglob('*') if p.is_file()}
    (output / 'inputs/start.pt').write_bytes(b'wrong-weights')
    with pytest.raises(ValueError, match='Missing or changed'):
        launcher.verify(output, config)


@pytest.mark.parametrize('damage', ['weights', 'incomplete', 'summary'])
def test_create_refuses_unverified_source(
    launcher: Any, source: Path, damage: str,
) -> None:
    if damage == 'weights':
        (source / 'cycle-2-train/attempt-original/train/control.pt').write_bytes(b'wrong')
    else:
        path = source / 'summary.json'
        value = json.loads(path.read_text())
        if damage == 'incomplete':
            value['complete'] = False
        else:
            value['stages']['cycle-2']['sha256'] = 'wrong'
        launcher.write_json(path, value)
    with pytest.raises(ValueError):
        launcher.create(source, cpu=True)
    assert not (source.parent / 'created').exists()
