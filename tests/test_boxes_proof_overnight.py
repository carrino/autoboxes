"""Preflight rejects a stale build, imported checkout or mistaken starting model."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def launcher():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "experiments/2026-10-17_14-14-boxes-proof-overnight/overnight.py"
    spec = importlib.util.spec_from_file_location("proof_overnight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_rejects_changed_import_even_when_recorded_binary_still_exists(
        launcher, tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "recorded.so"
    binary.write_bytes(b"original")
    config = dict(preflight=dict(build_inputs_sha256={str(binary): launcher.file_sha(binary)},
                                 runtime={"extension": str(binary)}))
    monkeypatch.setattr(launcher, "runtime_identity", lambda: {"extension": "stale.so"})
    with pytest.raises(ValueError, match="Runtime changed"):
        launcher.verify_preflight(config)


def test_resume_rejects_native_source_change(launcher, tmp_path: Path) -> None:
    source = tmp_path / "solver.cpp"
    source.write_text("old source")
    config = dict(preflight=dict(build_inputs_sha256={str(source): launcher.file_sha(source)}))
    source.write_text("changed source")
    with pytest.raises(ValueError, match="changed"):
        launcher.verify_preflight(config)


def test_resume_rejects_another_reference(launcher, tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "reference.pt"
    checkpoint.write_bytes(b"candidate instead of iter0")
    config = dict(reference=str(checkpoint),
                  preflight=dict(build_inputs_sha256={}, runtime={}))
    monkeypatch.setattr(launcher, "runtime_identity", lambda: {})
    with pytest.raises(ValueError, match="verified frozen iter0"):
        launcher.verify_preflight(config)


def test_explicit_reference_cannot_bypass_identity_before_build(
        launcher, tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"unapproved starting weights")
    monkeypatch.setattr(launcher, "pin_runtime", lambda: None)
    monkeypatch.setattr(sys, "argv", ["overnight.py", "--cpu", "--reference", str(checkpoint)])
    with pytest.raises(ValueError, match="Reference differs"):
        launcher.main()


def test_refuse_rebuild_after_extension_import(launcher, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "alpha_go_cpp", object())
    with pytest.raises(RuntimeError, match="already loaded"):
        launcher.build_and_check(tmp_path)
    assert not (tmp_path / "build.log").exists()


def test_reject_python_module_from_another_checkout(launcher, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(launcher.importlib, "import_module",
                        lambda _name: SimpleNamespace(__file__=str(tmp_path / "__init__.py")))
    with pytest.raises(ValueError, match="Wrong checkout imported"):
        launcher.runtime_identity()
