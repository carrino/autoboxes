"""Pinned artifact installation rejects missing or changed engine files."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

from alpha_go.boxes import boxeszero_install as installer


@pytest.fixture
def artifacts(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    files = {"ComputingCenter1/model.py": b"# synthetic source\n", "weights.pt": b"fake weights"}
    monkeypatch.setattr(installer, "SOURCE_BLOBS", {
        "ComputingCenter1/model.py": installer.git_blob_sha(files["ComputingCenter1/model.py"]),
    })
    monkeypatch.setattr(installer, "CHECKPOINT_PATH", "weights.pt")
    monkeypatch.setattr(installer, "CHECKPOINT_BLOB", installer.git_blob_sha(files["weights.pt"]))
    return files


class TestPinnedInstall:
    def test_download_verify_and_reuse(
        self, tmp_path: Path, artifacts: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests: list[str] = []
        prefix = (
            f"https://raw.githubusercontent.com/{installer.UPSTREAM_REPOSITORY}/"
            f"{installer.UPSTREAM_COMMIT}/"
        )

        def download(url: str, *, timeout: int) -> io.BytesIO:
            assert url.startswith(prefix) and timeout == 60
            requests.append(url)
            return io.BytesIO(artifacts[url[len(prefix):]])

        monkeypatch.setattr(installer.urllib.request, "urlopen", download)
        first = installer.install(tmp_path)
        assert first["checkpoint_sha256"] == hashlib.sha256(artifacts["weights.pt"]).hexdigest()
        assert first["source_files_verified"] == 1
        assert installer.install(tmp_path) == first
        assert len(requests) == 2  # A second installation makes no requests.

    def test_wrong_download_never_saved(
        self, tmp_path: Path, artifacts: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            installer.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"unexpected")
        )
        with pytest.raises(ValueError, match="wrong hash"):
            installer.install(tmp_path)
        assert not list(tmp_path.rglob("*.py"))

    @pytest.mark.parametrize("changed", ["ComputingCenter1/model.py", "weights.pt"])
    def test_modified_existing_artifact_is_rejected(
        self, tmp_path: Path, artifacts: dict[str, bytes], changed: str
    ) -> None:
        for relative, raw in artifacts.items():
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (tmp_path / changed).write_bytes(b"modified")
        with pytest.raises(ValueError, match="hash mismatch"):
            installer.verify_install(tmp_path, tmp_path / "weights.pt")

    def test_missing_artifact_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="boxeszero_install"):
            installer.verified_bytes(tmp_path / "absent", "0" * 40)

    @pytest.mark.parametrize("extra", ["__init__.py", "model.pyc", "__pycache__/model.pyc"])
    def test_unverified_importable_code_is_rejected(
        self, tmp_path: Path, artifacts: dict[str, bytes], extra: str
    ) -> None:
        for relative, raw in artifacts.items():
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        unexpected = tmp_path / "ComputingCenter1" / extra
        unexpected.parent.mkdir(parents=True, exist_ok=True)
        unexpected.write_bytes(b"unverified")
        with pytest.raises(ValueError, match="Unverified BoxesZero code"):
            installer.verify_install(tmp_path, tmp_path / "weights.pt")

    def test_cli_emits_reusable_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(installer, "install", lambda root: {"test_only": True})
        monkeypatch.setattr(sys, "argv", ["install", "--destination", str(tmp_path)])
        installer.main()
        config_path = tmp_path / "arena-cpu-s800.json"
        config = json.loads(config_path.read_text())
        assert config["root"] == str(tmp_path / "upstream")
        assert config["checkpoint"] == str(tmp_path / "upstream" / installer.CHECKPOINT_PATH)
        installer.main()
        config["simulations"] = 100
        config_path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="different config"):
            installer.main()

    @pytest.mark.parametrize("arguments", [
        ["--simulations", "1"], ["--threads", "0"], ["--response-timeout", "nan"],
    ])
    def test_invalid_limits_fail_before_download(
        self, arguments: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["install", *arguments])
        with pytest.raises(SystemExit, match="2"):
            installer.main()
