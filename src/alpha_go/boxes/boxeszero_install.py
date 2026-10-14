"""Fetch and verify a pinned external BoxesZero snapshot without vendoring its code.

The author links this repository to BoxesZero; the identity of the included weights
as the paper's final/competition model is not established. Installation downloads only
reviewed Python source and one checkpoint, never upstream bytecode or launch scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path

UPSTREAM_REPOSITORY = "neuqrui/-Distributed-Computer-Games-Framework"
UPSTREAM_COMMIT = "a62c5b213961f69cad9be1c2b716680cfeb98ebe"
CHECKPOINT_PATH = "ComputingCenter1/data/dotsboxes_6/model_best.pth.tar"
CHECKPOINT_BLOB = "4d3e02fd365bd03fedf8c2a4f7097a9f95dc468a"
SOURCE_BLOBS = {
    "ComputingCenter1/game_rules/abstract_game.py": "71b28358bb5f22af84c5d72802bef924d0620fe9",
    "ComputingCenter1/game_rules/dotsboxes.py": "75e9b8f8205e8502362a141c9a5638f563bd7c54",
    "ComputingCenter1/mcts/abstract_mcts.py": "5f96205add7aec92ea5cf4d1cf13e69c2392db64",
    "ComputingCenter1/mcts/dotsboxes.py": "b69c75750f44a1fa97682fab17f4bfb5d0bae24a",
    "ComputingCenter1/model/abstract_model.py": "637fe5db88f87de31c5b5dcbbe859c025f48f5bc",
    "ComputingCenter1/model/dotsboxes_6.py": "78e0431f9a01d63aded346d1b829ea0733d722b2",
    "ComputingCenter1/game_args/dotsboxes_6.py": "38456166192ab25c0bbb8dfb9d389e4c314d14d3",
    "ComputingCenter1/endgame.py": "a203889e42a9364a5c9f479af5c604f3a1335a4c",
    "ComputingCenter1/TrainNet.py": "a9dc56f63df121757fd8aaf5ae9f45f4d3ca7977",
    "ComputingCenter1/utils/__init__.py": "75ec3c80683e59458778e85fcb60a2640a251176",
    "ComputingCenter1/utils/helpers.py": "37e7fc1fa3b11e544d01508c6a37b099798ca0d7",
    "ComputingCenter1/utils/bar.py": "5ee968f0ce1c5bb1909aa9c69ddfe43f16e45f58",
    "ComputingCenter1/utils/AverageMeter.py": "0630e5b92a2d0a86952677fea866a28b084f12bb",
    "ControlCenter/game_args/dotsboxes_6.py": "924e7b833a8842fe5c25381db90f774a01569501",
    "ComputingCenter1/Storage.py": "fa3f01be07bbd80966f6bd1e663ba63f084c1e54",
}


def git_blob_sha(data: bytes) -> str:
    """Compute the Git blob identifier used by the pinned repository tree."""
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def verified_bytes(path: Path, expected: str) -> bytes:
    """Read a required file and reject missing or modified upstream artifacts."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing BoxesZero artifact: {path}; run python -m alpha_go.boxes.boxeszero_install"
        )
    raw = path.read_bytes()
    actual = git_blob_sha(raw)
    if actual != expected:
        raise ValueError(
            f"BoxesZero artifact hash mismatch: {path}; expected {expected}, got {actual}"
        )
    return raw


def verify_install(root: Path, checkpoint: Path) -> dict[str, object]:
    """Verify reviewed source plus the published checkpoint and return provenance."""
    # An upstream full clone contains extra package initializers and bytecode. Python
    # can import those even with -B, so require the minimal installer-managed tree.
    executable_suffixes = {".py", ".pyc", ".pyo", ".so", ".pyd", ".dll"}
    for center in ("ComputingCenter1", "ControlCenter"):
        for path in (root / center).rglob("*"):
            if path.is_file() and path.suffix in executable_suffixes:
                if path.relative_to(root).as_posix() not in SOURCE_BLOBS:
                    raise ValueError(
                        f"Unverified BoxesZero code: {path}; use a fresh installer destination"
                    )
    for relative, sha in SOURCE_BLOBS.items():
        verified_bytes(root / relative, sha)
    raw = verified_bytes(checkpoint, CHECKPOINT_BLOB)
    return {
        "repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_git_blob": CHECKPOINT_BLOB,
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_bytes": len(raw),
        "checkpoint_provenance": (
            "author-repository snapshot; paper/competition identity unverified"
        ),
        "source_files_verified": len(SOURCE_BLOBS),
    }


def install(root: Path) -> dict[str, object]:
    """Download absent artifacts, checking hashes before saving any executable source."""
    files = {**SOURCE_BLOBS, CHECKPOINT_PATH: CHECKPOINT_BLOB}
    for relative, sha in files.items():
        path = root / relative
        if path.exists():
            verified_bytes(path, sha)
            continue
        url = f"https://raw.githubusercontent.com/{UPSTREAM_REPOSITORY}/{UPSTREAM_COMMIT}/{relative}"
        print(f"Downloading {relative}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        if git_blob_sha(raw) != sha:
            raise ValueError(f"Downloaded BoxesZero artifact has the wrong hash: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves files created by another concurrent installer.
        with path.open("xb") as output:
            output.write(raw)
    return verify_install(root, root / CHECKPOINT_PATH)


def main() -> None:
    """Install the snapshot and write a ready-to-use arena opponent configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("work/boxeszero"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--response-timeout", type=float, default=120)
    args = parser.parse_args()
    if (
        args.simulations < 2
        or args.threads <= 0
        or not math.isfinite(args.response_timeout)
        or args.response_timeout <= 0
    ):
        parser.error("simulations must be at least two; threads and timeout must be positive")
    destination = args.destination.resolve()
    root = destination / "upstream"
    provenance = install(root)
    config = {
        "root": str(root),
        "checkpoint": str(root / CHECKPOINT_PATH),
        "simulations": args.simulations,
        "device": args.device,
        "threads": args.threads,
        "response_timeout": args.response_timeout,
    }
    config_path = destination / f"arena-{args.device}-s{args.simulations}.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError(f"Refusing to replace a different config: {config_path}")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"opponent": f"boxeszero:{config_path}", "provenance": provenance}, indent=2))


if __name__ == "__main__":
    main()
