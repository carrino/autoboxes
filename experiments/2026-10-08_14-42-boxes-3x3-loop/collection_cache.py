"""Publish complete collections atomically and verify them before a loop resumes.

`begin` prints `reuse` or `collect`; collection writes into `<games>.collecting`.
`finish` verifies the original request and game count, then publishes `<games>`.
Exact value relabelling may add/change only `exact_margin_p1` without invalidating
the collection receipt. All other NPZ arrays remain covered by semantic hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REQUEST = "collection-request.json"
RECEIPT = "collection-cache.json"


def checkpoint_hash(path: Path) -> str:
    """Hash checkpoint bytes without loading torch or the model."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_hash(path: Path) -> str:
    """Hash named array contents independently of ZIP compression and exact values."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as arrays:
        for name in sorted(set(arrays.files) - {"exact_margin_p1"}):
            array = arrays[name]
            header = json.dumps([name, array.dtype.descr, array.shape],
                                ensure_ascii=True, separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(array.nbytes.to_bytes(8, "big"))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def request(checkpoint: Path, signature: str, num_games: int) -> dict[str, object]:
    """Settings that must match at collection start, completion, and reuse."""
    if num_games <= 0:
        raise ValueError("num_games must be positive")
    return {"version": 1, "checkpoint_sha256": checkpoint_hash(checkpoint),
            "signature": signature, "num_games": num_games}


def staging_path(games: Path) -> Path:
    """Keep staging beside the final directory so publishing is one atomic rename."""
    return games.with_name(games.name + ".collecting")


def atomic_json(path: Path, value: dict[str, object]) -> None:
    """Publish a fully written JSON document through a sibling temporary file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def game_hashes(directory: Path, num_games: int) -> dict[str, str]:
    """Require every expected NPZ before admitting a collection as complete."""
    paths = sorted(directory.rglob("*.npz"))
    if len(paths) != num_games:
        raise ValueError(f"collection has {len(paths)} NPZ games; "
                         f"expected {num_games}: {directory}")
    return {str(path.relative_to(directory)): semantic_hash(path) for path in paths}


def begin(games: Path, checkpoint: Path, signature: str, num_games: int, archive: Path) -> str:
    """Reuse a verified collection or preserve interrupted staging and start fresh."""
    expected = request(checkpoint, signature, num_games)
    if games.exists():
        receipt_path = games / RECEIPT
        if not receipt_path.is_file():
            raise ValueError(f"existing collection has no completion receipt: {games}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "complete" or receipt.get("request") != expected:
            raise ValueError(f"collection checkpoint or settings differ from completion receipt: "
                             f"{games}")
        if receipt.get("files") != game_hashes(games, num_games):
            raise ValueError(f"collection NPZ content differs from completion receipt: {games}")
        if request(checkpoint, signature, num_games) != expected:
            raise ValueError("checkpoint changed while checking collection reuse")
        return "reuse"
    staging = staging_path(games)
    if staging.exists():
        archive.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix=games.name + "-interrupted-", dir=archive))
        saved = destination / staging.name
        shutil.move(str(staging), str(saved))
        print(f"Preserved interrupted collection: {saved}", file=sys.stderr, flush=True)
    staging.mkdir(parents=True)
    atomic_json(staging / REQUEST, expected)
    return "collect"


def finish(games: Path, checkpoint: Path, signature: str, num_games: int, archive: Path) -> str:
    """Verify the staged request and publish the complete collection without replacing data."""
    if games.exists():
        raise FileExistsError(f"refusing to replace an existing collection: {games}")
    staging = staging_path(games)
    expected = request(checkpoint, signature, num_games)
    if json.loads((staging / REQUEST).read_text()) != expected:
        raise ValueError("collection checkpoint or settings changed since begin")
    files = game_hashes(staging, num_games)
    if request(checkpoint, signature, num_games) != expected:
        raise ValueError("checkpoint changed while finishing collection")
    atomic_json(staging / RECEIPT, {"status": "complete", "request": expected, "files": files})
    staging.rename(games)
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["begin", "finish"])
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--num-games", type=int, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    operation = begin if args.action == "begin" else finish
    print(operation(args.games, args.checkpoint, args.signature, args.num_games, args.archive))


if __name__ == "__main__":
    main()
