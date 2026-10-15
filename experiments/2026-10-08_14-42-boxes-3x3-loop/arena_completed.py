"""Recognise an arena commit even when interruption prevented its timing marker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def checkpoint_hash(path: Path) -> str:
    """Hash model bytes without loading torch or executing the checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic_json(path: Path, contents: dict[str, object]) -> None:
    """Commit complete JSON through a sibling file and one atomic rename."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(contents, indent=2) + "\n")
    temporary.replace(path)


def completed(state: Path, checkpoint: Path, iteration: int, timing: Path) -> bool:
    """Verify exactly one committed arena, recovering its missing timing marker."""
    if not state.exists():
        return False
    league = json.loads(state.read_text())
    entries = [entry for entry in league["history"] if entry["iteration"] == iteration]
    if not entries:
        return False
    if len(entries) != 1:
        raise ValueError(f"duplicate committed arena entries for iteration {iteration}: {state}")
    digest = entries[0].get("candidate_checkpoint_sha256")
    if not isinstance(digest, str) or not digest:
        raise ValueError(f"committed arena iteration {iteration} has no checkpoint digest: {state}")
    if checkpoint_hash(checkpoint) != digest:
        raise ValueError(f"checkpoint differs from committed arena iteration {iteration}: "
                         f"{checkpoint}")
    if not timing.exists():
        timing.parent.mkdir(parents=True, exist_ok=True)
        write_atomic_json(timing, {"iteration": iteration, "recovered_from_league": True})
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    args = parser.parse_args()
    print(int(completed(args.state, args.checkpoint, args.iteration, args.timing)))


if __name__ == "__main__":
    main()
