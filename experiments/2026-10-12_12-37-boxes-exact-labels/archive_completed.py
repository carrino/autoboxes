"""Verify and archive a completed label stage without rerunning its solves."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from run_labels import HERE, verify, write_json


def main() -> None:
    """Recover the final packaging step from a completed production manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((HERE / "logs/5x5-oracle-31-36/manifest.json").read_text())
    assert result["state"] == "complete" and result["name"] == "5x5-oracle-31-36"
    result["verification"] = verify(result, HERE / "data")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        archive.add(result["output"], arcname=result["name"])
    result["archive"] = str(args.archive.resolve())
    result["archive_bytes"] = args.archive.stat().st_size
    result["archive_sha256"] = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    write_json(HERE / "data/completed.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
