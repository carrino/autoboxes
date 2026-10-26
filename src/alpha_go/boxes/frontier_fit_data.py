"""Reuse complete live-loop policy bundles for a frozen fitting experiment.

The copied trajectories, exact policies, child values and family lineage are byte-for-byte
unchanged. Both manifest names refer to the same targets: the downstream comparison
changes optimization or sampling, never the data's labels. No solver is invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from alpha_go.boxes.policy_ab_data import load_game, sha256, write_json
from alpha_go.boxes.policy_loop_data import receipt
from alpha_go.boxes.policy_target_ab import bundle_files


def verify_sources(directories: list[Path]) -> dict[str, str]:
    """Verify immutable iteration manifests, identities, source games and all bundles."""
    roots = [path.resolve() for path in directories]
    assert roots and len(set(roots)) == len(roots), "require distinct policy iterations"
    hashes: dict[str, str] = {}
    source_hashes: set[str] = set()
    settings: dict[str, Any] | None = None
    for folder in roots:
        metadata_path = folder / "preparation.json"
        metadata = json.loads(metadata_path.read_text())
        assert metadata["schema"] == 1 and metadata["status"] == "complete", (
            f"incomplete policy iteration: {folder}")
        identity = metadata["identity"]
        identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        assert metadata["identity_sha256"] == identity_hash, "preparation identity changed"
        if settings is None:
            settings = identity["settings"]
        assert settings == identity["settings"], "policy iterations use different label settings"
        records = identity["sources"]
        included = [source for source in records if not source["excluded_reason"]]
        assert included and metadata["completed_games"] == metadata["included_files"] == (
            len(included)), "incomplete published game count"
        assert metadata["source_files"] == len(records)
        assert metadata["excluded_files"] == len(records) - len(included)
        names = {f"game-{i:07d}" for i in range(len(included))}
        assert set(metadata["completed_receipts"]) == names, "incomplete receipt manifest"
        assert {p.name for p in (folder / "games").iterdir()} == names, (
            "unexpected or missing published game bundle")
        paths = [metadata_path, folder / "dataset.txt"]
        assert sha256(paths[-1]) == metadata["dataset_sha256"], "dataset manifest changed"
        expected_manifest: list[str] = []
        counts: Counter[str] = Counter()
        for index, source in enumerate(included):
            source_path = Path(source["path"]).resolve()
            assert sha256(source_path) == source["sha256"], f"source changed: {source_path}"
            assert source["sha256"] not in source_hashes, "duplicate source game across iterations"
            source_hashes.add(source["sha256"])
            paths.append(source_path)
            bundle = folder / "games" / f"game-{index:07d}"
            assert sha256(bundle / "receipt.json") == metadata["completed_receipts"][bundle.name]
            saved = receipt(bundle, source, identity_hash)
            counts.update(saved["counts"])
            exact_files = sorted((bundle / "exact").glob("*.npz"))
            assert len(exact_files) == 1, "expected one labelled trajectory per source game"
            game = load_game(exact_files[0])
            n = len(game["boards"])
            marked = np.asarray(game["policy_ab_root_mask"])
            assert marked.dtype == np.bool_ and marked.shape == (n,), "invalid exact-root mask"
            assert np.array_equal(marked, game["exact_policy_root_mask"]), (
                "live-loop root mask disagrees with declared exact policies")
            assert str(np.asarray(game["source_sha256"]).item()) == source["sha256"]
            assert int(marked.sum()) == saved["counts"].get("replaced_roots", 0)
            expected_manifest.append(str((bundle / "exact").relative_to(folder)))
            children = sorted((bundle / "children").glob("*.npz"))
            assert len(children) == saved["counts"].get("child_collections", 0)
            for child_path in children:
                child = load_game(child_path)
                assert str(np.asarray(child["source_sha256"]).item()) == source["sha256"]
                assert np.array_equal(child["source_replay_group"], game["source_replay_group"])
                assert not np.asarray(child["policy_mask"]).any(), "child has invented policy"
            if children:
                expected_manifest.append(str((bundle / "children").relative_to(folder)))
            paths.extend(path for path in bundle.rglob("*") if path.is_file())
        assert (folder / "dataset.txt").read_text() == "".join(
            name + "\n" for name in expected_manifest), "unexpected dataset directories"
        assert dict(counts) == metadata["counts"], "iteration counts disagree with receipts"
        assert {path.resolve() for path in folder.rglob("*.npz")} == {
            path.resolve() for path in paths
            if path.suffix == ".npz" and path.is_relative_to(folder)
        }, "unrecorded NPZ outside published game bundles"
        for path in paths:
            hashes[str(path)] = sha256(path)
    return hashes


def prepare(directories: list[Path], output: Path) -> dict[str, Any]:
    """Copy verified loop data into a portable exact-target fitting bundle, without solving."""
    assert not output.exists(), f"output must be fresh: {output}"
    sources = verify_sources(directories)
    output.mkdir(parents=True)
    manifests: list[str] = []
    counts: Counter[str] = Counter()
    for index, folder in enumerate(directories):
        folder = folder.resolve()
        metadata = json.loads((folder / "preparation.json").read_text())
        counts.update(metadata["counts"])
        prefix = Path("iterations") / f"iteration-{index:04d}"
        for bundle in sorted((folder / "games").iterdir()):
            for path in sorted(bundle.rglob("*")):
                if path.is_file() and (path.suffix == ".npz" or path.name in (
                        "roots.csv", "children.csv")):
                    destination = output / prefix / path.relative_to(folder)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
                    assert sha256(destination) == sources[str(path)], "copy changed source bytes"
        manifests.extend(str(prefix / line) for line in
                         (folder / "dataset.txt").read_text().splitlines())
    manifest = "".join(name + "\n" for name in manifests)
    for arm in ("control", "exact"):
        (output / f"{arm}-manifest.txt").write_text(manifest)
    assert verify_sources(directories) == sources, "source bundles changed during copy"
    summary = dict(schema=1, status="complete", source_sha256=sources,
                   directories=[str(path.resolve()) for path in directories], counts=dict(counts),
                   scope="Copied live-loop exact targets; control/exact manifests share labels",
                   artifact_sha256={str(path.relative_to(output)): sha256(path)
                                    for path in sorted(output.rglob("*")) if path.is_file()})
    write_json(output / "preparation.json", summary)
    bundle_files(output)
    print(f"Reused {len(directories)} iterations: "
          f"{counts.get('replaced_roots', 0)} complete policies; no solving", flush=True)
    return summary


def main() -> None:
    """Freeze saved per-iteration targets for a separate fitting experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.directories, args.output)


if __name__ == "__main__":
    main()
