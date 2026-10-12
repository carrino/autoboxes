"""Run up to four independent single-knob candidates against one frozen parent."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Log candidate commands/results and retain only gains of at least 9/300 positions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--candidates", required=True,
                        help="JSON mapping run name to one changed flag")
    parser.add_argument("--diagnostics", nargs="*", default=[],
                        help="logged ablations that do not participate in selection")
    args = parser.parse_args()
    parent = json.loads((HERE / "data" / f"{args.parent}.json").read_text())
    candidates: dict[str, dict[str, Any]] = json.loads(args.candidates)
    assert len(candidates) <= 4
    assert parent["positions_tag"] == "5x5-chains"

    def evaluate(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        name, change = item
        assert len(change) == 1
        assert set(change) <= set(parent["search_flags"])
        flags = {**parent["search_flags"], **change}
        command = [sys.executable, str(HERE / "bench.py"), "evaluate", "--positions",
                   str(HERE / "data" / "development.json"), "--name", name,
                   "--iteration", "54"]
        for key, value in flags.items():
            command.extend(["--" + key, str(int(value) if isinstance(value, bool) else value)])
        with (HERE / "logs" / f"{name}.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result: dict[str, Any] = json.loads((HERE / "data" / f"{name}.json").read_text())
        print(f"{name}: {result['correct']}/300 in {result['seconds']:.1f}s", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        results = list(pool.map(evaluate, candidates.items()))
    eligible = [r for r in results if r["name"] not in args.diagnostics]
    best = max(eligible, key=lambda result: (result["correct"], -result["seconds"]))
    retained = best["name"] if best["correct"] >= parent["correct"] + 9 else parent["name"]
    decision = {
        "parent": parent["name"], "candidates": candidates, "retained": retained,
        "diagnostics": args.diagnostics,
        "rule": "Require at least 9 additional correct moves out of 300; otherwise retain parent.",
        "results": [{"name": r["name"], "correct": r["correct"],
                     "delta": r["correct"] - parent["correct"],
                     "decision": ("diagnostic" if r["name"] in args.diagnostics else
                                  "keep" if r["name"] == retained else "discard")}
                    for r in results],
    }
    (HERE / "data" / f"{args.name}.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
