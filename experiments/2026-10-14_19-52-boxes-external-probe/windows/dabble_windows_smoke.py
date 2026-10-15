"""Run three native Dabble worker requests and retain raw diagnostic output."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    positions = [
        json.loads(line) for line in args.positions.read_text().splitlines() if line.strip()
    ][:3]
    worker = Path(__file__).with_name("dabble_worker.py")
    with (args.output / "results.jsonl").open("w") as output:
        for position in positions:
            request = dict(position, seconds=args.seconds)
            root = args.output / f"root-{position['position']:03}"
            root.mkdir()
            (root / "request.json").write_text(json.dumps(request) + "\n")
            run = subprocess.run(
                [sys.executable, str(worker), "--exe", str(args.exe)],
                input=json.dumps(request) + "\n",
                text=True,
                capture_output=True,
                check=False,
            )
            (root / "worker.stdout.log").write_text(run.stdout, encoding="utf-8")
            (root / "worker.stderr.log").write_text(run.stderr, encoding="utf-8")
            run.check_returncode()
            result = dict(
                json.loads(run.stdout),
                position=position["position"],
                adapter_status="unvalidated-smoke-prototype",
            )
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
