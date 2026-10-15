"""Run bounded native Dabble requests and retain each completed decision."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    assert 1 <= args.limit <= 100 and args.seconds > 0
    args.output.mkdir(parents=True, exist_ok=False)
    positions = [
        json.loads(line) for line in args.positions.read_text().splitlines() if line.strip()
    ][:args.limit]
    worker = Path(__file__).with_name("dabble_worker.py")
    with (args.output / "results.jsonl").open("w") as output:
        for position in positions:
            request = dict(position, seconds=args.seconds)
            root = args.output / f"root-{position['position']:03}"
            root.mkdir()
            (root / "request.json").write_text(json.dumps(request) + "\n")
            started = time.monotonic()
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
                adapter_status="native-prefix-replay-verified",
            )
            receipt = dict(request=request, response=result,
                           wall_seconds=time.monotonic() - started)
            temporary = root / "result.json.tmp"
            temporary.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            temporary.replace(root / "result.json")
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
