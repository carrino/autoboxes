"""Resume completed balanced training with an explicitly extended evaluation time limit."""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import shlex
import signal
import sys
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location(
    "balanced_runner", SCRIPT.with_name("run_balanced.py"))
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
PROBES = ("control-vs-balanced", "iter24-vs-balanced")


def resume(output: Path, minutes: int) -> None:
    """Verify completed training and extend only subprocess limits, preserving the protocol."""
    if minutes <= 0:
        raise ValueError("Evaluation minutes must be positive")
    config = json.loads((output / "config.json").read_text())
    runner.verify(output, config)
    receipt = output / "train/completed.json"
    if not receipt.is_file():
        raise ValueError("Training must be complete; this command never starts training")

    def no_training(_attempt: Path) -> list[str]:
        raise ValueError("Training must be complete; refusing to start a new training attempt")

    # Validate every saved training artifact before using the original runner's stage reuse.
    runner.pilot.stage(output, "train", 0, no_training)
    effective = copy.deepcopy(config)
    for name in PROBES:
        effective["caps_seconds"][name] = max(config["caps_seconds"][name], minutes * 60)
    record = dict(
        schema=1, original_config_sha256=runner.digest(output / "config.json"),
        training_receipt_sha256=runner.digest(receipt),
        recovery_script_sha256=runner.digest(SCRIPT),
        original_caps_seconds=config["caps_seconds"],
        effective_caps_seconds=effective["caps_seconds"],
        scope="Evaluation wall-time extension only. Training, checkpoints, positions, labels, "
              "seeds and search settings unchanged. Completed stages reused; interrupted "
              "comparisons restart in new attempt directories, preserving old diagnostics.")
    record_path = output / f"evaluation-extension-{minutes}m.json"
    if record_path.exists():
        if json.loads(record_path.read_text()) != record:
            raise ValueError("Saved evaluation-extension identity changed")
    else:
        runner.write_json(record_path, record)
    (output / "resume-evaluation.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(runner.REPO)) + "\n" +
        shlex.join([sys.executable, "-u", str(SCRIPT), "--run", str(output),
                    "--probe-minutes", str(minutes)]) + "\n")
    limits = ", ".join(f"{name}: {effective['caps_seconds'][name] / 60:g} min" for name in PROBES)
    print(f"Training verified and preserved. Evaluation limits: {limits}.\n"
          "Interrupted comparisons restart; completed comparisons are reused.\n"
          f"Resume: bash {output / 'resume-evaluation.sh'}", flush=True)
    runner.run(output, effective)
    runner.verify(output, config)
    runner.checked(receipt, record["training_receipt_sha256"])
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["evaluation_extension"] = record
    runner.write_json(summary_path, summary)


def main() -> None:
    """Use a separate recovery entry point so the original run's code hashes still match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--probe-minutes", type=int, default=180)
    args = parser.parse_args()
    runner.helpers.pin_runtime()
    output = args.run.resolve()
    if not output.is_dir():
        raise ValueError(f"Run directory does not exist: {output}")
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def terminate(_signal: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, terminate)
        try:
            resume(output, args.probe_minutes)
        finally:
            runner.pilot.package(output)
            print(f"Send this file: {output / 'results.tgz'}", flush=True)


if __name__ == "__main__":
    main()
