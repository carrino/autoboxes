"""Publish this report only after the running pipeline has completed verification."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_cpu_jobs import EXP, REPORT, REPO, record

BRANCH = "astra/cpu-solver-work"
SCRATCH = REPO.parent
STATE = REPORT / "data/pipeline-status.json"


def git(*arguments: str, capture: bool = False) -> str:
    """Run Git without a shell; stop immediately on failure."""
    result = subprocess.run(["git", *arguments], cwd=REPO, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def main() -> None:
    """Wait for completed verification, stage report artifacts, and push only our branch."""
    while True:
        state = json.loads(STATE.read_text())
        if state["stage"] == "failed":
            raise SystemExit("Pipeline failed; automatic publication stopped. See pipeline.log.")
        if state["stage"] == "complete":
            break
        time.sleep(15)
    verified = json.loads((REPORT / "data/full-verification.json").read_text())
    integrity = json.loads((REPORT / "data/input-integrity.json").read_text())
    archive = json.loads((REPORT / "data/archive.json").read_text())
    assert verified["verification_passed"] and integrity["verification_passed"]
    assert verified["positions"] == archive["positions"] == state["positions"]
    assert verified["maximum_policy_sum_error"] <= 1e-6
    assert archive["archive_verified_against_file_hashes"]
    assert Path(archive["path"]).stat().st_size == archive["size_bytes"]
    assert git("branch", "--show-current", capture=True) == BRANCH
    assert not git("diff", "--cached", "--name-only", capture=True), "Index is not empty"
    git("diff", "--exit-code", "b6a2034c07e4e1f0659ebfce6fbbb393abd775cd", "--",
        "src/", "tests/", str(EXP / "*.py"), str(EXP / "*.sh"))
    subprocess.run([sys.executable, str(REPORT / "write_report.py")], cwd=REPO, check=True)
    shutil.copy2(SCRATCH / "pipeline.log", REPORT / "logs/pipeline.txt")
    prefix = str(REPORT.relative_to(REPO)) + "/"
    record({"event": "publish_verified_report", "branch": BRANCH,
            "commands": ["git add -- " + prefix,
                         "git add -f -- <this report's .txt and .jsonl logs>",
                         "git diff --cached --check -- . ':(exclude)**/logs/*.txt'",
                         "git commit -m 'Report verified chains-only exact solver labels'",
                         "git push origin HEAD:refs/heads/" + BRANCH]})
    git("add", "--", prefix)
    logs = sorted(path for path in (REPORT / "logs").iterdir()
                  if path.suffix in {".txt", ".jsonl"})
    git("add", "-f", "--", *(str(path.relative_to(REPO)) for path in logs))
    staged = git("diff", "--cached", "--name-only", capture=True).splitlines()
    assert staged and all(path.startswith(prefix) for path in staged)
    assert all(Path(path).suffix in {".md", ".py", ".csv", ".json", ".txt", ".jsonl", ".png"}
               for path in staged)
    assert "scripts/build_cpp.sh" not in staged
    git("diff", "--cached", "--check", "--", ".", ":(exclude)**/logs/*.txt")
    git("commit", "-m", "Report verified chains-only exact solver labels")
    git("push", "origin", "HEAD:refs/heads/" + BRANCH)
    outcome = {"published": True, "branch": BRANCH,
               "commit": git("rev-parse", "HEAD", capture=True),
               "positions": verified["positions"], "archive": archive}
    (SCRATCH / "publication.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2), flush=True)


if __name__ == "__main__":
    main()
