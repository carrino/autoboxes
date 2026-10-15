#!/usr/bin/env bash
# Run from WSL. All files stay in a fresh scratch directory; no training changes.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git rev-parse --show-toplevel)
POSITIONS=${1:-"$HERE/../data/child-diagnostic-positions.jsonl"}
mkdir -p "$REPO/work"
RUN=$(mktemp -d "$REPO/work/boxes-windows-smoke-XXXXXX")
mkdir "$RUN/input" "$RUN/results"
touch "$RUN/launcher.log" "$RUN/code-commit.txt" "$RUN/input-sha256.txt"
archive_results() {
  status=$?
  trap - EXIT
  tar -czf "$RUN/results.tgz" -C "$RUN" results launcher.log code-commit.txt input-sha256.txt
  echo "Exit status: $status; send this file: $RUN/results.tgz"
  exit "$status"
}
trap archive_results EXIT
echo "Windows engine smoke scratch directory: $RUN" | tee -a "$RUN/launcher.log"
for tool in powershell.exe wslpath; do
  if ! command -v "$tool" >/dev/null; then
    echo "Missing $tool. Run this command inside WSL with Windows interoperability enabled." | tee -a "$RUN/launcher.log"
    exit 1
  fi
done
if [ ! -f "$POSITIONS" ]; then
  echo "Missing frozen positions file: $POSITIONS" | tee -a "$RUN/launcher.log"
  exit 1
fi
cp "$HERE/prsboxes_windows_smoke.py" "$HERE/dabble_windows_smoke.py" \
  "$HERE/dabble_worker.py" "$HERE/run_windows_smoke.ps1" "$HERE/run_windows_smoke.sh" "$RUN/input/"
cp "$POSITIONS" "$RUN/input/positions.jsonl"
sha256sum "$RUN/input/"* > "$RUN/input-sha256.txt"
git rev-parse HEAD > "$RUN/code-commit.txt"
status=0
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$RUN/input/run_windows_smoke.ps1")" \
  -InputDirectory "$(wslpath -w "$RUN/input")" -ResultDirectory "$(wslpath -w "$RUN/results")" \
  2>&1 | tee -a "$RUN/launcher.log" || status=$?
exit "$status"
