#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../.."
exec uv run python -u "$HERE/run_balanced.py" "$@"
