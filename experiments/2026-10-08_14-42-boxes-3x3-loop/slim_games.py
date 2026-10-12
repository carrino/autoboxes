"""Pack stored games down to their move lists, for CPU boxes that only replay positions.

The position samplers (`oracle_eval.late_positions`, so region_stats.py, solver_label.py and
oracle_eval.py, and solver_bench.py) read nothing but each game's `moves` and `start_moves`.
A stored self-play game also carries a board and a visit distribution per move, about 20 KB
against 240 bytes of moves. This writes one gzipped tar holding, for every game under
$GAME_DATA_DIR/experiments/<this folder>/<tag>/, an NPZ with just those two arrays at the same
relative path, so unpacking it into another machine's $GAME_DATA_DIR gives the same layout.

  uv run slim_games.py --tags 5x5-chains 5x5-mid --out ~/boxes-games-slim.tgz
"""
from __future__ import annotations

import argparse
import io
import os
import tarfile
from pathlib import Path

import numpy as np

EXP_NAME = Path(__file__).resolve().parent.name
GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")).resolve()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True, help="game folders to pack, e.g. 5x5-mid")
    p.add_argument("--out", required=True, help="output .tgz")
    args = p.parse_args()
    games = 0
    with tarfile.open(args.out, "w:gz") as tar:
        for tag in args.tags:
            root = GAME_DATA_DIR / "experiments" / EXP_NAME / tag
            paths = sorted(root.rglob("*.npz"))
            assert paths, f"no .npz games under {root}"
            for path in paths:
                game = np.load(path)
                start = game["start_moves"] if "start_moves" in game else np.zeros((0, 2), np.int16)
                buf = io.BytesIO()
                np.savez_compressed(buf, moves=game["moves"], start_moves=start)
                info = tarfile.TarInfo(str(path.relative_to(GAME_DATA_DIR)))
                info.size = buf.tell()
                buf.seek(0)
                tar.addfile(info, buf)
            games += len(paths)
            print(f"{tag}: {len(paths)} games", flush=True)
    print(f"wrote {args.out}: {games} games, {Path(args.out).stat().st_size / 1e6:.1f} MB; unpack "
          f"with: tar xzf <file> -C $GAME_DATA_DIR")


if __name__ == "__main__":
    main()
