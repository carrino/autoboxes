"""Tabulate league_state.json and timing/ into report.md (reads disk only)."""
from __future__ import annotations

import json
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent


def main() -> None:
    state = json.loads((EXP_DIR / "league_state.json").read_text())
    lines = ["# boxes-3x3-loop report", "", f"Champion: iter{state['champion']}", "",
             "| iter | vs champion | vs greedy | vs ab-d4 | promoted |", "|---|---|---|---|---|"]
    for e in state["history"]:
        champ = e["vs_champion"]
        vc = f"{champ['a_win_rate']:.2f} {champ['a_win_rate_ci95']}" if champ else "bootstrap"
        lines.append(f"| {e['iteration']} | {vc} | {e['vs_boxes-greedy']['a_win_rate']:.2f} "
                     f"(margin {e['vs_boxes-greedy']['a_mean_margin']:+.2f}) | "
                     f"{e['vs_boxes-ab-d4']['a_win_rate']:.2f} "
                     f"(margin {e['vs_boxes-ab-d4']['a_mean_margin']:+.2f}) | {e['promoted']} |")
    timing_dir = EXP_DIR / "timing"
    if timing_dir.exists():
        lines += ["", "| iter | collect s | train s | arena s |", "|---|---|---|---|"]
        for f in sorted(timing_dir.glob("it*.json")):
            t = json.loads(f.read_text())
            lines.append(f"| {t['iteration']} | {t.get('collect', 0):.0f} | {t.get('train', 0):.0f} "
                         f"| {t.get('arena', 0):.0f} |")
    (EXP_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
