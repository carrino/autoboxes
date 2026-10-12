"""Tabulate league_state-<tag>.json, the training RESULT lines in logs/<tag>/train-it*.log and
timing/<tag>/ into report-<tag>.md (reads disk only).

The second table is the training curve: train and held-out loss, held-out policy / value
accuracy, optimizer steps and positions per iteration, with the phase times. A held-out loss
that stops falling while the train loss keeps falling is the net memorising its replay
window.

Usage: uv run analyze.py <tag>   (tag = <rows>x<cols>, e.g. 3x3 or 5x5)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent


def result_line(log: Path) -> dict:
    """The JSON after the last ===RESULT=== marker in a log, or {} if the run never got there."""
    lines = log.read_text().splitlines()
    marks = [i for i, line in enumerate(lines) if line.strip() == "===RESULT==="]
    return json.loads(lines[marks[-1] + 1]) if marks else {}


def num(d: dict, key: str, decimals: int) -> str:
    return f"{d[key]:.{decimals}f}" if key in d else "-"


def main() -> None:
    tag = sys.argv[1]
    state = json.loads((EXP_DIR / f"league_state-{tag}.json").read_text())
    history = state["history"]
    baselines = sorted({k[3:] for e in history for k in e
                        if k.startswith("vs_") and k != "vs_champion"})
    header = "| iter | vs champion | " + " | ".join(
        f"vs {b.removeprefix('boxes-')}" for b in baselines)
    lines = [f"# boxes-{tag}-loop report", "", f"Champion: iter{state['champion']}", "",
             header + " | promoted |", "|---" * (3 + len(baselines)) + "|"]
    for e in history:
        champ = e["vs_champion"]
        # Paired arenas report the pair score (colour cancels inside a pair); older ones did not.
        vc = ("bootstrap" if not champ else
              f"pairs {champ['a_pair_score']:.2f} {champ['a_pair_score_ci95']}, "
              f"margin {champ['a_mean_margin']:+.1f}" if "a_pair_score" in champ else
              f"{champ['a_win_rate']:.2f} {champ['a_win_rate_ci95']}")
        cells = [f"{e[f'vs_{b}']['a_win_rate']:.2f} (margin {e[f'vs_{b}']['a_mean_margin']:+.2f})"
                 if f"vs_{b}" in e else "-" for b in baselines]
        lines.append(f"| {e['iteration']} | {vc} | " + " | ".join(cells) + f" | {e['promoted']} |")
    timing = {int(f.stem[2:]): json.loads(f.read_text())
              for f in (EXP_DIR / "timing" / tag).glob("it*.json")}
    train = {int(f.stem[8:]): result_line(f)
             for f in (EXP_DIR / "logs" / tag).glob("train-it*.log")}
    if timing or train:
        lines += ["", "| iter | train loss | val loss | val policy acc | val value acc | steps "
                  "| positions | collect s | train s | arena s |", "|---" * 10 + "|"]
        for it in sorted(timing | train):
            r, t = train.get(it, {}), timing.get(it, {})
            lines.append(
                f"| {it} | {num(r, 'train_loss', 3)} | {num(r, 'val_loss', 3)} "
                f"| {num(r, 'val_policy_acc', 3)} | {num(r, 'val_value_acc', 3)} "
                f"| {num(r, 'steps_completed', 0)} | {num(r, 'positions', 0)} "
                f"| {num(t, 'collect', 0)} | {num(t, 'train', 0)} | {num(t, 'arena', 0)} |")
    reads_path = EXP_DIR / "data" / f"oracle_reads-{tag}.csv"
    if reads_path.exists():
        # Latest read of each checkpoint per evaluation setting (positions, band, solver, sims).
        latest: dict[tuple[str, ...], dict[str, str]] = {}
        with reads_path.open() as f:
            for row in csv.DictReader(f):
                if row["iteration"]:
                    key = (row["positions_tag"], row["min_undrawn"], row["max_undrawn"],
                           row["solver_max_undrawn"], row["num_simulations"], row["iteration"])
                    latest[key] = row
        lines += ["", "Clean band reads (exact oracle; at 300 positions, differences under "
                  "3 points are noise):", "",
                  "| positions | band | solver | sims | iter | policy optimal | search optimal "
                  "| value sign | read at |", "|---" * 9 + "|"]
        for key in sorted(latest, key=lambda k: (k[:5], int(k[5]))):
            r = latest[key]
            lines.append(f"| {r['positions_tag']} | {r['min_undrawn']}..{r['max_undrawn']} "
                         f"| {r['solver_max_undrawn']} | {r['num_simulations']} | {r['iteration']} "
                         f"| {r['policy_optimal']} | {r['search_optimal']} | {r['value_sign']} "
                         f"| {r['read_at']} |")
    (EXP_DIR / f"report-{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
