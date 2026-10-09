"""Progress plot and summary of an autoresearch results.tsv (reads disk only).

  uv run analyze_runs.py            # writes progress.png and report.md next to results.tsv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent


def main() -> None:
    runs = pd.read_csv(EXP_DIR / "results.tsv", sep="\t")
    metric = runs.columns[0]
    runs["run"] = range(1, len(runs) + 1)
    kept = runs[runs.status == "keep"]
    best_so_far = runs[metric].where(runs.status == "keep").cummax().ffill()

    fig, ax = plt.subplots(figsize=(9, 5))
    discarded = runs[runs.status != "keep"]
    ax.scatter(discarded.run, discarded[metric], marker="x", color="0.6", label="discard / crash")
    ax.scatter(kept.run, kept[metric], color="tab:green", edgecolor="black", zorder=3, label="keep")
    ax.step(runs.run, best_so_far, where="post", color="tab:blue", alpha=0.6, label="running best")
    for _, r in kept.iterrows():
        ax.annotate(r.description[:40], (r.run, r[metric]), fontsize=7, rotation=30,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("run #")
    ax.set_ylabel(metric)
    ax.set_title(f"{EXP_DIR.name}: {len(runs)} runs, {len(kept)} kept, "
                 f"best {runs[metric].max():.4f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "progress.png", dpi=120)

    baseline = float(runs[metric].iloc[0])
    best = float(kept[metric].max()) if len(kept) else baseline
    deltas = kept[metric].diff().fillna(0.0)
    lines = [f"# {EXP_DIR.name} report", "",
             f"Baseline {metric} {baseline:.4f} -> best {best:.4f} "
             f"({(best - baseline) * 100:+.1f} points)", "",
             f"Runs: {len(runs)}; keep {int((runs.status == 'keep').sum())}, "
             f"discard {int((runs.status == 'discard').sum())}, "
             f"crash {int((runs.status == 'crash').sum())}; "
             f"keep rate {(runs.status == 'keep').mean():.2f}", "",
             "## Kept runs by gain over the previous kept run", "",
             "| run | gain | metric | description |", "|---|---|---|---|"]
    for (_, r), d in sorted(zip(kept.iterrows(), deltas), key=lambda x: -x[1]):
        lines.append(f"| {r.run} | {d * 100:+.1f} | {r[metric]:.4f} | {r.description} |")
    lines += ["", "## All runs", "", "| run | metric | status | description |", "|---|---|---|---|"]
    for _, r in runs.iterrows():
        lines.append(f"| {r.run} | {r[metric]:.4f} | {r.status} | {r.description} |")
    (EXP_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()
