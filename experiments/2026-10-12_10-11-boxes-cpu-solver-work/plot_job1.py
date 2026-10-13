"""Plot the printed region-bin summaries; p90 curves show tails, not confidence intervals."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / "data/solver_nodes_by_region.csv").open() as stream:
    rows = list(csv.DictReader(stream))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, undrawn in zip(axes, [30, 32]):
    group = [r for r in rows if int(r["undrawn"]) == undrawn]
    x = [int(r["largest_region"]) for r in group]
    median = [int(r["nodes_median"]) for r in group]
    tail = [int(r["nodes_p90"]) for r in group]
    counts = [int(r["positions"]) for r in group]
    ax.plot(x, median, color="#1764ab", linewidth=1.5, label="Median")
    ax.scatter(x, median, s=[12 + n for n in counts], color="#1764ab", zorder=3)
    ax.plot(x, tail, color="#b65b16", linestyle="--", linewidth=1.2, label="90th percentile")
    ax.axhline(2_000_000, color="#777777", linestyle=":", linewidth=1, label="Node cap")
    ax.set_yscale("log")
    ax.set_ylim(300, 3_000_000)
    ax.set_xlim(6, 33)
    ax.set_xticks([8, 12, 16, 20, 24, 28, 32])
    ax.set_xlabel("Undrawn edges in largest independent region")
    ax.set_title(f"{undrawn} undrawn edges · {sum(counts)} positions")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Solver nodes (log scale)")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Smaller regions usually reduce exact solver cost", fontsize=14)
fig.text(0.5, 0.025, "Clean 5x5-mid positions · fresh table per position · larger dots represent more positions",
         ha="center", fontsize=9, color="#444444")
fig.tight_layout(rect=(0, 0.065, 1, 0.93))
(root / "figures").mkdir(exist_ok=True)
fig.savefig(root / "figures/solver-nodes-by-region.png", dpi=180)
print(root / "figures/solver-nodes-by-region.png")
