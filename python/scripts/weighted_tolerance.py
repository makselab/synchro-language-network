"""Weighted balanced coloring under tractography weights (Fig. 7) and the
synthetic arcuate-fasciculus asymmetry test (Methods).

Outputs in results/: weighted_tolerance.csv, figure_7_weighted_tolerance.pdf/.png.
"""

import csv
import sys
from pathlib import Path

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import figstyle  # noqa: E402
figstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402
from synchro.config import REST_PARTITION, TASK_PARTITION                 # noqa: E402
from synchro.structure import fibers, load_baseline, select_by_percolation  # noqa: E402
from synchro.weighted import load_tract_weights, min_tolerance            # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"


def max_imbalance(H, partition, weight):
    col = {n: c for c, g in enumerate(partition) for n in g}

    def inp(n):
        s = np.zeros(len(partition))
        for u in H.predecessors(n):
            s[col[u]] += weight.get((u, n), 1.0)
        return s
    return max((np.abs(inp(p) - inp(q)).max() for g in partition for a, p in enumerate(g) for q in g[a + 1:]),
               default=0.0)


def main():
    RESULTS.mkdir(exist_ok=True)
    G = load_baseline()
    W = load_tract_weights(G)
    targets = [("Baseline", fibers(G)), ("Task", TASK_PARTITION), ("Resting state", REST_PARTITION)]
    rows = []
    for name, P in targets:
        eps = min_tolerance(G, P, W)
        # synthetic test: left arcuate fasciculus (BA_L <-> WA_L) scaled by lambda in the unweighted solution
        H = G if name == "Baseline" else select_by_percolation(G, P, W)[0]   # networks of Fig. 4
        lam = 1.2
        imb = max_imbalance(H, P, {("BA_L", "WA_L"): lam, ("WA_L", "BA_L"): lam})
        rows.append((name, len(P), 100 * eps, imb / (lam - 1)))
        print(f"{name:14s} {len(P)} classes  minimum tolerance {100 * eps:.2f}% of the strongest tract;"
              f"  synthetic asymmetry: imbalance / |lambda-1| = {imb / (lam - 1):.2f}")
    with open(RESULTS / "weighted_tolerance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["partition", "n_classes", "min_tolerance_percent_of_max_weight", "synthetic_imbalance_over_lambda_minus_1"])
        w.writerows([(a, b, f"{c:.3f}", f"{d:.3f}") for a, b, c, d in rows])

    fig, ax = plt.subplots(figsize=(figstyle.SINGLE, 52 * figstyle.MM))
    ax.axvspan(10, 20, color="#f0ad4e", alpha=0.18, zorder=0)
    ax.text(15, 1.0, "typical diffusion-tractography\nmeasurement uncertainty\n(10–20%)", ha="center",
            va="center", fontsize=5.5, color="#8a6116")
    y = np.arange(len(rows))
    ax.barh(y, [r[2] for r in rows], color="#3b6fb6", height=0.5, zorder=3)
    for yi, r in zip(y, rows):
        ax.text(r[2] + 0.3, yi, f"{r[2]:.1f}%", va="center", fontsize=6, color="#1a1a1a")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}\n({r[1]} fibers)" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 22)
    ax.set_xlabel(r"minimum tolerance $\varepsilon$ (% of the strongest tract weight)")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"figure_7_weighted_tolerance.{ext}", dpi=300)


if __name__ == "__main__":
    main()
