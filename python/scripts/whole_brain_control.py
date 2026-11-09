"""Whole-brain control for shared or external input (Fig. 5, Tables 1-3).

For each dataset (resting state, letter fluency, verb generation) and each of
the nine conditions, candidate shared signals are regressed out of the 11 ROI
time series, the group surrogate-corrected PLV is recomputed and the CS
partition reported in the paper is tested with Eq. (1), with the PLV tolerance
of the control analyses (config.EQ1_TOLERANCE).

Outputs in results/: whole_brain_matrices.npz (cache), whole_brain_tables.csv,
whole_brain_tables.tex, figure_5_whole_brain.pdf/.png.
Runtime: about 15 minutes (use --reuse to start from the cached matrices).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import figstyle                                         # noqa: E402
figstyle.apply()
import matplotlib.pyplot as plt                                     # noqa: E402
from synchro import ROI_LABELS, check_partition                     # noqa: E402
from synchro.config import EQ1_TOLERANCE, N_SURROGATES, REST_PARTITION, SEED, TASK_PARTITION  # noqa: E402
from synchro.wholebrain import CONDITIONS, WHOLE_BRAIN_DATASETS, condition_matrices, load_whole_brain  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
PARTITIONS = {"resting_state": REST_PARTITION, "letter_fluency": TASK_PARTITION,
              "verb_generation": TASK_PARTITION}
TITLES = {"resting_state": "Resting state", "letter_fluency": "Letter fluency",
          "verb_generation": "Verb generation"}
SHORT = {"raw": "raw", "global signal": "global", "A1": "A1", "PSL": "PSL", "thalamus": "thalamus",
         "MGN": "MGN", "inferior colliculus": "IC", "global+A1+PSL": "global+A1+PSL",
         "all combined": "all combined"}


def clique_name(c, tex=False):
    """Short name of a clique: 'SMG-SMG' for a homotopic pair, ROI list otherwise."""
    special = {frozenset(["SMA", "PreMA_L", "PreMA_R", "WA_L", "WA_R"]): "SMA-PreMA-PreMA-WA-WA",
               frozenset(["BA_R", "WA_L", "WA_R"]): "BA_R-WA-WA",
               frozenset(["SMA", "BA_L"]): "SMA-BA_L"}
    bases = [x.partition("_")[0] for x in c]
    if len(c) == 2 and bases[0] == bases[1]:
        name = f"{bases[0]}-{bases[0]}"
    else:
        name = special.get(frozenset(c), "-".join(c))
    if tex:
        return name.replace("_L", "$_{\\mathrm{L}}$").replace("_R", "$_{\\mathrm{R}}$")
    return name.replace("_L", " L").replace("_R", " R")


def ordered_clique(c):
    order = ["SMA", "PreMA_L", "PreMA_R", "BA_L", "BA_R", "WA_L", "WA_R", "AG_L", "AG_R", "SMG_L", "SMG_R"]
    return sorted(c, key=order.index)


def evaluate(mats):
    """rows[dataset][condition] = list of (clique, cluster PLV, valid) in partition order."""
    out = {}
    for name, P in PARTITIONS.items():
        out[name] = {}
        for cond in CONDITIONS:
            rows = {tuple(r["clique"]): r for r in check_partition(mats[name][cond], P, ROI_LABELS, EQ1_TOLERANCE)}
            out[name][cond] = [(ordered_clique(c), rows[tuple(c)]["cluster_plv"], rows[tuple(c)]["valid"])
                               for c in P]
    return out


def write_tables(res):
    with open(RESULTS / "whole_brain_tables.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "condition", "clique", "cluster_plv", "valid_eq1"])
        for name in res:
            for cond, rows in res[name].items():
                for c, v, ok in rows:
                    w.writerow([name, cond, "-".join(c), f"{v:.4f}", ok])
    lines = []
    for name in res:
        cl = [c for c, _, _ in res[name]["raw"]]
        lines.append(f"% {TITLES[name]}")
        lines.append("\\begin{tabular}{l" + "c" * len(cl) + "}\n\\toprule")
        lines.append("Condition & " + " & ".join(clique_name(c, tex=True) for c in cl) + " \\\\\n\\midrule")
        for cond, rows in res[name].items():
            cells = [f"{v:.3f}~\\ding{{{51 if ok else 55}}}" for _, v, ok in rows]
            lines.append(f"{cond:19s} & " + " & ".join(cells) + " \\\\")
        lines.append("\\bottomrule\n\\end{tabular}\n")
    (RESULTS / "whole_brain_tables.tex").write_text("\n".join(lines))


def figure(res):
    nrows = [len(res[n]["raw"]) for n in res]
    fig, axes = plt.subplots(3, 1, figsize=(figstyle.DOUBLE, 150 * figstyle.MM),
                             gridspec_kw={"height_ratios": nrows})
    fig.subplots_adjust(left=0.2, right=0.9, top=0.955, bottom=0.11, hspace=0.24)
    vmin, vmax = 0.15, 0.85
    cmap = plt.get_cmap("Blues")
    conds = list(CONDITIONS)
    for k, (ax, name) in enumerate(zip(axes, res)):
        cliques = [c for c, _, _ in res[name]["raw"]]
        V = np.array([[v for _, v, _ in res[name][c]] for c in conds]).T
        im = ax.imshow(V, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                dark = (V[i, j] - vmin) / (vmax - vmin) > 0.55
                ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if dark else "#1a1a1a")
        ax.set_xticks(range(len(conds)))
        if k == len(res) - 1:
            ax.set_xticklabels([SHORT[c] for c in conds], rotation=30, ha="right")
        else:
            ax.set_xticklabels([])
        ax.set_yticks(range(len(cliques)))
        ax.set_yticklabels([clique_name(c) for c in cliques])
        ax.set_title(TITLES[name], fontsize=figstyle.TITLE, loc="left")
        figstyle.panel_label(ax, "abc"[k], x=-0.27, y=1.0)
        ax.set_xticks(np.arange(-.5, len(conds)), minor=True)
        ax.set_yticks(np.arange(-.5, len(cliques)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.015)
    cb.set_label("Cluster PLV")
    cb.outline.set_linewidth(0.5)
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"figure_5_whole_brain.{ext}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="use results/whole_brain_matrices.npz if present")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    cache = RESULTS / "whole_brain_matrices.npz"
    if args.reuse and cache.exists():
        z = np.load(cache)
        mats = {n: {c: z[f"{n}|{c}"] for c in CONDITIONS} for n in WHOLE_BRAIN_DATASETS}
    else:
        mats = {}
        for n in WHOLE_BRAIN_DATASETS:
            X, R, _ = load_whole_brain(n)
            mats[n] = condition_matrices(X, R, N_SURROGATES, SEED)
            print("done", n, flush=True)
        np.savez(cache, **{f"{n}|{c}": M for n in mats for c, M in mats[n].items()})
    res = evaluate(mats)
    write_tables(res)
    figure(res)
    for name in res:
        print(f"\n{TITLES[name]}")
        for cond, rows in res[name].items():
            print(f"  {cond:20s} " + "  ".join(f"{clique_name(c)} {v:.3f}{'' if ok else ' (not valid)'}"
                                              for c, v, ok in rows))


if __name__ == "__main__":
    main()
