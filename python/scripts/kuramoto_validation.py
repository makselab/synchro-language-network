"""Dynamical validation with Kuramoto oscillators (Fig. 6, Methods).

Resting-state loop: RS partition -> deletion-only MILP -> network of Fig. 4b
(one of the optimal solutions) -> Kuramoto dynamics -> PLV clusters.
N_DRAWS random draws of natural frequencies and initial phases are simulated
with coupling K; PLV and within-fiber coherence are computed on the second
half of each run (3,000 time units).

Usage: python scripts/kuramoto_validation.py [--K 0.3]
Outputs in results/: kuramoto_summary.txt, kuramoto_outcomes.csv,
figure_6_kuramoto.pdf/.png, kuramoto_coherence_timecourse.csv.  Runtime: about 2 minutes.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import figstyle  # noqa: E402
figstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402
from synchro.config import REST_PARTITION                               # noqa: E402
from synchro.kuramoto import observed_clusters, simulate_batch           # noqa: E402
from synchro.structure import certify, graph_from_edges, load_baseline, same_partition  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
DATA = Path(__file__).resolve().parents[2] / "data"
N_DRAWS = 500
RUN_LENGTHS = [3000.0]
NAMES = ["SMG", "SMA-PreMA-WA (5 ROIs)", "AG", "BA"]
P = REST_PARTITION


def load_network():
    with open(DATA / "resting_state_network_fig4b.json") as f:
        return graph_from_edges([tuple(e) for e in json.load(f)["edges"]])


def outcome(clusters):
    if same_partition(clusters, P):
        return "partition reproduced"
    fiber = {x: c for c, g in enumerate(P) for x in g}
    if any(len({fiber[x] for x in c}) > 1 for c in clusters):
        return "fibers merged"
    return "a fiber split"


def run(args):
    K, T = args
    return T, simulate_batch(load_network(), P, range(N_DRAWS), K=K, T=T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=float, default=0.3, help="coupling strength (default 0.3)")
    K = ap.parse_args().K
    RESULTS.mkdir(exist_ok=True)
    G, H = load_baseline(), load_network()
    nodes = list(H.nodes())
    runs = dict(map(run, [(K, T) for T in RUN_LENGTHS]))
    outcomes = {T: [outcome(observed_clusters(runs[T]["plv"][s], nodes)) for s in range(N_DRAWS)]
                for T in RUN_LENGTHS}
    main_T = RUN_LENGTHS[-1]
    r = runs[main_T]
    lines = ["Resting-state network of Fig. 4b",
             f"  removed edges: {G.number_of_edges() - H.number_of_edges()} of {G.number_of_edges()}; "
             f"checks: {certify(G, H, P)}",
             f"  K = {K}, {N_DRAWS} draws, PLV and coherence on the second half of each run", ""]
    for T in RUN_LENGTHS:
        lines.append(f"  run length {T:g}: " + ", ".join(f"{k} {v}" for k, v in Counter(outcomes[T]).most_common()))
    lines.append(f"  within-fiber coherence > 0.99 (run length {main_T:g}, draws): " +
                 ", ".join(f"{n} {int((r['coherence'][:, i] > 0.99).sum())}" for i, n in enumerate(NAMES)))
    text = "\n".join(lines)
    print(text)
    (RESULTS / "kuramoto_summary.txt").write_text(text + "\n")
    with open(RESULTS / "kuramoto_outcomes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "seed", *[f"outcome_T{int(T)}" for T in RUN_LENGTHS], *[f"coherence_{n}" for n in NAMES]])
        w.writerows([(K, s, *[outcomes[T][s] for T in RUN_LENGTHS], *[f"{x:.4f}" for x in r["coherence"][s]])
                     for s in range(N_DRAWS)])

    t, coh = r["times"], r["coherence_t"]
    with open(RESULTS / "kuramoto_coherence_timecourse.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", *[f"{n} {q}" for n in NAMES for q in ("median", "q25", "q75")]])
        for k, tk in enumerate(t):
            row = []
            for i in range(len(NAMES)):
                c = coh[k, :, i]
                row += [f"{np.median(c):.6f}", f"{np.percentile(c, 25):.6f}", f"{np.percentile(c, 75):.6f}"]
            w.writerow([f"{tk:g}", *row])
    colors = ["#c0392b", "#e08e2b", "#3b6fb6", "#d4b106"]
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE, 58 * figstyle.MM))
    t, coh = r["times"], r["coherence_t"]
    for i, n in enumerate(NAMES):
        lo, hi = np.percentile(coh[:, :, i], [25, 75], axis=1)
        ax.fill_between(t, lo, hi, color=colors[i], alpha=0.18, linewidth=0)
        ax.plot(t, np.median(coh[:, :, i], 1), color=colors[i], linewidth=1.0, label=n)
    ax.set_xscale("log")
    ax.set_xlim(1, main_T)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])      # plain labels, no exponents
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(0.88, 1.003)
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("within-fiber coherence")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"figure_6_kuramoto.{ext}", dpi=300)


if __name__ == "__main__":
    main()
