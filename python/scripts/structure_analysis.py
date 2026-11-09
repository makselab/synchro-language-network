"""Symmetry of the baseline connectome and deletion-only inference of the
resting-state and task networks (Fig. 4, Supplementary Figs. 4-6).

1. The MILP finds all configurations with the minimum number of removed links
   whose fiber partition equals the CS partition (binary connectome only).
2. Among them, the network shown in Fig. 4 is the one whose removed links are
   weakest in the percolation order of the tract weights (Rosen and Halgren).

Writes results/structure_summary.txt and results/inferred_networks.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro.config import REST_PARTITION, TASK_PARTITION                     # noqa: E402
from synchro.structure import (all_min_deletions, automorphisms, certify, fibers,  # noqa: E402
                               load_baseline, orbits, same_partition, select_by_percolation)
from synchro.weighted import load_tract_weights                              # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
DATA = Path(__file__).resolve().parents[2] / "data"


def fmt(partition):
    return "  ".join("{" + ", ".join(g) + "}" for g in partition)


def main():
    RESULTS.mkdir(exist_ok=True)
    G = load_baseline()
    W = load_tract_weights(G)
    lines = [f"Baseline connectome: {G.number_of_nodes()} ROIs, {G.number_of_edges() // 2} tracts, "
             f"{G.number_of_edges()} directed edges",
             f"  automorphisms: {len(automorphisms(G))}",
             f"  orbits ({len(orbits(G))}): {fmt(orbits(G))}",
             f"  fibers ({len(fibers(G))}): {fmt(fibers(G))}",
             f"  fibers equal orbits: {same_partition(fibers(G), orbits(G))}", ""]
    out = {}
    for name, P, fname in [("resting_state", REST_PARTITION, "resting_state_network_fig4b.json"),
                           ("task", TASK_PARTITION, "task_network_fig4c.json")]:
        sols = all_min_deletions(G, P)
        H, n, ties = select_by_percolation(G, P, W)
        removed_always = set.intersection(*[set(G.edges()) - set(S.edges()) for S in sols])
        removed = sorted(set(G.edges()) - set(H.edges()))
        with open(DATA / fname) as f:
            shown = {tuple(e) for e in json.load(f)["edges"]}
        lines += [f"{name}: target partition ({len(P)} classes) {fmt(P)}",
                  f"  minimum number of removed edges: {G.number_of_edges() - H.number_of_edges()} of {G.number_of_edges()}",
                  f"  optimal configurations: {n} (all certified: {all(all(certify(G, S, P).values()) for S in sols)};"
                  f" automorphisms {sorted({len(automorphisms(S)) for S in sols})})",
                  f"  edges removed in every optimal configuration: {len(removed_always)}",
                  f"  percolation selection: unique = {ties == 1}; removed edges: {removed}",
                  f"  selected network: {len(automorphisms(H))} automorphism(s), {len(fibers(H))} fibers, "
                  f"{len(orbits(H))} orbits; equals data/{fname}: {set(H.edges()) == shown}", ""]
        out[name] = {"n_optimal_configurations": n, "selected_removed_edges": [list(e) for e in removed],
                     "selected_edges": sorted(map(list, H.edges()))}
    text = "\n".join(lines)
    print(text)
    (RESULTS / "structure_summary.txt").write_text(text + "\n")
    with open(RESULTS / "inferred_networks.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
