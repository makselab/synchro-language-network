"""Print the Cluster PLVs and the Eq. (1) test reported in the paper.

Requires the output of compute_matrices.py. Writes results/cluster_values.csv.

Usage:  python scripts/report_values.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import check_partition, ROI_LABELS                    # noqa: E402
from synchro.config import DATASETS, REST_PARTITION, display_label, eq1_tolerance  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"


def load_group(name, kind="plv"):
    return np.loadtxt(RESULTS / f"{name}_{kind}_group.csv", delimiter=",", skiprows=1)


def report(name, M, partition, naming, writer, kind="plv"):
    print(f"\n{name} ({kind})")
    rows = check_partition(M, partition, ROI_LABELS, eq1_tolerance(name))
    for r in rows:
        clique = "-".join(display_label(x, naming) for x in r["clique"])
        link = "-".join(display_label(x, naming) for x in r["max_outside_link"]) if r["max_outside_link"][0] else "none"
        print(f"  {clique:34s} {r['cluster_plv']:.3f}   strongest outside link {r['max_outside']:.3f} ({link})"
              f"   Eq. (1) {'valid' if r['valid'] else 'NOT valid'}")
        writer.writerow([name, kind, clique, f"{r['cluster_plv']:.3f}", f"{r['max_outside']:.3f}", link, r["valid"]])
    return all(r["valid"] for r in rows)


def main():
    with open(RESULTS / "cluster_values.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "measure", "clique", "cluster_plv", "max_outside_link_value", "max_outside_link", "eq1_valid"])
        for name, (partition, _, _, naming) in DATASETS.items():
            report(name, load_group(name), partition, naming, w)
        M = load_group("resting_state_spheres", "corr")
        np.fill_diagonal(M, 0)
        report("resting_state_spheres", M, REST_PARTITION, "spheres", w, kind="corr")
        # inter-clique links quoted in the Methods (resting state)
        M = load_group("resting_state_spheres")
        print("\nresting_state_spheres: strongest links between different clusters")
        lab = {x: i for i, c in enumerate(REST_PARTITION) for x in c}
        links = sorted(((M[i, j], ROI_LABELS[i], ROI_LABELS[j]) for i in range(11) for j in range(i)
                        if lab[ROI_LABELS[i]] != lab[ROI_LABELS[j]]), reverse=True)[:3]
        for v, a, b in links:
            print(f"  {display_label(a)}-{display_label(b)}: {v:.3f}")


if __name__ == "__main__":
    main()
