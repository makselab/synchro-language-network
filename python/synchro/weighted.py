"""Weighted balanced coloring with tolerance (quasi-fibration analysis, Fig. 7).

Tract weights are taken from the Rosen and Halgren (2021) probabilistic
tractography connectome: the weight of u -> v is the mean of 10**Fpt over the
parcels of u and v (SMA = left and right SCEF). For a target partition, the
minimum tolerance eps is the smallest value for which some deletion-only
subgraph of the connectome (every node keeping at least one input) has
weighted in-strengths from each class that agree within eps for all nodes of
the same class. eps is reported as a fraction of the strongest tract weight.
"""

import json
from pathlib import Path

import numpy as np

from .io import DATA_DIR
from .structure import certify, min_deletion


def load_tract_weights(G, data_dir=DATA_DIR):
    """Dict edge -> linear tract weight for the edges of G."""
    with open(Path(data_dir) / "rosen_halgren_submatrix.json") as f:
        doc = json.load(f)
    F = np.array([[np.nan if x is None else x for x in row] for row in doc["log10_Fpt"]])
    pos = {p: i for i, p in enumerate(doc["parcels"])}
    parcels = {roi: [pos[p] for p in ps] for roi, ps in doc["roi_parcels"].items()}

    def w(u, v):
        vals = [10.0 ** F[a, b] for a in parcels[u] for b in parcels[v]]
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"undefined weight for {u}->{v}")
        return float(np.mean(vals))

    return {(u, v): w(u, v) for u, v in G.edges()}


def feasible(G, partition, weight, eps):
    """Deletion-only weighted problem at tolerance eps; returns (ok, subgraph).

    ok is True only if the MILP solution passes the independent certification."""
    H = min_deletion(G, partition, weight=weight, eps=eps)
    if H is None:
        return False, None
    c = certify(G, H, partition, weight=weight, eps=eps)
    return all(c.values()), H


def min_tolerance(G, partition, weight, rel_precision=1e-6):
    """Smallest eps (fraction of the strongest tract weight) making partition feasible."""
    wmax = max(weight.values())
    if feasible(G, partition, weight, 0.0)[0]:
        return 0.0
    lo, hi = 0.0, wmax
    if not feasible(G, partition, weight, hi)[0]:
        return None
    while hi - lo > rel_precision * wmax:
        mid = 0.5 * (lo + hi)
        if feasible(G, partition, weight, mid)[0]:
            hi = mid
        else:
            lo = mid
    return hi / wmax
