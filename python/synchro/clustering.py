"""Cluster synchronization (CS) criterion of Eq. (1).

A partition of the ROIs into cliques is accepted when, taking the cliques in
decreasing order of Cluster PLV (mean PLV over the links inside the clique),
each clique C satisfies

    mean_{i<j in C} sigma_ij >= sigma_kk'   for all k in C and k' outside C,

where k' ranges over the ROIs not belonging to C or to a clique with higher
Cluster PLV.
"""

import itertools

import numpy as np


def _indices(clique, labels):
    return [labels.index(x) for x in clique]


def cluster_plv(M, clique, labels):
    """Mean of M over the links inside clique (list of labels)."""
    idx = _indices(clique, labels)
    if len(idx) < 2:
        raise ValueError("a clique needs at least two ROIs")
    return float(np.mean([M[a, b] for a, b in itertools.combinations(idx, 2)]))


def check_partition(M, partition, labels, tol=0.0):
    """Test Eq. (1) for every clique of partition on the synchronization matrix M.

    Returns a list of dicts, in decreasing order of Cluster PLV, with the clique,
    its Cluster PLV, the strongest link to an ROI that is not yet assigned
    (value and ROI pair) and whether Eq. (1) holds. With tol > 0 a clique is
    valid when its Cluster PLV is less than tol below that strongest link (PLV
    tolerance of the control analyses, see config.EQ1_TOLERANCE).
    """
    M = np.asarray(M, dtype=float)
    flat = [x for c in partition for x in c]
    if len(flat) != len(set(flat)):
        raise ValueError("cliques overlap")
    unknown = set(flat) - set(labels)
    if unknown:
        raise ValueError(f"unknown ROIs: {sorted(unknown)}")
    scored = sorted(((cluster_plv(M, c, labels), list(c)) for c in partition), key=lambda t: -t[0])
    assigned, out = set(), []
    for value, clique in scored:
        idx = _indices(clique, labels)
        outside = [(M[k, o], labels[k], labels[o]) for k in idx for o in range(len(labels))
                   if o not in idx and o not in assigned]
        strongest = max(outside) if outside else (0.0, None, None)
        out.append({"clique": clique, "cluster_plv": value, "max_outside": float(strongest[0]),
                    "max_outside_link": strongest[1:], "valid": bool(value >= strongest[0] if tol == 0 else value > strongest[0] - tol)})
        assigned |= set(idx)
    return out


def percolation_components(M, labels):
    """Percolation sequence on M: links added in decreasing order of weight.

    Returns a list of (weight, (roi_a, roi_b), components), where components is
    the partition of the ROIs into connected components after adding that link.
    The candidate cliques of the Methods are read from this sequence.
    """
    n = len(labels)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links = sorted(((M[i, j], i, j) for i in range(n) for j in range(i)), reverse=True)
    seq = []
    for w, i, j in links:
        parent[find(i)] = find(j)
        groups = {}
        for k in range(n):
            groups.setdefault(find(k), []).append(labels[k])
        seq.append((float(w), (labels[i], labels[j]), sorted(groups.values(), key=lambda g: labels.index(g[0]))))
    return seq
