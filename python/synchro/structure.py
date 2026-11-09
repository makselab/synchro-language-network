"""Symmetry analysis of the structural networks and deletion-only MILP inference.

Networks are directed graphs on the 11 ROIs (networkx.DiGraph); an edge u -> v
means that v receives input from u. The baseline connectome is bidirected.

fibers()        minimal balanced coloring (fiber partition) by color refinement
orbits()        orbits of the automorphism group (exact search, VF2)
min_deletion()  minimum number of edges to remove from a graph so that a given
                partition becomes a balanced coloring (mixed integer linear
                programming, MILP), solved with HiGHS through scipy.optimize.milp
certify()       independent check of a solution
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .io import DATA_DIR, ROI_LABELS


def load_baseline(data_dir=DATA_DIR):
    """Baseline connectome of Fig. 1a / Fig. 4a (data/baseline_connectome.json)."""
    with open(Path(data_dir) / "baseline_connectome.json") as f:
        doc = json.load(f)
    G = nx.DiGraph()
    G.add_nodes_from(ROI_LABELS)
    for u, v in doc["tracts"]:
        if u not in ROI_LABELS or v not in ROI_LABELS:
            raise ValueError(f"unknown ROI in tract {u}-{v}")
        G.add_edge(u, v)
        G.add_edge(v, u)
    return G


def graph_from_edges(edges, nodes=ROI_LABELS):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def _canonical(groups, nodes):
    groups = [sorted(g, key=nodes.index) for g in groups]
    return sorted(groups, key=lambda g: nodes.index(g[0]))


def same_partition(p, q):
    return {frozenset(g) for g in p} == {frozenset(g) for g in q}


def fibers(G):
    """Fiber partition: coarsest coloring in which nodes of the same color
    receive the same number of inputs from each color (color refinement)."""
    nodes = list(G.nodes())
    color = {n: 0 for n in nodes}
    while True:
        sig = {n: (color[n], tuple(sorted(color[p] for p in G.predecessors(n)))) for n in nodes}
        ids = {}
        new = {n: ids.setdefault(sig[n], len(ids)) for n in nodes}
        if len(ids) == len(set(color.values())):
            break
        color = new
    groups = {}
    for n in nodes:
        groups.setdefault(color[n], []).append(n)
    return _canonical(groups.values(), nodes)


def automorphisms(G):
    return list(nx.algorithms.isomorphism.DiGraphMatcher(G, G).isomorphisms_iter())


def orbits(G):
    """Orbits of the automorphism group of G."""
    nodes = list(G.nodes())
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for a in automorphisms(G):
        for k, v in a.items():
            parent[find(k)] = find(v)
    groups = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return _canonical(groups.values(), nodes)


def is_balanced(G, partition, weight=None, tol=0.0):
    """True if every pair of nodes of the same class receives the same input
    (number of edges, or total weight if weight is a dict) from every class."""
    col = {n: c for c, g in enumerate(partition) for n in g}
    def inp(n):
        s = np.zeros(len(partition))
        for u in G.predecessors(n):
            s[col[u]] += 1.0 if weight is None else weight[(u, n)]
        return s
    return all(np.all(np.abs(inp(p) - inp(q)) <= tol + 1e-12)
               for g in partition for a, p in enumerate(g) for q in g[a + 1:])


def balance_rows(G, partition, weight=None):
    """Linear constraints A k = 0 on the keep variables k_e of the edges of G."""
    E = list(G.edges())
    ix = {e: i for i, e in enumerate(E)}
    col = {n: c for c, g in enumerate(partition) for n in g}
    rows = []
    # one constraint for every pair of nodes of the same class, so that with a
    # tolerance eps all in-strengths of a class lie within eps of each other
    for g in partition:
        for a, p in enumerate(g):
            for q in g[a + 1:]:
                for c in range(len(partition)):
                    r = np.zeros(len(E))
                    for u in G.predecessors(p):
                        if col[u] == c:
                            r[ix[(u, p)]] += 1.0 if weight is None else weight[(u, p)]
                    for u in G.predecessors(q):
                        if col[u] == c:
                            r[ix[(u, q)]] -= 1.0 if weight is None else weight[(u, q)]
                    if np.any(r):
                        rows.append(r)
    return E, np.array(rows).reshape(-1, len(E))


def min_deletion(G, partition, weight=None, eps=0.0, min_in=1):
    """Remove the minimum number of edges of G so that partition is balanced.

    Only deletions are allowed (no edge is added). Every node keeps at least
    min_in incoming edges. With weight (dict edge -> weight), the weighted
    in-strengths must agree within eps. Returns the subgraph, or None if the
    problem is infeasible.
    """
    flat = sorted(n for g in partition for n in g)
    if flat != sorted(G.nodes()):
        raise ValueError("the partition must cover every node exactly once")
    E, A = balance_rows(G, partition, weight)
    m = len(E)
    cons = []
    if len(A):
        cons.append(LinearConstraint(A, -eps, eps))
    if min_in:
        nodes = list(G.nodes())
        B = np.zeros((len(nodes), m))
        for i, (_, v) in enumerate(E):
            B[nodes.index(v), i] = 1.0
        cons.append(LinearConstraint(B, min_in, np.inf))
    res = milp(c=-np.ones(m), integrality=np.ones(m), bounds=Bounds(0, 1), constraints=cons)
    if res.status != 0:
        return None
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(e for e, k in zip(E, res.x) if k > 0.5)
    return H


def certify(G, H, partition, weight=None, eps=0.0, min_in=1):
    """Independent check of a min_deletion solution H of G.

    Returns a dict of booleans: H is a subgraph of G, every node of H has at
    least min_in inputs, partition is balanced in H (within eps if weighted).
    """
    out = {
        "subgraph": set(H.edges()) <= set(G.edges()),
        "min_in_degree": min(dict(H.in_degree()).values()) >= min_in,
        "balanced": is_balanced(H, partition, weight, eps),
    }
    if weight is None:
        out["fibers_equal_target"] = same_partition(fibers(H), partition)
    return out


def all_min_deletions(G, partition, min_in=1, max_solutions=1000):
    """All deletion-only solutions with the minimum number of removed edges
    whose fiber partition equals the target (enumeration with no-good cuts)."""
    E, A = balance_rows(G, partition)
    m = len(E)
    nodes = list(G.nodes())
    B = np.zeros((len(nodes), m))
    for i, (_, v) in enumerate(E):
        B[nodes.index(v), i] = 1.0
    cons = [LinearConstraint(A, 0, 0)] if len(A) else []
    cons.append(LinearConstraint(B, min_in, np.inf))
    best, sols = None, []
    while len(sols) < max_solutions:
        res = milp(c=-np.ones(m), integrality=np.ones(m), bounds=Bounds(0, 1), constraints=cons)
        if res.status != 0:
            break
        x = np.round(res.x)
        if best is None:
            best = x.sum()
        if x.sum() < best:
            break
        H = graph_from_edges([E[i] for i in range(m) if x[i] > 0.5], nodes)
        if same_partition(fibers(H), partition):
            sols.append(H)
        cons.append(LinearConstraint(np.where(x > 0.5, 1.0, -1.0)[None, :], -np.inf, x.sum() - 1))
    return sols


def select_by_percolation(G, partition, weight):
    """Among the deletion-only solutions with the minimum number of removed
    edges, select the one whose removed edges are weakest in the percolation
    order of the tract weights: the removed weights, sorted from the strongest,
    are compared lexicographically and the smallest sequence is chosen.

    Returns (subgraph, number of minimal solutions, number of solutions tied
    under the criterion).
    """
    sols = all_min_deletions(G, partition)
    if not sols:
        return None, 0, 0
    E = set(G.edges())

    def key(H):
        return sorted((weight[e] for e in E - set(H.edges())), reverse=True)
    sols.sort(key=key)
    ties = sum(1 for H in sols if key(H) == key(sols[0]))
    return sols[0], len(sols), ties
