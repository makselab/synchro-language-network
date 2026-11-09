"""Synchronization-matrix figures of the paper.

Requires the output of compute_matrices.py. Writes to results/:
  Figure_2.pdf                    matrix panels b, d, f (+ network panels a, c, e from assets/)
  Supplementary_Figure_2.pdf      single participant, three conditions
  Supplementary_Figure_7.pdf      resting state, Glasser parcels
  Supplementary_Figure_8.pdf      coefficient of variation across participants
  Supplementary_Figure_9.pdf      Jensen-Shannon divergence between participants
  Supplementary_Figure_10.pdf     resting state, Pearson correlation
  Supplementary_Figure_11.pdf     resting state, HRF-deconvolved, spheres
  Supplementary_Figure_12.pdf     resting state, HRF-deconvolved, Glasser parcels
  Supplementary_Figure_13.pdf     band-pass filtered task data

Usage:  python scripts/make_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402
from matplotlib.patches import Rectangle                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import ROI_LABELS, check_partition                     # noqa: E402
from synchro import figstyle                                         # noqa: E402
import networkx as nx                                               # noqa: E402
from synchro.config import (DATASETS, REST_ORDER, LETTER_ORDER, VERB_ORDER,  # noqa: E402
                            REST_SIZES, TASK_SIZES, display_label)

ROOT = Path(__file__).resolve().parents[2]
RESULTS, ASSETS = ROOT / "results", ROOT / "assets"
MM = 1 / 25.4

figstyle.apply()
# approximation of MATLAB's parula colormap, used in the original figures
PARULA = LinearSegmentedColormap.from_list("parula", [
    (0.00, (0.2081, 0.1663, 0.5292)), (0.13, (0.0118, 0.3906, 0.8710)),
    (0.25, (0.0780, 0.5200, 0.8290)), (0.38, (0.0230, 0.6290, 0.7570)),
    (0.50, (0.1540, 0.7260, 0.6120)), (0.63, (0.4500, 0.7470, 0.4460)),
    (0.75, (0.7340, 0.7240, 0.2630)), (0.88, (0.9940, 0.7380, 0.2360)),
    (1.00, (0.9763, 0.9831, 0.0538))])
SKY = LinearSegmentedColormap.from_list("sky", ["#ffffff", "#9fcbe6", "#0b72b9"])
TITLES = {"rest": "Resting state", "letter": "Letter fluency", "verb": "Verb generation"}


def participants(name, kind="plv"):
    return np.load(RESULTS / f"{name}_{kind}_participants.npy")


def reorder(M, order):
    idx = [ROI_LABELS.index(x) for x in order]
    return M[np.ix_(idx, idx)]


def draw_matrix(ax, M, order, sizes, naming="spheres", norm=None, cmap=PARULA, numbers=True, fs=5,
                diagonal=1.0):
    A = reorder(M, order).astype(float)
    if diagonal is not None:
        np.fill_diagonal(A, diagonal)
    norm = norm or Normalize(0, 1)
    im = ax.imshow(A, cmap=cmap, norm=norm, interpolation="nearest")
    labels = [display_label(x, naming) for x in order]
    ax.set_xticks(range(len(order)), labels, rotation=90)
    ax.set_yticks(range(len(order)), labels)
    ax.tick_params(length=2, pad=1.5)
    if numbers:
        for i in range(len(order)):
            for j in range(i + 1):
                r, g, b, _ = cmap(norm(A[i, j]))
                ink = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=fs, color=ink)
    start = 0
    for k in sizes:
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), k, k, fill=False, ec="#e00000", lw=1.2, zorder=3))
        start += k
    return im


def colorbar(fig, ax, im, label, **kw):
    cb = fig.colorbar(im, ax=ax, pad=0.03, **kw)
    cb.set_label(label, fontsize=6)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    return cb


def panel(ax, letter, x=-0.28, y=1.04):
    figstyle.panel_label(ax, letter, x, y)


def single_matrix_figure(M, order, sizes, naming, label, path):
    fig, ax = plt.subplots(figsize=(95 * MM, 85 * MM), layout="constrained")
    im = draw_matrix(ax, M, order, sizes, naming)
    colorbar(fig, ax, im, label, fraction=0.046)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def network_graph(M, partition):
    """Functional network of Fig. 2a,c,e: the links of the percolation on the
    group PLV matrix (links added in decreasing order of PLV) down to the
    threshold at which all ROIs are connected, plus all links inside each CS
    clique. Links between different cliques are flagged inter=True."""
    links = sorted(((M[i, j], ROI_LABELS[i], ROI_LABELS[j]) for i in range(len(ROI_LABELS)) for j in range(i)),
                   reverse=True)
    P = nx.Graph()
    P.add_nodes_from(ROI_LABELS)
    for w, u, v in links:
        P.add_edge(u, v)
        if nx.is_connected(P):
            break
    cl = {x: k for k, c in enumerate(partition) for x in c}
    G = nx.Graph()
    G.add_nodes_from(ROI_LABELS)
    for u, v in P.edges():
        G.add_edge(u, v, inter=cl[u] != cl[v])
    for c in partition:
        for a in range(len(c)):
            for b in range(a + 1, len(c)):
                G.add_edge(c[a], c[b], inter=False)
    return G


def _cross(p1, p2, p3, p4):
    """True if segments p1-p2 and p3-p4 intersect (proper crossing)."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def network_layout(G, partition, seeds=(0, 1, 2, 3), steps=8000):
    """Layout of the functional network: the ROIs of each CS clique on a small
    regular polygon; clique positions and rotations optimized (simulated
    annealing) so that inter-clique links do not cross any other link, nodes keep
    clear of other nodes and links, and inter-clique links leave their cliques
    outwards. Links inside a five-node clique (a complete graph, not planar) are
    the only crossings left."""
    cl = {x: k for k, c in enumerate(partition) for x in c}
    K = len(partition)
    rad = {2: 0.17, 3: 0.2, 4: 0.24, 5: 0.28}
    E = list(G.edges())
    inter = [(u, v) for u, v in E if G[u][v]["inter"]]
    T = nx.Graph()
    T.add_nodes_from(range(K))
    T.add_edges_from((cl[u], cl[v]) for u, v in inter)
    c0 = nx.kamada_kawai_layout(T, scale=0.45 * K ** 0.5)

    def positions(C, th, order):
        P = {}
        for k, c in enumerate(order):
            n = len(c)
            for i, x in enumerate(c):
                a = th[k] + 2 * np.pi * i / n
                P[x] = C[k] + rad[n] * np.array([np.cos(a), np.sin(a)])
        return P

    def cost(C, th, order):
        P = positions(C, th, order)
        e = 0.0
        for a in range(len(E)):
            u, v = E[a]
            for b in range(a + 1, len(E)):
                x, y = E[b]
                if {u, v} & {x, y}:
                    continue
                if cl[u] == cl[v] == cl[x] == cl[y]:
                    continue                     # crossings inside a clique (K5) are unavoidable
                if _cross(P[u], P[v], P[x], P[y]):
                    e += 300.0
        nodes = list(P)
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                if cl[nodes[a]] != cl[nodes[b]]:
                    d = np.linalg.norm(P[nodes[a]] - P[nodes[b]])
                    e += 4000.0 * max(0.0, 0.5 - d) ** 2
        for n in nodes:
            for u, v in E:
                if n in (u, v):
                    continue
                d = _seg_dist(P[n], P[u], P[v])
                e += 300.0 * max(0.0, 0.14 - d)
        for u, v in inter:
            L = np.linalg.norm(P[u] - P[v])
            e += 6.0 * (L - 0.55) ** 2
            for a, b in ((u, v), (v, u)):         # leave the clique outwards
                out = P[a] - C[cl[a]]
                d = P[b] - P[a]
                cos = np.dot(out, d) / (np.linalg.norm(out) * np.linalg.norm(d) + 1e-12)
                e += 4.0 * (1 - cos)
        return e

    best = None
    for seed in seeds:                            # independent restarts, best kept
        rng = np.random.default_rng(seed)
        C = np.array([c0[k] for k in range(K)], float) + rng.normal(0, 0.05, (K, 2))
        th = rng.uniform(0, 2 * np.pi, K)
        order = [list(c) for c in partition]
        cur = cost(C, th, order)
        for t in range(steps):
            temp = 3.0 * (1 - t / steps) + 0.01
            C2, th2, o2 = C.copy(), th.copy(), [list(c) for c in order]
            k = rng.integers(K)
            r = rng.random()
            if r < 0.4:
                C2[k] += rng.normal(0, 0.08, 2)
            elif r < 0.75 or len(o2[k]) == 2:
                th2[k] += rng.normal(0, 0.5)
            else:                                 # cliques are complete: any order on the polygon
                a, b = rng.choice(len(o2[k]), 2, replace=False)
                o2[k][a], o2[k][b] = o2[k][b], o2[k][a]
            new = cost(C2, th2, o2)
            if new < cur or rng.random() < np.exp((cur - new) / temp):
                C, th, order, cur = C2, th2, o2, new
                if best is None or cur < best[0]:
                    best = (cur, C.copy(), th.copy(), [list(c) for c in order])
    _, C, th, order = best
    pos = positions(C, th, order)
    n_cross = sum(_cross(pos[u], pos[v], pos[x], pos[y]) for a, (u, v) in enumerate(E) for (x, y) in E[a + 1:]
                  if not {u, v} & {x, y} and not cl[u] == cl[v] == cl[x] == cl[y])
    print(f"network layout: {n_cross} crossings between links (excluding links inside a clique)")
    centre = {k: C[k] for k in range(K)}
    return pos, centre, cl


# Crossing-free starting drawings of the task networks (Fig. 2c,e): the triangle
# WA_L-BA_L-BA_R as outer face, PreMA_L inside it and the SMG pair outside the
# BA_L-BA_R link (used as they are, after checking that no links cross).
TASK_START = {
    "letter_fluency_spheres": {"BA_L": (0, 2.2), "WA_L": (-2.3, -0.6), "BA_R": (1.6, -0.5), "PreMA_L": (-0.45, 0.5),
                               "SMA": (0.15, 1.3), "WA_R": (-0.4, -0.15), "PreMA_R": (-0.8, 0.85),
                               "SMG_L": (1.5, 1.2), "SMG_R": (2.1, 0.9), "AG_R": (2.3, 0.2), "AG_L": (2.2, -0.45)},
    "verb_generation_spheres": {"BA_L": (0, 2.2), "WA_L": (-2.3, -1.0), "BA_R": (1.6, -1.0), "PreMA_L": (0.1, 0.9),
                                "SMA": (-0.22, 1.5), "PreMA_R": (-0.55, 0.75), "WA_R": (-0.4, 0.1),
                                "AG_R": (-0.35, -0.45), "AG_L": (0.3, -0.6), "SMG_L": (1.4, 1.3), "SMG_R": (2.0, 1.1)},
}


def free_layout(G, partition, seeds=tuple(range(24)), steps=30000, init=None):
    """Layout for planar functional networks (task states): every node moves
    freely. Simulated annealing from several starts first removes all link
    crossings (strong penalty), then refines the drawing with crossings
    forbidden, keeping nodes clear of other nodes and links, the links of a
    clique short and the other links of moderate length."""
    cl = {x: k for k, c in enumerate(partition) for x in c}
    nodes = list(G.nodes())
    n = len(nodes)
    ix = {x: i for i, x in enumerate(nodes)}
    U = np.array([ix[u] for u, v in G.edges()]); V = np.array([ix[v] for u, v in G.edges()])
    inter = np.array([G[u][v]["inter"] for u, v in G.edges()])
    m = len(U)
    pa, pb = np.triu_indices(m, 1)
    disjoint = np.array([len({U[a], V[a], U[b], V[b]}) == 4 for a, b in zip(pa, pb)])
    pa, pb = pa[disjoint], pb[disjoint]
    na, nb = np.triu_indices(n, 1)
    notinc = np.array([[k != U[e] and k != V[e] for e in range(m)] for k in range(n)])

    def ncross(X):
        A, B, C, D = X[U[pa]], X[V[pa]], X[U[pb]], X[V[pb]]
        o = lambda p, q, r: (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
        return int(np.sum((o(C, D, A) * o(C, D, B) < 0) & (o(A, B, C) * o(A, B, D) < 0)))

    def cost(X, wcross):
        e = wcross * ncross(X)
        d = np.linalg.norm(X[na] - X[nb], axis=1)
        e += 4000.0 * np.sum(np.clip(0.45 - d, 0, None) ** 2)
        A, B = X[U], X[V]
        AB = B - A
        t = np.clip(np.einsum("kej,ej->ke", X[:, None, :] - A[None], AB) / (np.sum(AB ** 2, 1) + 1e-12), 0, 1)
        dist = np.linalg.norm(X[:, None, :] - (A[None] + t[..., None] * AB[None]), axis=2)
        e += 3000.0 * np.sum(np.clip(0.22 - dist, 0, None)[notinc])
        L = np.linalg.norm(AB, axis=1)
        e += np.sum(np.where(inter, 2.0 * (L - 0.8) ** 2, 300.0 * (L - 0.38) ** 2))
        e += 0.3 * np.sum((X - X.mean(0)) ** 2)
        return e

    best = None
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # crossing-free start: straight-line planar drawing of a randomly relabelled copy
        # (different combinatorial embeddings for different seeds)
        perm = list(nodes)
        rng.shuffle(perm)
        H = nx.Graph()
        H.add_nodes_from(perm)
        H.add_edges_from(sorted(G.edges(), key=lambda e: rng.random()))
        pl = nx.planar_layout(H, scale=2.0) if init is None else init
        X = np.array([pl[x] for x in nodes], float) + rng.normal(0, 1e-3, (n, 2))
        if ncross(X) > 0:
            continue
        cur = cost(X, 0.0)
        for t in range(steps):
            temp = 2.0 * (1 - t / steps) + 0.005
            X2 = X.copy()
            k = rng.integers(n)
            X2[k] += rng.normal(0, 0.1 if t < steps // 2 else 0.03, 2)
            if ncross(X2) > 0:
                continue
            new = cost(X2, 0.0)
            if new < cur or rng.random() < np.exp((cur - new) / temp):
                X, cur = X2, new
        if best is None or cur < best[0]:
            best = (cur, X.copy())
    if best is None:
        raise RuntimeError("no crossing-free layout found")
    X = best[1]
    pos = {x: X[ix[x]] for x in nodes}
    centre = {k: np.mean([pos[x] for x in c], 0) for k, c in enumerate(partition)}
    print(f"network layout (free): {ncross(X)} crossings between links")
    return pos, centre, cl


def _place_labels(ax, pos, G, groups, obstacles=(), fs=5.5, node_r=4.5):
    """Place each ROI label next to its node, in the free direction that avoids
    other labels, nodes and links (positions tested in display coordinates)."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    D = {n: ax.transData.transform(p) for n, p in pos.items()}
    px = fig.dpi / 72.0
    segs = [(D[u], D[v]) for u, v in G.edges()]
    boxes = [b for b in obstacles]

    def rect_hits_seg(r, a, b):
        x0, y0, x1, y1 = r
        for t in np.linspace(0, 1, 40):
            p = a + t * (b - a)
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                return True
        return False

    cen = {n: np.mean([D[m] for m in g], 0) for g in groups for n in g}

    def rect_dist(r, q):
        cx = min(max(q[0], r[0]), r[2]); cy = min(max(q[1], r[1]), r[3])
        return np.hypot(q[0] - cx, q[1] - cy)

    order = sorted(pos, key=lambda n: -G.degree(n))
    for n in order:
        best = None
        for k in range(32):
            phi = 2 * np.pi * (k % 16) / 16
            extra = 0.0 if k < 16 else 5.0
            dx, dy = np.cos(phi), np.sin(phi)
            ha = "left" if dx > 0.38 else ("right" if dx < -0.38 else "center")
            va = "bottom" if dy > 0.38 else ("top" if dy < -0.38 else "center")
            off = (node_r + 2.5 + extra) * px
            t = ax.text(*pos[n], display_label(n), fontsize=fs, ha=ha, va=va, zorder=4,
                        transform=ax.transData)
            t.set_position(ax.transData.inverted().transform(D[n] + off * np.array([dx, dy])))
            bb = t.get_window_extent(rend)
            r = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
            t.remove()
            c = 0.0
            for m, q in D.items():                          # nodes
                cx = min(max(q[0], r[0]), r[2]); cy = min(max(q[1], r[1]), r[3])
                if np.hypot(q[0] - cx, q[1] - cy) < node_r * px + 1:
                    c += 100
            for a, b in segs:                               # links
                if rect_hits_seg(r, a, b):
                    c += 50
            for o in boxes:                                 # labels and other obstacles
                g = 3 * px                                  # keep a small gap between labels
                if not (r[2] + g < o[0] or o[2] + g < r[0] or r[3] + g < o[1] or o[3] + g < r[1]):
                    c += 100
            own = rect_dist(r, D[n])                        # unambiguous: nearer to its own node
            if any(rect_dist(r, q) < own + 7 * px for m, q in D.items() if m != n):
                c += 120
            c += 0.5 * extra / 5.0                          # prefer labels close to the node
            out = D[n] - cen[n]
            if np.linalg.norm(out) > 1e-6:                  # prefer the side facing away from the clique
                c += 1.0 * (1 - np.dot(out, [dx, dy]) / np.linalg.norm(out))
            if best is None or c < best[0]:
                best = (c, ha, va, D[n] + off * np.array([dx, dy]), r)
        _, ha, va, xy, r = best
        ax.text(*ax.transData.inverted().transform(xy), display_label(n), fontsize=fs, ha=ha, va=va, zorder=4)
        boxes.append(r)


def draw_network(ax, M, partition, start=None, key_below=False):
    rows = check_partition(M, partition, ROI_LABELS)          # cliques in decreasing Cluster PLV
    rank = {x: k for k, r in enumerate(rows) for x in r["clique"]}
    G = network_graph(M, partition)
    if nx.check_planarity(G)[0]:
        if start is not None:               # hand-made crossing-free drawing (task networks)
            pos = {k: 0.5 * np.array(v, float) for k, v in start.items()}
            cl = {x: k for k, c in enumerate(partition) for x in c}
            centre = {k: np.mean([pos[x] for x in c], 0) for k, c in enumerate(partition)}
        else:
            pos, centre, cl = free_layout(G, partition)
    else:                                   # resting state: the five-node clique is not planar
        pos, centre, cl = network_layout(G, partition)
    # orientation: the most synchronized clique at the top
    ctr = np.mean(list(centre.values()), 0)
    top = centre[cl[rows[0]["clique"][0]]] - ctr
    ang = 0.0 if start is not None else np.pi / 2 - np.arctan2(top[1], top[0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pos = {n: R @ (pos[n] - ctr) for n in pos}
    E = list(G.edges())
    n_cross = sum(_cross(pos[a], pos[b], pos[c], pos[d]) for i, (a, b) in enumerate(E) for (c, d) in E[i + 1:]
                  if not {a, b} & {c, d} and not cl[a] == cl[b] == cl[c] == cl[d])
    assert n_cross == 0, f"{n_cross} link crossings in the network drawing"
    for u, v, d in G.edges(data=True):
        inter = d["inter"]
        ax.plot(*zip(pos[u], pos[v]), color="#a6a6a6" if inter else figstyle.RANK_COLORS[rank[u]],
                lw=0.9 if inter else 1.5, zorder=1, solid_capstyle="round")
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=60, color=figstyle.RANK_COLORS[rank[n]], edgecolor="#333333", linewidth=0.5, zorder=3)
    xs = np.array([p[0] for p in pos.values()]); ys = np.array([p[1] for p in pos.values()])
    pad = 0.4
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal", adjustable="datalim")    # same panel size for the three networks
    ax.axis("off")
    key = rank_key_below(ax, rows) if key_below else rank_key(ax, rows)
    _place_labels(ax, pos, G, partition, obstacles=key)
    return rows


def rank_key(ax, rows):
    """Colour key inside the network panel: cliques in decreasing Cluster PLV.
    Returns the display bounding boxes of its elements (obstacles for the labels)."""
    arts = [ax.text(1.0, 0.30, "Cluster PLV", transform=ax.transAxes, fontsize=5.5, fontweight="bold", ha="left")]
    for k, r in enumerate(rows):
        y = 0.24 - k * 0.065
        arts.append(ax.scatter(1.02, y, s=18, color=figstyle.RANK_COLORS[k], edgecolor="#333333", linewidth=0.4,
                               transform=ax.transAxes, clip_on=False))
        arts.append(ax.text(1.06, y, f"{r['cluster_plv']:.3f}", transform=ax.transAxes, fontsize=5.5, va="center"))
    ax.figure.canvas.draw()
    rend = ax.figure.canvas.get_renderer()
    return [tuple(a.get_window_extent(rend).extents) for a in arts]


def rank_key_below(ax, rows):
    """Colour key in one row under the network panel (cliques in decreasing Cluster PLV)."""
    arts = [ax.text(0.0, -0.06, "Cluster PLV", transform=ax.transAxes, fontsize=5.5, fontweight="bold",
                    ha="left", va="center")]
    x = 0.27
    for k, r in enumerate(rows):
        arts.append(ax.scatter(x, -0.06, s=16, color=figstyle.RANK_COLORS[k], edgecolor="#333333", linewidth=0.4,
                               transform=ax.transAxes, clip_on=False))
        arts.append(ax.text(x + 0.03, -0.06, f"{r['cluster_plv']:.3f}", transform=ax.transAxes, fontsize=5.5,
                            va="center"))
        x += 0.15
    ax.figure.canvas.draw()
    rend = ax.figure.canvas.get_renderer()
    return [tuple(a.get_window_extent(rend).extents) for a in arts]


def figure_2():
    """One column per condition: functional network (a, c, e) above its PLV matrix (b, d, f)."""
    W, H = 180.0, 132.0
    fig = plt.figure(figsize=(W * MM, H * MM))

    def box(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    cols = [("resting_state_spheres", REST_ORDER, REST_SIZES, "rest", "ab"),
            ("letter_fluency_spheres", LETTER_ORDER, TASK_SIZES, "letter", "cd"),
            ("verb_generation_spheres", VERB_ORDER, TASK_SIZES, "verb", "ef")]
    for c, (name, order, sizes, key, letters) in enumerate(cols):
        x0 = 2 + c * 58
        M = participants(name).mean(0)
        partition = DATASETS[name][0]
        axn = box(x0 + 2, 70, 50, 54)
        draw_network(axn, M, partition, start=TASK_START.get(name), key_below=True)
        fig.text((x0 + 0.5) / W, (H - 1.5) / H, letters[0], fontsize=8, fontweight="bold", va="top")
        fig.text((x0 + 5) / W, (H - 1.7) / H, TITLES[key], fontsize=figstyle.TITLE, va="top")
        axm = box(x0 + 11, 15, 40, 40)
        im = draw_matrix(axm, M, order, sizes, norm=LogNorm(0.1, 1), numbers=False)
        fig.text((x0 + 0.5) / W, 61 / H, letters[1], fontsize=8, fontweight="bold", va="top")
    cax = box(169.5, 15, 2.0, 40)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("PLV", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    cb.set_ticks([0.1, 0.2, 0.5, 1], labels=["0.1", "0.2", "0.5", "1"])
    fig.savefig(RESULTS / "Figure_2.pdf")
    fig.savefig(RESULTS / "Figure_2.png", dpi=200)
    plt.close(fig)


def three_conditions(values, path, cbar_label, numbers, cmap=PARULA, vmax=1.0):
    if numbers:   # matrices with values: one above the other, large enough to read the numbers
        fig, axs = plt.subplots(3, 1, figsize=(120 * MM, 235 * MM), layout="constrained")
    else:
        fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 66 * MM), layout="constrained")
    specs = [("rest", REST_ORDER, REST_SIZES), ("letter", LETTER_ORDER, TASK_SIZES), ("verb", VERB_ORDER, TASK_SIZES)]
    for ax, (key, order, sizes), letter in zip(axs, specs, "abc"):
        im = draw_matrix(ax, values[key], order, sizes, norm=Normalize(0, vmax), cmap=cmap, numbers=numbers,
                         fs=5, diagonal=1.0 if numbers else 0.0)
        ax.set_title(TITLES[key], fontsize=figstyle.TITLE)
        panel(ax, letter)
    colorbar(fig, list(axs), im, cbar_label, fraction=0.02)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def jensen_shannon(P, Q):
    m = (P + Q) / 2

    def kl(p, q):
        s = (p > 0) & (q > 0)
        return np.sum(p[s] * np.log(p[s] / q[s]))
    return 0.5 * (kl(P, m) + kl(Q, m))


def main():
    names = {"rest": "resting_state_spheres", "letter": "letter_fluency_spheres", "verb": "verb_generation_spheres"}
    C = {k: participants(v) for k, v in names.items()}
    figure_2()

    # Supplementary Fig. 2: the participant whose resting-state matrix is closest to the group average
    iu = np.triu_indices(11, 1)
    group = C["rest"].mean(0)[iu]
    subj = int(np.argmax([np.corrcoef(c[iu], group)[0, 1] for c in C["rest"]]))
    three_conditions({k: v[subj] for k, v in C.items()}, RESULTS / "Supplementary_Figure_2.pdf", "PLV", True)
    print(f"Supplementary Fig. 2 shows participant P{subj + 1:02d}")

    # Supplementary Fig. 8: coefficient of variation across participants
    cv = {}
    for k, v in C.items():
        mean = v.mean(0)
        cv[k] = np.divide(v.std(0), mean, out=np.zeros_like(mean), where=mean > 0)
    three_conditions(cv, RESULTS / "Supplementary_Figure_8.pdf", "Coefficient of variation", False, cmap=SKY)

    # Supplementary Fig. 9: Jensen-Shannon divergence between participants
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 66 * MM), layout="constrained")
    J = {}
    for k, v in C.items():
        P = [c[iu] / c[iu].sum() for c in v]
        n = len(P)
        J[k] = np.array([[jensen_shannon(P[a], P[b]) if a != b else 0.0 for b in range(n)] for a in range(n)])
    vmax = max(j.max() for j in J.values())
    plab = [f"P{i + 1}" for i in range(len(C["rest"]))]
    for ax, (k, j), letter in zip(axs, J.items(), "abc"):
        im = ax.imshow(j, cmap=SKY, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(len(plab)), plab, rotation=90, fontsize=5)
        ax.set_yticks(range(len(plab)), plab, fontsize=5)
        ax.set_title(TITLES[k], fontsize=figstyle.TITLE)
        panel(ax, letter)
    colorbar(fig, list(axs), im, "Jensen–Shannon divergence", fraction=0.02)
    fig.savefig(RESULTS / "Supplementary_Figure_9.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    # single-matrix supplementary figures
    for name, fig_no in [("resting_state_glasser", 7), ("resting_state_spheres_hrf_deconvolved", 11),
                         ("resting_state_glasser_hrf_deconvolved", 12)]:
        _, order, sizes, naming = DATASETS[name]
        single_matrix_figure(participants(name).mean(0), order, sizes, naming, "PLV",
                             RESULTS / f"Supplementary_Figure_{fig_no}.pdf")
    single_matrix_figure(participants("resting_state_spheres", "corr").mean(0), REST_ORDER, REST_SIZES,
                         "spheres", "Pearson correlation", RESULTS / "Supplementary_Figure_10.pdf")

    # Supplementary Fig. 13: band-pass filtered task data
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 82 * MM), layout="constrained")
    for ax, (name, order, key), letter in zip(axs, [("letter_fluency_spheres_bandpass", LETTER_ORDER, "letter"),
                                                    ("verb_generation_spheres_bandpass", VERB_ORDER, "verb")], "ab"):
        im = draw_matrix(ax, participants(name).mean(0), order, TASK_SIZES, fs=5)
        ax.set_title(TITLES[key], fontsize=figstyle.TITLE)
        panel(ax, letter)
    colorbar(fig, list(axs), im, "PLV", fraction=0.02)
    fig.savefig(RESULTS / "Supplementary_Figure_13.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("figures written to", RESULTS)


if __name__ == "__main__":
    main()
