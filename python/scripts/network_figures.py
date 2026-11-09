"""Network drawings: Fig. 4 and Supplementary Figs. 4-6 (fibers, orbits, automorphisms or
a non-symmetry permutation, and input trees).

Outputs in results/: figure_4_networks.pdf/.png, supplementary_figure_4.pdf/.png, supplementary_figure_5.pdf/.png,
supplementary_figure_6.pdf/.png.
"""

import json

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import figstyle  # noqa: E402
figstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from synchro.structure import automorphisms, fibers, graph_from_edges, load_baseline, orbits  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
DATA = Path(__file__).resolve().parents[2] / "data"
NUM = ["SMA", "PreMA_L", "PreMA_R", "BA_L", "BA_R", "AG_L", "AG_R", "WA_L", "WA_R", "SMG_L", "SMG_R"]
POS = {"SMA": (0, 3.0), "PreMA_L": (-1.6, 2.35), "PreMA_R": (1.6, 2.35), "BA_L": (-0.72, 1.8), "BA_R": (0.72, 1.8),
       "AG_L": (-1.6, 1.15), "AG_R": (1.6, 1.15), "WA_L": (-0.72, 0.6), "WA_R": (0.72, 0.6),
       "SMG_L": (-1.6, 0.0), "SMG_R": (1.6, 0.0)}
LAB = {"SMA": (-0.32, 0.12, "right"), "PreMA_L": (-0.3, 0.2, "right"), "PreMA_R": (0.3, 0.2, "left"),
       "BA_L": (-0.2, 0.0, "right"), "BA_R": (0.2, 0.0, "left"), "AG_L": (-0.25, 0, "right"),
       "AG_R": (0.25, 0, "left"), "WA_L": (0.0, -0.27, "center"), "WA_R": (0.0, -0.27, "center"),
       "SMG_L": (0, -0.3, "center"), "SMG_R": (0, -0.3, "center")}
C = {"green": "#99C55A", "orange": "#F0AE4E", "yellow": "#FCEC3F", "blue": "#4EA3DC", "red": "#DA3229"}
COL_BASE = {"SMA": "green", "PreMA_L": "orange", "PreMA_R": "orange", "BA_L": "yellow", "BA_R": "yellow",
            "AG_L": "blue", "AG_R": "blue", "WA_L": "red", "WA_R": "red", "SMG_L": "red", "SMG_R": "red"}
COL_REST = dict(COL_BASE, SMA="orange", WA_L="orange", WA_R="orange")
COL_TASK = dict(COL_BASE, BA_L="green", WA_L="yellow", WA_R="yellow")
ORBIT_COLORS = ["#E8562A", "#E0158C", "#F0AE4E", "#FCEC3F", "#0E8A3E", "#4EA3DC", "#8B5A2B",
                "#DA3229", "#8C8C8C", "#159E95", "#22348F"]


def load_network(fname):
    with open(DATA / fname) as f:
        return {tuple(e) for e in json.load(f)["edges"]}


def draw_network(ax, E, colors, ox=0.0, oy=0.0, names=None, label_color=None, scale=1.0, glyph=0.55,
                 edge_color="black"):
    """colors: dict node -> hex color or None (white)."""
    P = {k: (x * scale + ox, y * scale + oy) for k, (x, y) in POS.items()}
    names = names or {n: n for n in NUM}
    done = set()
    for u, v in sorted(E):
        if (u, v) in done:
            continue
        both = (v, u) in E
        done |= {(u, v), (v, u)}
        ax.add_patch(FancyArrowPatch(P[u], P[v], arrowstyle="<|-|>" if both else "-|>", mutation_scale=10 * glyph,
                                     shrinkA=9.5 * glyph, shrinkB=9.5 * glyph, color=edge_color, lw=0.6, zorder=1))
    for n, (x, y) in P.items():
        ax.scatter(x, y, s=260 * glyph ** 2, color=colors[n] or "white", edgecolor=edge_color if colors[n] is None else "#222",
                   linewidth=0.6, zorder=3)
        dx, dy, ha = LAB[n]
        ax.text(x + dx * scale, y + dy * scale, names[n].replace("_", " "), ha=ha, va="center",
                fontsize=max(5.5, 9 * glyph), zorder=2, color=(label_color or {}).get(n, "black"))


def figure_4(base, rest, task):
    fig, ax = plt.subplots(figsize=(figstyle.DOUBLE, figstyle.DOUBLE * 10.4 / 14.8))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(-7.4, 7.4)                  # room for the PreMA labels at the outer edges
    ax.set_ylim(-6.6, 3.8)
    ax.axis("off")
    ax.set_aspect("equal")
    for E, col, ox, oy, letter, title, sub in [
            (base, COL_BASE, 0, 0.0, "a", "Baseline connectome", "Fibration symmetry\nand group symmetry"),
            (rest, COL_REST, -4.6, -5.2, "b", "Resting state", "Fibration symmetry\nand broken group symmetry"),
            (task, COL_TASK, 4.6, -5.2, "c", "Task", "Broken fibration symmetry\nand broken group symmetry")]:
        draw_network(ax, E, {n: C[c] for n, c in col.items()}, ox, oy)
        ax.text(ox - 2.2, oy + 3.55, letter, fontsize=8, fontweight="bold", va="center")
        ax.text(ox - 1.8, oy + 3.55, title, fontsize=figstyle.TITLE, va="center")
        ax.text(ox, oy - 0.62, sub, fontsize=6, ha="center", va="top", linespacing=1.3)
    for a, b in [((-2.2, 0.6), (-3.4, -0.6)), ((2.2, 0.6), (3.4, -0.6)), ((-2.2, -3.4), (2.2, -3.4))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.4, color="black"))
    save(fig, "figure_4_networks", tight=False)


BEIGE = "#B9A98C"
RED = "#E3211C"
# geometry and label placement of Supplementary Fig. 4
POS4 = {"SMA": (0, 3.3), "PreMA_L": (-1.6, 2.5), "PreMA_R": (1.6, 2.5), "BA_L": (-0.58, 1.9), "BA_R": (0.58, 1.9),
        "AG_L": (-1.6, 1.25), "AG_R": (1.6, 1.25), "WA_L": (-0.58, 0.58), "WA_R": (0.58, 0.58),
        "SMG_L": (-1.6, 0.0), "SMG_R": (1.6, 0.0)}
LAB4 = {"SMA": (-0.2, 0.08, "right"), "PreMA_L": (-0.05, 0.27, "right"), "PreMA_R": (0.05, 0.27, "left"),
        "BA_L": (-0.2, 0.25, "right"), "BA_R": (0.2, 0.25, "left"), "AG_L": (-0.22, 0.0, "right"),
        "AG_R": (0.22, 0.0, "left"), "WA_L": (0.0, -0.3, "center"), "WA_R": (0.0, -0.3, "center"),
        "SMG_L": (0.0, -0.32, "center"), "SMG_R": (0.0, -0.32, "center")}


def _label_spot(n, P, E, s, text, fs, used):
    """Label offset of node n that keeps the label clear of edges, nodes and
    other labels (first free candidate, starting from the Supplementary Fig. 4
    placement)."""
    x, y = P[n]
    u = 1 / (72 * figstyle.DOUBLE / 16)       # data units per point (16 units across the page)
    w = 0.56 * fs * len(text) * u + 0.03      # approximate label box
    h = 0.95 * fs * u
    side = 1 if n.endswith("_R") else -1
    out, inn = ("left", "right") if side > 0 else ("right", "left")
    cands = [LAB4[n], (side * 0.2, 0.25, out), (side * 0.22, 0.0, out), (side * 0.2, -0.25, out),
             (0.0, 0.3, "center"), (0.0, -0.3, "center"), (-side * 0.2, 0.25, inn), (-side * 0.2, -0.25, inn),
             (side * 0.2, 0.33, out), (side * 0.2, -0.33, out), (0.0, 0.38, "center"), (0.0, -0.38, "center")]

    def box(dx, dy, ha):
        cx, cy = x + s * dx, y + s * dy
        x0 = cx - w if ha == "right" else (cx - w / 2 if ha == "center" else cx)
        return x0, cy - h / 2, x0 + w, cy + h / 2

    def hits(b):
        x0, y0, x1, y1 = b
        k = 0
        for a_, b_ in E:
            (ax_, ay), (bx, by) = P[a_], P[b_]
            k += sum(x0 <= ax_ + t * (bx - ax_) <= x1 and y0 <= ay + t * (by - ay) <= y1
                     for t in np.linspace(0.04, 0.96, 80))
        m = 0.13 * s
        k += 100 * sum(x0 - m <= mx <= x1 + m and y0 - m <= my <= y1 + m for q, (mx, my) in P.items() if q != n)
        k += 100 * sum(not (b[2] < o[0] or o[2] < b[0] or b[3] < o[1] or o[3] < b[1]) for o in used)
        return k

    best = min(cands, key=lambda c: (hits(box(*c)), cands.index(c)))
    used.append(box(*best))
    return best


def net4(ax, E, colors, ox, oy, s=1.0, r=0.14, edge="black", names=None, label_color=None, fs=6.5):
    """Network drawn as in Supplementary Fig. 4; returns node positions and circles."""
    names = names or {n: n for n in NUM}
    P = {k: (ox + s * x, oy + s * y) for k, (x, y) in POS4.items()}
    circ, used = {}, []
    for n, (x, y) in P.items():
        fc = colors[n] if colors else "white"
        circ[n] = plt.Circle((x, y), r * s, facecolor=fc, edgecolor=edge if fc == "white" else "#222",
                             lw=0.7, zorder=3)
        ax.add_patch(circ[n])
    for n, (x, y) in P.items():
        text = names[n].replace("_", " ")
        dx, dy, ha = _label_spot(n, P, E, s, text, fs, used)
        ax.text(x + s * dx, y + s * dy, text, ha=ha, va="center", fontsize=fs, zorder=4,
                color=(label_color or {}).get(n, "black"))
    done = set()
    for u, v in sorted(E):
        if (u, v) in done:
            continue
        both = (v, u) in E
        done |= {(u, v), (v, u)}
        ax.add_patch(FancyArrowPatch(P[u], P[v], patchA=circ[u], patchB=circ[v], arrowstyle="<|-|>" if both else "-|>",
                                     mutation_scale=7, shrinkA=1, shrinkB=1, color=edge, lw=0.7, zorder=2))
    return P, circ


def tree4(ax, G, root, color, x0, y0, width, dy=1.15, r=0.19):
    """Input tree of root to depth 2, as in Supplementary Fig. 4."""
    def circ(x, y, n, fc="white", big=False):
        c = plt.Circle((x, y), r * (1.15 if big else 1.0), facecolor=fc, edgecolor="#222", lw=0.7, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, str(NUM.index(n) + 1), ha="center", va="center", fontsize=7.5, zorder=4)
        return c

    def arrow(ca, cb, a, b):
        ax.add_patch(FancyArrowPatch(a, b, patchA=ca, patchB=cb, arrowstyle="-|>", mutation_scale=7,
                                     shrinkA=0.5, shrinkB=0.5, color="black", lw=0.7, zorder=2))
    kids = sorted(G.predecessors(root), key=NUM.index)
    grand = [sorted(G.predecessors(k), key=NUM.index) for k in kids]
    n_leaves = sum(len(g) for g in grand)
    step = width / n_leaves
    ax.text(x0, y0 + 0.3, root.replace("_", " "), ha="center", va="bottom", fontsize=10)
    croot = circ(x0, y0, root, color, big=True)
    i = 0
    for k, g in zip(kids, grand):
        xl = [x0 - width / 2 + step * (i + j + 0.5) for j in range(len(g))]
        i += len(g)
        xk = float(np.mean(xl))
        ck = circ(xk, y0 - dy, k)
        arrow(ck, croot, (xk, y0 - dy), (x0, y0))
        for gg, xg in zip(g, xl):
            cg = circ(xg, y0 - 2 * dy, gg)
            arrow(cg, ck, (xg, y0 - 2 * dy), (xk, y0 - dy))
            ax.add_patch(FancyArrowPatch((xg, y0 - 2 * dy - 0.85), (xg, y0 - 2 * dy), patchB=cg, arrowstyle="-|>",
                                         mutation_scale=7, shrinkB=0.5, lw=0.6, linestyle=(0, (2.5, 2.5)),
                                         color="black", zorder=2))


def permutation(ax, x, y, name, image, w=3.6, fs=8):
    """pi = ( 1 2 ... 11 / arrows / image ), as in Supplementary Fig. 4."""
    from matplotlib.patches import Arc
    step = w / 10
    for i in range(11):
        xi = x + i * step
        ax.text(xi, y + 0.36, str(i + 1), ha="center", va="center", fontsize=fs)
        ax.add_patch(FancyArrowPatch((xi, y + 0.2), (xi, y - 0.2), arrowstyle="-|>", mutation_scale=5,
                                     lw=0.5, color="black"))
        ax.text(xi, y - 0.36, str(image[i]), ha="center", va="center", fontsize=fs)
    for cx, t1, t2 in [(x - 0.32, 90, 270), (x + w + 0.32, -90, 90)]:
        ax.add_patch(Arc((cx, y), 0.4, 1.25, theta1=t1, theta2=t2, lw=0.8))
    ax.text(x - 0.62, y, name + " =", ha="right", va="center", fontsize=12)


def swap_arrows(ax, P, circ, pairs):
    for a, b in pairs:
        pa, pb = P[a], P[b]
        if abs(pa[1] - pb[1]) < 1e-6:               # left-right pair: arc below (above for BA)
            left, right = (a, b) if pa[0] < pb[0] else (b, a)
            rad = -0.6 if left.startswith("BA") else (0.1 if left.startswith("PreMA") else 0.25)
            ax.add_patch(FancyArrowPatch(P[left], P[right], patchA=circ[left], patchB=circ[right],
                                         arrowstyle="<|-|>", mutation_scale=8, shrinkA=2, shrinkB=2,
                                         connectionstyle=f"arc3,rad={rad}", color=RED, lw=1.0, zorder=5))
        else:                                       # same side (WA <-> SMG): short arc on the outer side
            lo, up = (a, b) if pa[1] < pb[1] else (b, a)
            rad = -0.45 if lo.endswith("_L") else 0.45
            ax.add_patch(FancyArrowPatch(P[lo], P[up], patchA=circ[lo], patchB=circ[up], arrowstyle="<|-|>",
                                         mutation_scale=8, shrinkA=2, shrinkB=2,
                                         connectionstyle=f"arc3,rad={rad}", color=RED, lw=1.0, zorder=5))


def self_loops(ax, P, r):
    """Identity: a double-headed red loop on every node, as in the original figure."""
    for n, (x, y) in P.items():
        x = x + (0.0 if n == "SMA" else (0.1 if n.endswith("_L") else -0.1))
        ax.add_patch(FancyArrowPatch((x - 0.15, y + r - 0.03), (x + 0.15, y + r - 0.03), arrowstyle="<|-|>",
                                     mutation_scale=4.5, connectionstyle="arc3,rad=-1.25", color=RED, lw=0.8,
                                     zorder=5))


def non_symmetry(ax, E, perm_name, perm, by):
    """Identity and a permutation that is not a symmetry (Supplementary Figs. 5 and 6)."""
    mapping = dict(zip(NUM, [NUM[p - 1] for p in perm]))
    assert set((mapping[u], mapping[v]) for u, v in E) != set(E)       # the permutation is not a symmetry
    permutation(ax, -6.1, -2.55, r"$\pi_0$", list(range(1, 12)))
    permutation(ax, 2.6, -2.55, rf"$\pi_{perm_name}$", perm)
    ax.text(-5.3, by + 4.05, r"$\pi_0$", ha="center", va="center", fontsize=16)
    P0, c0 = net4(ax, E, None, -5.3, by, s=1.0, edge=BEIGE)
    self_loops(ax, P0, 0.14)
    pairs = sorted({tuple(sorted((n, mapping[n]), key=NUM.index)) for n in NUM if mapping[n] != n})
    moved = {n: RED for n in mapping if mapping[n] != n}
    P1, c1 = net4(ax, E, None, -0.2, by, s=1.0, edge=BEIGE, label_color=moved)
    swap_arrows(ax, P1, c1, pairs)
    net4(ax, E, None, 5.3, by, s=1.0, edge=BEIGE, names=mapping, label_color=moved)
    ax.text(2.55, by + 1.55, "≠", ha="center", va="center", fontsize=26)
    ax.add_patch(FancyArrowPatch((0.6, by + 3.6), (4.6, by + 3.6), arrowstyle="-|>", mutation_scale=11,
                                 connectionstyle="arc3,rad=-0.3", color=RED, lw=1.1))
    ax.text(2.6, by + 3.72, rf"$\pi_{perm_name}$", ha="center", va="center", fontsize=16)


def supplementary(E, colmap, perm_name, perm, tree_rows, title, fname, autos=None, orbit_colors=None):
    """Supplementary Figs. 4-6. With autos (list of (name, image, swapped pairs)) panel b shows
    the automorphisms of a symmetric network; otherwise the identity and a non-symmetry perm."""
    G = graph_from_edges(E)
    F, O = fibers(G), orbits(G)
    auts = automorphisms(G)
    if autos is None:
        assert len(auts) == 1
    else:
        images = {tuple(NUM.index(a[n]) + 1 for n in NUM) for a in auts}
        assert all(tuple(im) in images for _, im, _ in autos)
    top, cy = 5.0, -9.6
    bottom = cy - 1.05 - 4.0 * len(tree_rows)
    fig = plt.figure(figsize=(figstyle.DOUBLE, figstyle.DOUBLE * (top - bottom) / 16))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-8, 8)
    ax.set_ylim(bottom, top)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.text(0, 4.45, title, ha="center", va="center", fontsize=14)
    # a) fibers and orbits
    ax.text(-7.6, 3.75, "a)", fontsize=12, va="center")
    same = {frozenset(g) for g in F} == {frozenset(g) for g in O}
    ax.text(0, 3.45, "Fibration symmetry " + ("=" if same else "≠") + " Group symmetry", ha="center",
            va="center", fontsize=10)
    net4(ax, E, {n: C[c] for n, c in colmap.items()}, -2.5, -0.2, s=0.9)
    net4(ax, E, orbit_colors or {n: ORBIT_COLORS[NUM.index(n)] for n in NUM}, 2.4, -0.2, s=0.9)
    ax.text(-2.5, -0.78, f"{len(F)} Fibres", ha="center", va="center", fontsize=12)
    ax.text(2.4, -0.78, f"{len(O)} Orbits", ha="center", va="center", fontsize=12)
    ax.text(5.6, 2.75, "\n".join(f"{i + 1}: {n.replace('_', ' ')}" for i, n in enumerate(NUM)), fontsize=8.5,
            va="top", linespacing=1.25)
    # b) automorphisms
    ax.text(-7.6, -1.25, "b)", fontsize=12, va="center")
    ax.text(0, -1.4, "Automorphisms", ha="center", va="center", fontsize=10)
    by = -8.15
    if autos is None:
        non_symmetry(ax, E, perm_name, perm, by)
    else:
        for (name, image, pairs), cx in zip(autos, (-5.3, 0.0, 5.3)):
            permutation(ax, cx - 1.25, -2.55, rf"$\pi_{name}$", image, w=2.9, fs=7)
            ax.text(cx, by + 4.05, rf"$\pi_{name}$", ha="center", va="center", fontsize=16)
            P, cc = net4(ax, E, None, cx, by, s=1.0, edge=BEIGE)
            swap_arrows(ax, P, cc, pairs)
    # c) input trees, two per row
    ax.text(-7.6, cy + 0.1, "c)", fontsize=12, va="center")
    ax.text(0, cy, "Isomorphic Input Trees", ha="center", va="center", fontsize=10)
    for rr, row in enumerate(tree_rows):
        leaves = [sum(len(list(G.predecessors(k))) for k in G.predecessors(node)) for node in row]
        gap = 0.75 if len(row) > 3 else (1.2 if len(row) == 3 else 1.0)
        step = min(0.39, (15.6 - gap * (len(row) - 1)) / sum(leaves))
        rr_ = min(0.19, step / 2)
        widths = [step * k for k in leaves]
        x = -(sum(widths) + gap * (len(row) - 1)) / 2
        for node, w in zip(row, widths):
            tree4(ax, G, node, C[colmap[node]], x + w / 2, cy - 1.3 - 4.0 * rr, w, r=rr_)
            x += w + gap
    save(fig, fname)


def save(fig, name, tight=True):
    RESULTS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(RESULTS / f"{name}.{ext}", bbox_inches="tight" if tight else None, dpi=200)
    plt.close(fig)


def main():
    base = set(load_baseline().edges())
    rest = load_network("resting_state_network_fig4b.json")
    task = load_network("task_network_fig4c.json")
    figure_4(base, rest, task)
    LR = [("PreMA_L", "PreMA_R"), ("BA_L", "BA_R"), ("AG_L", "AG_R"), ("WA_L", "WA_R"), ("SMG_L", "SMG_R")]
    WS = [("WA_L", "SMG_L"), ("WA_R", "SMG_R")]
    supplementary(base, COL_BASE, None, None, [["BA_L", "BA_R"], ["WA_L", "WA_R"], ["SMG_L", "SMG_R"]],
                  "Symmetry analysis of dorsal stream baseline connectome", "supplementary_figure_4",
                  autos=[("1", [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], LR), ("2", [1, 2, 3, 4, 5, 6, 7, 10, 11, 8, 9], WS),
                         ("3", [1, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8], LR + WS)],
                  orbit_colors=dict(SMA="#F0AE4E", PreMA_L="#E0158C", PreMA_R="#E0158C", BA_L="#2F6DB5",
                                    BA_R="#2F6DB5", AG_L="#8C8C8C", AG_R="#8C8C8C", WA_L="#0E8A3E",
                                    WA_R="#0E8A3E", SMG_L="#0E8A3E", SMG_R="#0E8A3E"))
    # permutation pi_2 exchanges WA and SMG; pi_1 exchanges left and right
    supplementary(rest, COL_REST, "2", [1, 2, 3, 4, 5, 6, 7, 10, 11, 8, 9],
                  [["SMA", "PreMA_L", "PreMA_R", "WA_L", "WA_R"], ["AG_L", "AG_R"]],
                  "Symmetry analysis of inferred resting state structural network", "supplementary_figure_5")
    supplementary(task, COL_TASK, "1", [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10],
                  [["BA_R", "WA_L", "WA_R"], ["SMA", "BA_L"]],
                  "Symmetry analysis of inferred language structural network", "supplementary_figure_6")


if __name__ == "__main__":
    main()
