"""Figure 3: symmetry formalism (automorphism and symmetry fibration), vector drawing."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synchro import figstyle  # noqa: E402
figstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
W, H = 180, 132
RED = "#E8305A"
COL = {"red": "#E8301C", "green": "#4DB848", "yellow": "#F7E11B", "magenta": "#D62AD0", "grey": "#B0B0B0"}

# example network: A[i, j] = 1 if i -> j (nodes 1..5), as in the matrix of panel a
A = np.array([[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]])
EDGES = [(i + 1, j + 1) for i in range(5) for j in range(5) if A[i, j]]
POS = {1: (-1.0, 0.55), 2: (-1.0, -0.55), 3: (0.0, 0.0), 4: (1.0, -0.55), 5: (1.0, 0.55), 6: (1.8, -0.55)}

fig = plt.figure(figsize=(W * figstyle.MM, H * figstyle.MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


def node(x, y, label, color=None, r=2.0):
    ax.add_patch(plt.Circle((x, y), r, facecolor=COL[color] if color else "white", edgecolor="black", lw=0.6, zorder=3))
    ax.text(x, y, str(label), ha="center", va="center", fontsize=5.5, zorder=4)


def arrow(p, q, both=False, rad=0.0, color="black", lw=0.6, shrink=5.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=6,
                                 shrinkA=shrink, shrinkB=shrink, connectionstyle=f"arc3,rad={rad}",
                                 color=color, lw=lw, zorder=2))


def graph(cx, cy, s, edges, labels=None, colors=None):
    labels = labels or {k: k for k in POS}
    nodes = {k for e in edges for k in e}
    P = {k: (cx + s * POS[k][0], cy + s * POS[k][1]) for k in nodes}
    done = set()
    for u, v in edges:
        if (u, v) in done:
            continue
        both = (v, u) in edges
        done |= {(u, v), (v, u)}
        arrow(P[u], P[v], both)
    for k in nodes:
        node(*P[k], labels[k], colors.get(k) if colors else None)


def matrix(cx, cy, M, cell=3.2, fs=5.5):
    n = len(M)
    for i in range(n):
        for j in range(n):
            ax.text(cx + (j - (n - 1) / 2) * cell, cy - (i - (n - 1) / 2) * cell, str(M[i][j]),
                    ha="center", va="center", fontsize=fs)
    h = n * cell + 1.5
    for side in (-1, 1):
        ax.add_patch(Arc((cx + side * (n * cell / 2 - 0.5), cy), 3.2, h, theta1=90 if side < 0 else -90,
                         theta2=270 if side < 0 else 90, lw=0.6))


def tree(root, x0, y0, width, depth=3, dy=8.0, inputs=None):
    """Input tree of `root` drawn downwards to `depth` levels, with dashed continuation."""
    levels = [[(root, None)]]
    for _ in range(depth):
        levels.append([(c, i) for i, (p, _) in enumerate(levels[-1]) for c in sorted(inputs[p])])
    xs = [None] * (depth + 1)
    n = len(levels[-1])
    xs[depth] = [x0 - width / 2 + width * (k + 0.5) / n for k in range(n)]
    for d in range(depth - 1, -1, -1):
        xs[d] = [np.mean([xs[d + 1][k] for k, (_, par) in enumerate(levels[d + 1]) if par == i])
                 for i in range(len(levels[d]))]
    for d in range(depth + 1):
        y = y0 - d * dy
        for k, (lab, par) in enumerate(levels[d]):
            node(xs[d][k], y, lab, r=1.9 if d == 0 else 1.7)
            if par is not None:
                arrow((xs[d][k], y), (xs[d - 1][par], y + dy), shrink=4.5)
    for x in xs[depth]:
        ax.add_patch(FancyArrowPatch((x, y0 - depth * dy - 6), (x, y0 - depth * dy), arrowstyle="-|>",
                                     mutation_scale=5, shrinkB=4.5, lw=0.5, linestyle=(0, (1.5, 1.5)), zorder=2))
    return {lab: (xs[1][k], y0 - dy) for k, (lab, _) in enumerate(levels[1])}


# ---------------------------------------------------------------- panel a
ya = H - 6
ax.text(2, H - 2, "a", fontsize=8, fontweight="bold", va="top")
ax.text(W / 2, H - 2.5, "Network with a symmetry group", fontsize=figstyle.TITLE, ha="center", va="top")
perm = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
cols = [18, 55, 92, 128, 163]
for x, t in zip(cols, ["A", "A′", "A", "A′", "Orbits"]):
    ax.text(x, ya - 6, t, fontsize=7, ha="center", va="top", style="italic" if t != "Orbits" else "normal")
gy = ya - 20
graph(cols[0], gy, 12, EDGES)
graph(cols[1], gy, 12, EDGES, labels={k: perm.get(k, k) for k in POS})
matrix(cols[2], gy, A.tolist())
matrix(cols[3], gy, A.tolist())
# swaps of rows/columns 1<->5 and 2<->4 under the permutation
c3, cell = cols[3], 3.2
ax.add_patch(FancyArrowPatch((c3 - 2 * cell, gy + 9.5), (c3 + 2 * cell, gy + 9.5), arrowstyle="<|-|>", mutation_scale=4,
                             connectionstyle="arc3,rad=-0.3", color=RED, lw=0.6))
ax.add_patch(FancyArrowPatch((c3 - cell, gy - 9.5), (c3 + cell, gy - 9.5), arrowstyle="<|-|>", mutation_scale=4,
                             connectionstyle="arc3,rad=0.4", color=RED, lw=0.6))
ax.add_patch(FancyArrowPatch((c3 - 10.2, gy + 2 * cell), (c3 - 10.2, gy - 2 * cell), arrowstyle="<|-|>",
                             mutation_scale=4, connectionstyle="arc3,rad=0.3", color=RED, lw=0.6))
ax.add_patch(FancyArrowPatch((c3 + 10.2, gy + cell), (c3 + 10.2, gy - cell), arrowstyle="<|-|>",
                             mutation_scale=4, connectionstyle="arc3,rad=-0.5", color=RED, lw=0.6))
graph(cols[4], gy, 12, EDGES, colors={1: "red", 5: "red", 3: "green", 2: "yellow", 4: "yellow"})
# permutation and permutation matrix
py = gy - 26
ax.text(cols[1] - 23, py, "π =", fontsize=7, ha="right", va="center")
for k in range(1, 6):
    x = cols[1] - 19 + (k - 1) * 3.4
    ax.text(x, py + 3.4, str(k), fontsize=5.5, ha="center", va="center")
    ax.text(x, py, "↓", fontsize=5.5, ha="center", va="center")
    ax.text(x, py - 3.4, str(perm[k]), fontsize=5.5, ha="center", va="center")
for side in (-1, 1):
    ax.add_patch(Arc((cols[1] - 19 + 6.8 + side * 8.8, py), 3.2, 12, theta1=90 if side < 0 else -90,
                     theta2=270 if side < 0 else 90, lw=0.6))
ax.add_patch(FancyArrowPatch((cols[1] + 3, py), (cols[2] - 20, py), arrowstyle="<|-|>", mutation_scale=6, lw=0.6))
ax.text(cols[2] - 12, py, r"$P$ =", fontsize=7, ha="right", va="center")
Pm = [[1 if perm[i + 1] == j + 1 else 0 for j in range(5)] for i in range(5)]
matrix(cols[2] + 1, py, Pm)
ax.text(cols[3] + 8, py, r"$A = P^{-1}A^{\prime}P$", fontsize=7.2, ha="center", va="center")

# ---------------------------------------------------------------- panel b
yb = 62
ax.text(2, yb, "b", fontsize=8, fontweight="bold", va="top")
ax.text(W / 2, yb - 0.5, "Network with no symmetry group and a fibration symmetry", fontsize=figstyle.TITLE, ha="center", va="top")
EB = EDGES + [(4, 6)]
inputs = {k: [u for u, v in EB if v == k] for k in range(1, 7)}
ax.text(45, yb - 8, "Isomorphic input trees", fontsize=7, ha="center", va="top")
t1 = tree(1, 22, yb - 18, 40, inputs=inputs)
t5 = tree(5, 68, yb - 18, 40, inputs=inputs)
ax.add_patch(FancyArrowPatch((25, yb - 16), (65, yb - 16), arrowstyle="<|-|>", mutation_scale=5,
                             connectionstyle="arc3,rad=-0.15", color=RED, lw=0.6))
ax.add_patch(FancyArrowPatch((t1[2][0] + 2.5, t1[2][1] + 1.5), (t5[4][0] - 2.5, t5[4][1] + 1.5), arrowstyle="<|-|>",
                             mutation_scale=5, connectionstyle="arc3,rad=-0.12", color=RED, lw=0.6))
fib = {1: "yellow", 5: "yellow", 3: "green", 2: "magenta", 4: "magenta", 6: "grey"}
ax.text(126, yb - 12, "G", fontsize=7, ha="center", va="top", style="italic")
graph(124, yb - 30, 12, EB, colors=fib)
ax.text(167, yb - 12, "Base B", fontsize=7, ha="center", va="top")
bx, by = 165, yb - 30
B = {"Y": (bx, by + 9), "G": (bx, by), "M": (bx, by - 9), "X": (bx + 10, by - 9)}
arrow(B["Y"], B["G"], rad=0.45)
arrow(B["Y"], B["G"], rad=0.95)
arrow(B["G"], B["Y"], rad=0.45)
arrow(B["G"], B["M"], rad=0.45)
arrow(B["M"], B["Y"], rad=-0.9)
arrow(B["M"], B["X"])
for k, lab, c in [("Y", 1, "yellow"), ("G", 3, "green"), ("M", 2, "magenta"), ("X", 6, "grey")]:
    node(*B[k], lab, c)
ax.text(146, yb - 50, r"$\phi: G \rightarrow B$", fontsize=7, ha="center", va="center")

RESULTS.mkdir(exist_ok=True)
fig.savefig(RESULTS / "figure_3_schematic.pdf")
fig.savefig(RESULTS / "figure_3_schematic.png", dpi=250)
