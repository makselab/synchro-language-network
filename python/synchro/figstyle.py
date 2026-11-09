"""Common style for the figures (Nature Portfolio artwork guidelines).

Sans-serif font (Arial or a metric-compatible substitute), text of 5-7 pt at
final size, TrueType fonts embedded in the PDF (editable text), RGB colours,
widths of 88 mm (single column) or 180 mm (double column), as required by
Nature Communications.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
SINGLE = 88 * MM
DOUBLE = 180 * MM
MAX_HEIGHT = 247 * MM
TITLE = 8      # panel titles (same size as the bold panel letters)
SUBTITLE = 7   # titles inside a panel

# rank colours of the CS cliques (highest Cluster PLV first), as in the original figures
RANK_COLORS = ["#E8301C", "#F39C1E", "#F7E11B", "#4DB848", "#1F8CC8"]

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 6, "axes.titlesize": 8, "axes.labelsize": 6,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "legend.fontsize": 5.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.5, "lines.linewidth": 1.0,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "savefig.dpi": 600,
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
}


def apply():
    plt.rcParams.update(STYLE)


def panel_label(ax, letter, x=-0.12, y=1.02, transform=None):
    """Bold lower-case panel label (8 pt)."""
    ax.text(x, y, letter, transform=transform or ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")
