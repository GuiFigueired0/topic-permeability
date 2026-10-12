"""Shared figure style and saving (PNG for slides, SVG for the paper)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as cfg

# Categorical slots in fixed order (validated CVD-safe palette); text stays in neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def use_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 2, "lines.markersize": 4, "legend.frameon": False,
    })


def blues_cmap() -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list("blues", BLUES)


def savefig(fig: plt.Figure, name: str) -> None:
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(cfg.RESULTS / f"{name}.{ext}")
