"""Shared settings: label names, class order, colours, plotting style."""
from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
RES = ROOT / "results"
SPACY_MODEL = ROOT / "models" / "lt_core_news_md"
SEED = 42

# Folder letter -> emotion (D=džiaugsmas, L=liūdesys, N=nustebimas, P=pyktis, Z=neutral)
LABEL_NAMES = {"Z": "neutral", "D": "joy", "N": "surprise", "L": "sadness", "P": "anger"}
ORDER = ["neutral", "joy", "surprise", "sadness", "anger"]
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "legend.frameon": False,
    })
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
