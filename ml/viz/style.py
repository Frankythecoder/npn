"""Shared chrome for every figure.

Kept in one place so the metric plots in figures.py and the structure plots in
structure.py cannot drift into looking like two different reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

# Training runs headless -- in CI, in a container, over SSH. Selecting the Agg
# backend before pyplot is imported is what stops matplotlib looking for a
# display and failing at import time. Every viz module imports this one first.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402

# The console's palette, so a figure in a slide deck and the dashboard behind it
# do not disagree about which colour means "flagged".
FLAG = "#e5484d"
CLEAR = "#2ec4b6"
NEUTRAL = "#8b949e"
ACCENT = "#f5a524"
GRID = "#d0d7de"

DPI = 130


def style(ax: Any, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """One place for the chrome, so a dozen panels cannot drift apart."""
    if title:
        ax.set_title(title, fontsize=9, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def grid(n: int, ncols: int = 4, size: tuple[float, float] = (3.4, 2.6)):
    """A figure sized to hold `n` panels, with the unused ones removed."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * size[0], nrows * size[1]), squeeze=False
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.remove()
    return fig, flat[:n]


def save(fig: Any, dest: Path, name: str) -> Path:
    path = Path(dest) / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def scatter_flags(ax: Any, points: np.ndarray, flags: np.ndarray, size: float = 4.0) -> None:
    """Clear rows behind, flagged rows in front — the ordering matters, because
    5% of points drawn first would be buried under the other 95%."""
    flagged = np.asarray(flags).astype(bool)
    ax.scatter(
        points[~flagged, 0], points[~flagged, 1],
        s=size, c=CLEAR, alpha=0.35, linewidths=0,
    )
    ax.scatter(
        points[flagged, 0], points[flagged, 1],
        s=size * 2.2, c=FLAG, alpha=0.9, linewidths=0,
    )
