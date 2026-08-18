"""Structure plots: what each unsupervised model actually built.

The metric figures in figures.py say how a detector *behaved*. These say what it
*is* -- the trees an isolation forest grew, the neighbourhood graph LOF reasons
over, the boundary a one-class SVM placed, the reachability DBSCAN's eps buys,
the ellipses GMM and MCD fitted, the elbow that justifies k. That is the view
you need to tune a hyperparameter, and it is the view a stakeholder needs to
believe the model is doing something and not guessing.

Everything two-dimensional is drawn on the shared PCA plane from
ml.viz.reduction, so panels are mutually comparable, and so a fitted Gaussian
can be projected exactly rather than approximated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import plot_tree

from ml.viz.reduction import Reduction
from ml.viz.style import (
    ACCENT,
    CLEAR,
    FLAG,
    NEUTRAL,
    grid,
    plt,
    save,
    scatter_flags,
    style,
)

# A neighbourhood graph is unreadable past a few hundred nodes -- 2,000 points
# at k=20 is 40,000 segments of solid ink. The graph's shape reads fine from a
# sample; the point is the connectivity pattern, not every edge.
GRAPH_NODES = 260

# Grid resolution for a decision surface. 60x60 is 3,600 lifted rows to score,
# which is the expensive part, and is enough for a smooth contour.
SURFACE_STEPS = 60

# Range of k swept for the KMeans elbow.
ELBOW_RANGE = range(2, 13)


def _subsample(n: int, size: int, random_state: int) -> np.ndarray:
    if n <= size:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=size, replace=False))


def _plane_grid(reduction: Reduction, points: np.ndarray, steps: int = SURFACE_STEPS):
    """A mesh spanning the drawn points, plus its lift into feature space."""
    pad_x = 0.05 * np.ptp(points[:, 0])
    pad_y = 0.05 * np.ptp(points[:, 1])
    xs = np.linspace(points[:, 0].min() - pad_x, points[:, 0].max() + pad_x, steps)
    ys = np.linspace(points[:, 1].min() - pad_y, points[:, 1].max() + pad_y, steps)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    plane = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    return mesh_x, mesh_y, reduction.from_plane(plane)


def _ellipse(ax: Any, mean: np.ndarray, cov: np.ndarray, colour: str, sigma: float = 2.0):
    """Draw a covariance ellipse at `sigma` standard deviations."""
    values, vectors = np.linalg.eigh(cov)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    width, height = 2.0 * sigma * np.sqrt(np.maximum(values, 1e-12))
    ax.add_patch(
        Ellipse(
            mean, width, height, angle=angle,
            facecolor="none", edgecolor=colour, linewidth=1.4, alpha=0.9,
        )
    )


# ---------------------------------------------------------------- isolation forest


def plot_isolation_forest(detector: Any, frame: pd.DataFrame, dest: Path) -> Path:
    """The forest itself: one tree drawn, and the shape of all the others.

    Isolation Forest's whole premise is that anomalies fall out of a random
    split in fewer steps than ordinary rows. That claim lives in the trees, so
    this draws one, then shows the depth and leaf-count spread across the rest.
    A forest whose trees all bottom out at the same depth has stopped isolating
    anything and is only memorising the sample.
    """
    forest = detector._model
    estimators = list(forest.estimators_)
    values = frame.to_numpy(dtype=float)

    fig = plt.figure(figsize=(15.0, 8.6))
    spec = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0])

    tree_ax = fig.add_subplot(spec[0, :])
    plot_tree(
        estimators[0],
        max_depth=3,
        feature_names=[str(c) for c in frame.columns],
        filled=False,
        impurity=False,
        proportion=True,
        rounded=True,
        fontsize=6,
        ax=tree_ax,
    )
    tree_ax.set_title(
        f"Tree 1 of {len(estimators)} — first 3 levels of {estimators[0].get_depth()}",
        fontsize=9,
    )

    depths = np.array([e.get_depth() for e in estimators])
    leaves = np.array([e.get_n_leaves() for e in estimators])

    ax = fig.add_subplot(spec[1, 0])
    ax.hist(depths, bins=max(4, len(np.unique(depths))), color=NEUTRAL)
    ax.axvline(depths.mean(), color=FLAG, linestyle="--", linewidth=1.2)
    style(ax, f"Tree depth (mean {depths.mean():.1f})", "depth", "trees")

    ax = fig.add_subplot(spec[1, 1])
    ax.hist(leaves, bins=25, color=NEUTRAL)
    ax.axvline(leaves.mean(), color=FLAG, linestyle="--", linewidth=1.2)
    style(ax, f"Leaves per tree (mean {leaves.mean():.0f})", "leaves", "trees")

    # Mean isolation depth per row: the quantity the score is a monotone
    # transform of, and the one the forest's premise is actually about.
    ax = fig.add_subplot(spec[1, 2])
    depth_per_row = np.zeros(len(values))
    for estimator in estimators:
        # decision_path is sparse; asarray().ravel() flattens it without
        # relying on np.matrix, which is on its way out.
        steps = np.asarray(estimator.decision_path(values).sum(axis=1)).ravel()
        depth_per_row += steps - 1.0
    depth_per_row /= len(estimators)
    flags = np.asarray(detector.fit_flags_, dtype=bool)
    ax.hist(depth_per_row[~flags], bins=50, color=CLEAR, alpha=0.8, label="clear")
    ax.hist(depth_per_row[flags], bins=25, color=FLAG, alpha=0.9, label="flagged")
    style(ax, "Mean isolation depth per row", "average path length", "rows")
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle("Isolation Forest — structure", fontsize=11, y=1.0)
    return save(fig, dest, "06-isolation-forest.png")


# ---------------------------------------------------------------- per-model panels


def _panel_lof(ax, detector, frame, reduction, index, random_state):
    """The k-nearest-neighbour graph LOF's density estimate is computed over."""
    values = frame.to_numpy(dtype=float)[index]
    points = reduction.projection[index]
    k = int(detector.params["n_neighbors"])

    neighbours = NearestNeighbors(n_neighbors=min(k, len(index) - 1)).fit(values)
    _, adjacency = neighbours.kneighbors(values)

    segments = [
        [points[i], points[j]] for i, row in enumerate(adjacency) for j in row[1:]
    ]
    ax.add_collection(
        LineCollection(segments, colors=NEUTRAL, linewidths=0.18, alpha=0.35)
    )
    scatter_flags(ax, points, np.asarray(detector.fit_flags_)[index], size=7)
    style(ax, f"LOF — k={k} neighbourhood graph", "PC1", "PC2")


def _panel_one_class_svm(ax, detector, frame, reduction, index, random_state):
    """The fitted boundary, sliced through the PCA plane."""
    points = reduction.projection
    mesh_x, mesh_y, lifted = _plane_grid(reduction, points)
    surface = detector._model.decision_function(lifted).reshape(mesh_x.shape)

    ax.contourf(mesh_x, mesh_y, surface, levels=14, cmap="RdYlGn", alpha=0.35)
    ax.contour(mesh_x, mesh_y, surface, levels=[0], colors=[FLAG], linewidths=1.6)
    scatter_flags(ax, points, detector.fit_flags_, size=3)
    n_support = int(detector._model.support_vectors_.shape[0])
    style(ax, f"One-Class SVM — boundary, {n_support} support vectors", "PC1", "PC2")


def _panel_dbscan(ax, detector, frame, reduction, index, random_state):
    """The k-distance graph: the standard way eps is chosen, and a direct
    reading of whether the configured value sits at the knee or past it."""
    values = frame.to_numpy(dtype=float)
    k = int(detector.params["min_samples"])
    distances, _ = NearestNeighbors(n_neighbors=k).fit(values).kneighbors(values)
    curve = np.sort(distances[:, -1])[::-1]

    ax.plot(np.arange(len(curve)), curve, color=NEUTRAL, linewidth=1.4)
    eps = float(detector.params["eps"])
    ax.axhline(eps, color=FLAG, linestyle="--", linewidth=1.3)
    ax.annotate(
        f"eps = {eps:g}\nnative noise {detector.native_noise_rate_:.2%}",
        xy=(0.55, 0.72), xycoords="axes fraction", fontsize=7.5, color=FLAG,
    )
    style(ax, f"DBSCAN — {k}-distance graph ({detector.n_clusters_} clusters)",
          "rows, sorted by distance", f"distance to {k}th neighbour")


def _panel_mcd(ax, detector, frame, reduction, index, random_state):
    """The robust ellipse, on the two continuous directions it was fitted over."""
    values = frame.to_numpy(dtype=float)
    model = detector._model
    location = np.asarray(model.location_, dtype=float)
    covariance = np.asarray(model.covariance_, dtype=float)

    ax.scatter(values[:, 0], values[:, 1], s=3, c=CLEAR, alpha=0.3, linewidths=0)
    flags = np.asarray(detector.fit_flags_, dtype=bool)
    ax.scatter(values[flags, 0], values[flags, 1], s=8, c=FLAG, alpha=0.9, linewidths=0)
    for sigma, colour in ((1.0, ACCENT), (2.0, FLAG), (3.0, NEUTRAL)):
        _ellipse(ax, location[:2], covariance[:2, :2], colour, sigma=sigma)
    ax.scatter(*location[:2], marker="x", c="black", s=40, linewidths=1.4)
    style(ax, "MCD — robust covariance (1/2/3σ)",
          str(frame.columns[0]), str(frame.columns[1]))


def _panel_gmm(ax, detector, frame, reduction, index, random_state):
    """Component ellipses, projected exactly onto the plane.

    PCA is linear, so a component's mean maps through `transform` and its
    covariance as `W S W'`. These are the fitted Gaussians, not a refit.
    """
    model = detector._model
    points = reduction.projection
    basis = reduction.pca.components_[:2]  # (2, n_features)

    scatter_flags(ax, points, detector.fit_flags_, size=3)

    means = reduction.to_plane(np.asarray(model.means_, dtype=float))
    colours = plt.cm.turbo(np.linspace(0.1, 0.9, len(means)))
    for i, (mean, colour) in enumerate(zip(means, colours)):
        cov = model.covariances_[i]
        if cov.ndim == 0:            # 'spherical'
            cov = np.eye(basis.shape[1]) * float(cov)
        elif cov.ndim == 1:          # 'diag'
            cov = np.diag(cov)
        projected = basis @ cov @ basis.T
        _ellipse(ax, mean, projected, colour, sigma=2.0)

    style(ax, f"GMM — {len(means)} components ({model.covariance_type}), 2σ",
          "PC1", "PC2")


def _panel_kmeans(ax, detector, frame, reduction, index, random_state):
    """The elbow: inertia against k, with the configured k marked.

    Refits are over the reduced matrix rather than raw features -- an order of
    magnitude cheaper, and the elbow's position is what is being read, not its
    absolute height.
    """
    reduced = reduction.components
    inertias = []
    for k in ELBOW_RANGE:
        model = KMeans(n_clusters=k, n_init=4, random_state=random_state).fit(reduced)
        inertias.append(float(model.inertia_))

    ax.plot(list(ELBOW_RANGE), inertias, marker="o", color=NEUTRAL, linewidth=1.4)
    chosen = int(detector.params["n_clusters"])
    ax.axvline(chosen, color=FLAG, linestyle="--", linewidth=1.3)
    ax.annotate(f"k = {chosen}", xy=(0.62, 0.78), xycoords="axes fraction",
                fontsize=8, color=FLAG)
    ax.set_xticks(list(ELBOW_RANGE))
    style(ax, "K-Means — elbow over the reduced matrix", "k", "inertia")


PANELS = {
    "lof": _panel_lof,
    "one_class_svm": _panel_one_class_svm,
    "dbscan": _panel_dbscan,
    "mcd": _panel_mcd,
    "gmm": _panel_gmm,
    "kmeans": _panel_kmeans,
}


def plot_model_structure(
    detectors: Sequence[Any],
    frames: dict[str, pd.DataFrame],
    reduction: Reduction,
    dest: Path,
    random_state: int = 42,
) -> Path:
    """One structural panel per model that has one, on a single sheet."""
    by_name = {d.name: d for d in detectors}
    present = [name for name in PANELS if name in by_name]
    index = _subsample(len(reduction.projection), GRAPH_NODES, random_state)

    fig, axes = grid(len(present), ncols=3, size=(5.0, 4.0))

    for ax, name in zip(axes, present):
        detector = by_name[name]
        try:
            PANELS[name](
                ax, detector, frames[detector.scaler], reduction, index, random_state
            )
        except Exception as exc:  # noqa: BLE001
            # One unhappy panel must not cost the other five, nor the run.
            ax.text(0.5, 0.5, f"{name}\nnot drawn: {exc}", ha="center", va="center",
                    fontsize=7, transform=ax.transAxes, color=NEUTRAL)
            style(ax, name)

    fig.suptitle("Model structure — what each unsupervised model fitted",
                 fontsize=11, y=1.005)
    return save(fig, dest, "07-model-structure.png")


# ---------------------------------------------------------------- the reduction itself


def plot_reduction(
    reduction: Reduction, labels: np.ndarray, dest: Path
) -> Path:
    """PCA as the reduction step: how much it keeps, the plane it yields, and
    what it fails to reconstruct.

    The third panel is the signal that used to be the eighth detector. It is
    still worth looking at -- a row PCA cannot rebuild is genuinely unusual --
    it just no longer casts a vote.
    """
    fig, axes = grid(3, ncols=3, size=(5.0, 4.0))
    explained = reduction.explained
    cumulative = np.cumsum(explained)

    ax = axes[0]
    order = np.arange(1, len(explained) + 1)
    ax.bar(order, explained, color=NEUTRAL, width=0.7)
    twin = ax.twinx()
    twin.plot(order, cumulative, color=FLAG, marker="o", markersize=3, linewidth=1.4)
    twin.set_ylim(0, 1.02)
    twin.tick_params(labelsize=7)
    twin.set_ylabel("cumulative", fontsize=8, color=FLAG)
    style(ax, f"Scree — {reduction.n_components} components keep "
              f"{cumulative[-1]:.1%}", "component", "explained variance")

    ax = axes[1]
    scatter_flags(ax, reduction.projection, labels, size=4)
    style(ax, "The plane every structure plot is drawn on",
          f"PC1 ({explained[0]:.1%})",
          f"PC2 ({explained[1]:.1%})" if len(explained) > 1 else "PC2")

    ax = axes[2]
    error = reduction.reconstruction_error
    flagged = np.asarray(labels).astype(bool)
    ax.hist(error[~flagged], bins=60, color=CLEAR, alpha=0.8, label="clear")
    ax.hist(error[flagged], bins=30, color=FLAG, alpha=0.9, label="consensus-flagged")
    ax.set_yscale("log")
    style(ax, "Reconstruction error (a diagnostic, no longer a vote)",
          "mean squared error", "rows (log)")
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle("PCA — the dimensionality reduction behind every figure",
                 fontsize=11, y=1.005)
    return save(fig, dest, "08-pca-reduction.png")
