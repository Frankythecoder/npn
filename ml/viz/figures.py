"""Diagnostic figures for a training run, covering all seven detectors.

Rendered on request only -- `run_training` takes a `figures_dir` and does
nothing here unless it is given one, so an ordinary retrain costs exactly what
it did before. The work that is expensive (one PCA, one t-SNE embedding, one
scoring pass per detector, a silhouette matrix) is done once and shared across
every figure that needs it.

This module draws how the detectors BEHAVED -- scores, curves, votes. What they
each actually fitted is drawn by ml.viz.structure, and the PCA both are drawn
through is ml.viz.reduction.

A NOTE ON ROC AND PR
--------------------
The dataset carries no fraud label. Nothing here is measured against ground
truth, and none of these curves says how accurate detection is -- there is no
such number to compute. What they measure is FIDELITY: how closely each
detector's continuous score reproduces the ensemble's consensus verdict, the
same quantity, and the same reasoning, as the surrogate's held-out AUC in
ml/explain/surrogate.py. Reading them as accuracy is the category error that
module's docstring warns about.

Pass `labels=` if real annotations ever exist; then the curves mean what they
usually mean, and the figures say so.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_samples,
)

from ml.viz.reduction import Reduction, fit_reduction, tsne_input
from ml.viz.structure import (
    plot_isolation_forest,
    plot_model_structure,
    plot_reduction,
)
from ml.viz.style import (
    CLEAR,
    FLAG,
    GRID,
    NEUTRAL,
    grid as _grid,
    plt,
    save as _save,
    scatter_flags,
    style as _style,
)

# t-SNE is O(n^2) in memory and time. Above this many rows the embedding is
# built from a stratified sample instead -- the shape of the manifold is what
# the plot is for, and it survives sampling; a twenty-minute retrain does not.
MAX_EMBEDDING_ROWS = 2000

# Detectors whose fitted model partitions the data, and so have a cluster
# structure worth showing separately from their anomaly score.
CLUSTERED = ("kmeans", "gmm", "dbscan")

DPI = 130


def row_scores(detectors: Sequence[Any], frames: dict[str, pd.DataFrame]) -> dict:
    """Row-aligned training scores for every detector.

    `train_scores_` is sorted at fit time, so it cannot be lined up against a
    label vector; `fit_flags_` is aligned but already thresholded. Neither can
    draw a curve. `_training_scores` is the hook `fit` itself used, which is
    exactly the quantity the threshold was cut from -- and it is the reason LOF
    overrides that hook rather than `_score`.
    """
    scores = {}
    for detector in detectors:
        values = frames[detector.scaler].to_numpy(dtype=float)
        scores[detector.name] = np.asarray(
            detector._training_scores(values), dtype=float
        )
    return scores


def plot_score_distributions(
    detectors: Sequence[Any], scores: dict, dest: Path
) -> Path:
    """Per-detector score histogram with the fitted threshold drawn on it.

    The first thing to look at when a rate-table entry surprises you: a
    threshold sitting on a fat continuous shoulder means the cut is arbitrary
    and small hyperparameter changes will move the flag set a long way, whereas
    one sitting past a clear gap means it is not.
    """
    fig, axes = _grid(len(detectors))

    for ax, detector in zip(axes, detectors):
        values = scores[detector.name]
        threshold = float(detector.threshold_)
        flagged = values >= threshold

        ax.hist(values[~flagged], bins=60, color=CLEAR, alpha=0.75, label="clear")
        ax.hist(values[flagged], bins=25, color=FLAG, alpha=0.9, label="flagged")
        ax.axvline(threshold, color=NEUTRAL, linewidth=1.2, linestyle="--")
        ax.set_yscale("log")
        _style(
            ax,
            f"{detector.name}  ({flagged.mean():.2%} flagged)",
            "anomaly score",
            "rows (log)",
        )
        ax.legend(fontsize=6.5, frameon=False)

    fig.suptitle(
        "Anomaly score distributions — dashed line is the fitted threshold",
        fontsize=11,
        y=1.005,
    )
    return _save(fig, dest, "01-score-distributions.png")


def plot_roc_pr(
    detectors: Sequence[Any],
    scores: dict,
    labels: np.ndarray,
    label_name: str,
    surrogate_auc: float | None,
    dest: Path,
) -> Path:
    """ROC and precision-recall for every detector against `labels`.

    Both panels are drawn because they answer different questions on a 5%
    positive rate: ROC stays flattering when positives are rare, while PR
    collapses honestly. The dashed baselines are what a coin flip scores.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    positive_rate = float(np.mean(labels))
    colours = plt.cm.turbo(np.linspace(0.08, 0.92, len(detectors)))

    for detector, colour in zip(detectors, colours):
        values = scores[detector.name]
        fpr, tpr, _ = roc_curve(labels, values)
        precision, recall, _ = precision_recall_curve(labels, values)
        axes[0].plot(
            fpr,
            tpr,
            color=colour,
            linewidth=1.5,
            label=f"{detector.name}  {roc_auc_score(labels, values):.3f}",
        )
        axes[1].plot(
            recall,
            precision,
            color=colour,
            linewidth=1.5,
            label=f"{detector.name}  {average_precision_score(labels, values):.3f}",
        )

    axes[0].plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=1)
    axes[1].axhline(positive_rate, color=NEUTRAL, linestyle="--", linewidth=1)

    _style(axes[0], "ROC (AUC in legend)", "false positive rate", "true positive rate")
    _style(axes[1], "Precision-recall (AP in legend)", "recall", "precision")
    for ax in axes:
        ax.legend(fontsize=7.5, frameon=False, loc="lower right" if ax is axes[0] else "upper right")

    subtitle = f"scored against {label_name} ({positive_rate:.2%} positive)"
    if surrogate_auc is not None:
        subtitle += f" · surrogate held-out AUC {surrogate_auc:.4f}"
    fig.suptitle(f"Detector fidelity — {subtitle}", fontsize=11, y=1.0)
    return _save(fig, dest, "02-roc-pr-curves.png")


def _embedding(
    reduction: Reduction, labels: np.ndarray, random_state: int, max_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """A 2-D t-SNE of the PCA-reduced matrix, plus the row indices it covers.

    Computed once and handed to every panel that needs it. Recomputing per
    detector would be seven times the cost for seven views of one manifold --
    and worse, seven *different* manifolds, which cannot be compared.

    t-SNE runs on the leading components rather than the raw features: cheaper,
    and it stops the embedding spending its budget modelling noise directions.
    """
    source = tsne_input(reduction)
    n = len(source)
    index = np.arange(n)

    if n > max_rows:
        # Stratified, so the 5% that are flagged are not sampled away.
        rng = np.random.default_rng(random_state)
        keep = []
        for value in np.unique(labels):
            members = index[labels == value]
            take = max(1, int(round(len(members) * max_rows / n)))
            keep.append(rng.choice(members, size=min(take, len(members)), replace=False))
        index = np.sort(np.concatenate(keep))

    values = source[index]
    # Perplexity must stay below the sample count; sklearn raises otherwise.
    perplexity = float(min(30, max(5, (len(index) - 1) // 3)))
    embedded = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=random_state,
    ).fit_transform(values)
    return embedded, index


def plot_embedding(
    detectors: Sequence[Any],
    reduction: Reduction,
    votes: np.ndarray,
    labels: np.ndarray,
    random_state: int,
    dest: Path,
    max_rows: int = MAX_EMBEDDING_ROWS,
) -> Path:
    """One shared t-SNE, overlaid with each detector's flags in turn.

    Because every panel is the same embedding, the panels are comparable: two
    detectors firing on the same lobe are finding the same structure, and one
    firing on scattered singletons is finding something else.
    """
    embedded, index = _embedding(reduction, labels, random_state, max_rows)
    fig, axes = _grid(len(detectors) + 1, ncols=3, size=(3.7, 3.2))

    for ax, detector in zip(axes, detectors):
        flags = np.asarray(detector.fit_flags_, dtype=int)[index]
        ax.scatter(
            embedded[flags == 0, 0], embedded[flags == 0, 1],
            s=4, c=CLEAR, alpha=0.35, linewidths=0,
        )
        ax.scatter(
            embedded[flags == 1, 0], embedded[flags == 1, 1],
            s=9, c=FLAG, alpha=0.9, linewidths=0,
        )
        _style(ax, detector.name)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[-1]
    scatter = ax.scatter(
        embedded[:, 0], embedded[:, 1],
        s=6, c=votes[index], cmap="turbo", alpha=0.85, linewidths=0,
    )
    _style(ax, "ensemble votes")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scatter, ax=ax, fraction=0.046).ax.tick_params(labelsize=7)

    total = len(reduction.projection)
    shown = "all rows" if len(index) == total else f"{len(index):,} of {total:,} rows (stratified)"
    fig.suptitle(
        f"t-SNE over {reduction.n_components} PCA components — one embedding, {shown}",
        fontsize=11, y=1.005,
    )
    return _save(fig, dest, "03-tsne-embedding.png")


def plot_cluster_separation(
    detectors: Sequence[Any],
    reduction: Reduction,
    frames: dict[str, pd.DataFrame],
    labels: np.ndarray,
    random_state: int,
    dest: Path,
    max_rows: int = MAX_EMBEDDING_ROWS,
) -> Path:
    """Silhouettes for the three detectors that actually partition the data.

    KMeans, GMM and DBSCAN reach a verdict via cluster membership, so how well
    separated those clusters are is a tuning signal their anomaly score hides:
    a silhouette hugging zero means k or eps is wrong, whatever the flag rate
    happens to look like.
    """
    by_name = {d.name: d for d in detectors}
    present = [name for name in CLUSTERED if name in by_name]

    n = len(reduction.projection)
    index = np.arange(n)
    if n > max_rows:
        rng = np.random.default_rng(random_state)
        index = np.sort(rng.choice(index, size=max_rows, replace=False))

    fig, axes = _grid(len(present) + 1, ncols=2, size=(5.4, 3.6))

    for ax, name in zip(axes, present):
        detector = by_name[name]
        values = frames[detector.scaler].to_numpy(dtype=float)
        model = detector._model

        if name == "dbscan":
            # DBSCAN cannot predict; its fitted labels_ are already row-aligned.
            assignments = np.asarray(model.labels_, dtype=int)
        else:
            assignments = np.asarray(model.predict(values), dtype=int)

        sub_values = values[index]
        sub_assign = assignments[index]
        unique = np.unique(sub_assign)

        if len(unique) < 2:
            _style(ax, f"{name} — one cluster, no silhouette")
            continue

        silhouette = silhouette_samples(sub_values, sub_assign)
        offset = 0
        for cluster in unique:
            band = np.sort(silhouette[sub_assign == cluster])
            colour = NEUTRAL if cluster == -1 else plt.cm.turbo(
                (cluster + 1) / (len(unique) + 1)
            )
            ax.fill_betweenx(
                np.arange(offset, offset + len(band)), 0, band,
                color=colour, linewidth=0,
                label=("noise" if cluster == -1 else f"c{cluster}"),
            )
            offset += len(band) + 8

        ax.axvline(float(silhouette.mean()), color=FLAG, linestyle="--", linewidth=1.2)
        _style(ax, f"{name} — mean silhouette {silhouette.mean():.3f}", "silhouette", "rows by cluster")
        ax.set_yticks([])
        ax.legend(fontsize=6, frameon=False, ncol=2)

    # The shared PCA plane alongside, so the silhouettes can be read against
    # the geometry that produced them. Refitting a second PCA here would put
    # the panels on two different planes.
    ax = axes[-1]
    scatter_flags(ax, reduction.projection, labels, size=4)
    explained = reduction.explained
    _style(
        ax,
        f"PCA plane — consensus flags ({explained[:2].sum():.0%} of variance)",
        f"PC1 ({explained[0]:.0%})",
        f"PC2 ({explained[1]:.0%})" if len(explained) > 1 else "PC2",
    )

    fig.suptitle("Cluster separation", fontsize=11, y=1.005)
    return _save(fig, dest, "04-cluster-separation.png")


def plot_ensemble_overview(
    live_names: Sequence[str],
    live_flags: dict,
    votes: np.ndarray,
    required: int,
    dest: Path,
) -> Path:
    """How the vote is distributed, what a different quorum would cost, and
    which pairs of detectors are actually redundant."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    n = len(votes)
    total = len(live_names)

    counts = [int((votes == v).sum()) for v in range(total + 1)]
    bars = axes[0].bar(range(total + 1), counts, color=NEUTRAL, width=0.65)
    for v in range(required, total + 1):
        bars[v].set_color(FLAG)
    axes[0].set_yscale("log")
    _style(axes[0], f"Vote histogram (quorum {required}/{total})", "votes cast", "rows (log)")

    sweep = [int((votes >= r).sum()) for r in range(1, total + 1)]
    axes[1].plot(range(1, total + 1), [c / n for c in sweep], marker="o", color=FLAG, linewidth=1.5)
    axes[1].axvline(required, color=NEUTRAL, linestyle="--", linewidth=1)
    axes[1].set_xticks(range(1, total + 1))
    _style(axes[1], "Flag rate by quorum", "votes required", "share of rows flagged")

    size = len(live_names)
    matrix = np.ones((size, size))
    for i, a in enumerate(live_names):
        for j, b in enumerate(live_names):
            if i == j:
                continue
            fa, fb = live_flags[a] == 1, live_flags[b] == 1
            union = int((fa | fb).sum())
            matrix[i, j] = float((fa & fb).sum() / union) if union else 0.0

    image = axes[2].imshow(matrix, cmap="magma", vmin=0, vmax=1)
    axes[2].set_xticks(range(size), live_names, rotation=45, ha="right", fontsize=7)
    axes[2].set_yticks(range(size), live_names, fontsize=7)
    for i in range(size):
        for j in range(size):
            axes[2].text(
                j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if matrix[i, j] < 0.6 else "black",
            )
    axes[2].set_title("Pairwise agreement (Jaccard)", fontsize=9, pad=6)
    axes[2].grid(False)
    fig.colorbar(image, ax=axes[2], fraction=0.046).ax.tick_params(labelsize=7)

    fig.suptitle("Ensemble behaviour", fontsize=11, y=1.02)
    return _save(fig, dest, "05-ensemble-overview.png")


def render_training_figures(
    *,
    X: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    detectors: Sequence[Any],
    live_names: Sequence[str],
    live_flags: dict,
    votes: np.ndarray,
    ensemble_labels: np.ndarray,
    required: int,
    dest: str | Path,
    labels: np.ndarray | None = None,
    surrogate_auc: float | None = None,
    random_state: int = 42,
    max_embedding_rows: int = MAX_EMBEDDING_ROWS,
    pca_components: float | int = 0.95,
) -> list[Path]:
    """Render every figure into `dest`, returning the paths written.

    `labels` defaults to the ensemble consensus -- see this module's docstring
    for why that makes the curves a fidelity measure and not an accuracy one.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = np.asarray(ensemble_labels, dtype=int)
        label_name = "the ensemble consensus (no ground truth exists)"
    else:
        labels = np.asarray(labels, dtype=int)
        label_name = "the supplied ground-truth label"

    # One PCA, fitted once, over the STANDARD-SCALED frame rather than raw X.
    # That matters twice over. Unscaled, PC1 would be whichever raw column has
    # the largest units (AccountBalance) and the plane would describe nothing.
    # And because five of the seven detectors are fitted on this same frame,
    # points lifted off the plane by Reduction.from_plane are valid inputs to
    # them -- which is what lets the one-class SVM's real boundary be drawn
    # instead of a second model refitted in 2-D.
    reduction = fit_reduction(frames.get("standard", X), pca_components, random_state)
    scores = row_scores(detectors, frames)
    by_name = {d.name: d for d in detectors}

    written = [
        plot_score_distributions(detectors, scores, dest),
        plot_embedding(
            detectors, reduction, votes, labels, random_state, dest, max_embedding_rows
        ),
        plot_cluster_separation(
            detectors, reduction, frames, labels, random_state, dest, max_embedding_rows
        ),
        plot_ensemble_overview(list(live_names), live_flags, votes, required, dest),
        plot_model_structure(detectors, frames, reduction, dest, random_state),
        plot_reduction(reduction, labels, dest),
    ]

    if "isolation_forest" in by_name:
        detector = by_name["isolation_forest"]
        written.append(
            plot_isolation_forest(detector, frames[detector.scaler], dest)
        )

    # A curve needs both classes. A degenerate run that flagged everything or
    # nothing still gets the other figures rather than crashing the train.
    if 0 < int(labels.sum()) < len(labels):
        written.append(
            plot_roc_pr(detectors, scores, labels, label_name, surrogate_auc, dest)
        )

    return sorted(written)
