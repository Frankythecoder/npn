"""PCA: the dimensionality reduction every figure is drawn through.

PCA used to be an eighth detector, voting on reconstruction error. It is not a
detector here. It is the reduction step: it supplies the 2-D plane the structure
plots are drawn on, it pre-reduces the feature matrix before t-SNE, and its
reconstruction error survives as a diagnostic rather than as a vote.

Pre-reducing before t-SNE is not only cheaper. t-SNE on raw features spends its
budget modelling the noise directions too; running it on the components that
carry the variance is the standard recipe, and it is why sklearn's own default
init is "pca".

Being a linear map is what makes PCA the right choice for the structure plots:
a fitted Gaussian's mean and covariance can be pushed through it exactly
(`W S W'`), so GMM and MCD ellipses drawn on the projection are the real fitted
ellipses, not a redrawn approximation. No non-linear embedding can do that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Components handed to t-SNE. Past roughly this many the extra directions are
# noise as far as a 2-D embedding is concerned, and t-SNE's cost grows with them.
TSNE_INPUT_COMPONENTS = 10


@dataclass
class Reduction:
    """A fitted PCA and everything downstream needs from it."""

    pca: PCA
    components: np.ndarray  # (n_rows, n_components) full reduced matrix
    projection: np.ndarray  # (n_rows, 2) the plane figures are drawn on
    reconstruction_error: np.ndarray  # per-row mean squared error
    explained: np.ndarray  # per-component explained variance ratio

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    def to_plane(self, values: np.ndarray) -> np.ndarray:
        """Project feature-space rows onto the same 2-D plane."""
        return self.pca.transform(np.asarray(values, dtype=float))[:, :2]

    def from_plane(self, plane: np.ndarray) -> np.ndarray:
        """Lift 2-D plane points back into feature space.

        Used to evaluate a detector's real decision function over a grid: the
        grid is built in the plane, lifted here, and scored by the model as
        fitted. That draws the boundary the model actually has, sliced through
        the plane, rather than refitting a second model in 2-D and drawing that.
        """
        plane = np.asarray(plane, dtype=float)
        padded = np.zeros((len(plane), self.n_components), dtype=float)
        padded[:, :2] = plane
        return self.pca.inverse_transform(padded)


def fit_reduction(
    X: pd.DataFrame, n_components: float | int = 0.95, random_state: int = 42
) -> Reduction:
    """Fit the shared PCA over the feature matrix."""
    values = X.to_numpy(dtype=float)
    pca = PCA(n_components=n_components, random_state=random_state).fit(values)

    components = pca.transform(values)
    reconstructed = pca.inverse_transform(components)
    error = ((values - reconstructed) ** 2).mean(axis=1)

    # A single-component fit still has to yield a plane to draw on.
    projection = components[:, :2]
    if projection.shape[1] == 1:
        projection = np.column_stack([projection[:, 0], np.zeros(len(projection))])

    return Reduction(
        pca=pca,
        components=components,
        projection=projection,
        reconstruction_error=error,
        explained=pca.explained_variance_ratio_,
    )


def tsne_input(reduction: Reduction) -> np.ndarray:
    """The matrix t-SNE should embed: the leading components, not raw features."""
    return reduction.components[:, :TSNE_INPUT_COMPONENTS]
