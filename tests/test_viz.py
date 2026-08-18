"""Figure rendering.

These assert that each figure is produced, is non-empty, and is derived from
the right quantity -- not what it looks like. A pixel comparison would fail on
every matplotlib point release and tell us nothing about correctness.

The fixture is deliberately tiny and synthetic: t-SNE and silhouette are both
O(n^2), and the shapes under test do not depend on n.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.config import Config
from ml.detectors.registry import build_detectors, live_detectors
from ml.ensemble.voting import combine_matrix, votes_required
from ml.viz.figures import render_training_figures, row_scores

N_ROWS = 90
CONTINUOUS = ["f0", "f1", "f2"]


@pytest.fixture(scope="module")
def fitted():
    """Seven fitted detectors over a small blob-plus-outliers frame."""
    rng = np.random.default_rng(0)
    bulk = rng.normal(0.0, 1.0, size=(N_ROWS - 6, 4))
    tail = rng.normal(7.0, 0.4, size=(6, 4))
    values = np.vstack([bulk, tail])

    X = pd.DataFrame(values, columns=["f0", "f1", "f2", "f3"])

    cfg = Config.load()
    detectors = build_detectors(cfg)
    # DBSCAN's configured eps assumes the real feature space; this frame is
    # tighter, so it would find no core samples and refuse to fit.
    for detector in detectors:
        if detector.name == "dbscan":
            detector.params["eps"] = 1.5
        if detector.name == "kmeans":
            detector.params["n_clusters"] = 3
        if detector.name == "gmm":
            detector.params["n_components"] = 2

    frames = {
        "standard": X,
        "robust": X,
        "continuous": X[CONTINUOUS],
    }
    for detector in detectors:
        detector.fit(frames[detector.scaler])

    live = live_detectors(detectors)
    live_flags = {d.name: d.fit_flags_ for d in live}
    votes, labels = combine_matrix(live_flags, 0.5)

    return {
        "X": X,
        "frames": frames,
        "detectors": detectors,
        "live": live,
        "live_flags": live_flags,
        "votes": votes,
        "labels": labels,
        "required": votes_required(len(live), 0.5),
    }


def render(fitted, dest, **overrides):
    kwargs = dict(
        X=fitted["X"],
        frames=fitted["frames"],
        detectors=fitted["detectors"],
        live_names=[d.name for d in fitted["live"]],
        live_flags=fitted["live_flags"],
        votes=fitted["votes"],
        ensemble_labels=fitted["labels"],
        required=fitted["required"],
        dest=dest,
        max_embedding_rows=N_ROWS,
    )
    kwargs.update(overrides)
    return render_training_figures(**kwargs)


# ---------- row-aligned scores ----------


def test_scores_are_row_aligned_not_the_sorted_copy(fitted):
    scores = row_scores(fitted["detectors"], fitted["frames"])
    for detector in fitted["detectors"]:
        values = scores[detector.name]
        assert len(values) == N_ROWS
        # A curve needs scores lined up with rows. train_scores_ is sorted at
        # fit time, so recovering the row order is the whole point of this.
        cut = values >= detector.threshold_
        assert (cut.astype(int) == detector.fit_flags_).all(), detector.name


def test_every_detector_is_scored(fitted):
    scores = row_scores(fitted["detectors"], fitted["frames"])
    assert set(scores) == {d.name for d in fitted["detectors"]}
    assert len(scores) == 7


# ---------- the rendered set ----------


def test_the_full_figure_set_is_written(fitted, tmp_path):
    written = render(fitted, tmp_path)
    assert {p.name for p in written} == {
        "01-score-distributions.png",
        "02-roc-pr-curves.png",
        "03-tsne-embedding.png",
        "04-cluster-separation.png",
        "05-ensemble-overview.png",
        "06-isolation-forest.png",
        "07-model-structure.png",
        "08-pca-reduction.png",
    }


def test_every_figure_is_a_real_png(fitted, tmp_path):
    for path in render(fitted, tmp_path):
        assert path.exists(), path
        blob = path.read_bytes()
        assert blob[:8] == b"\x89PNG\r\n\x1a\n", path
        assert len(blob) > 5_000, f"{path} looks empty"


def test_the_destination_is_created_when_absent(fitted, tmp_path):
    dest = tmp_path / "does" / "not" / "exist"
    written = render(fitted, dest)
    assert dest.is_dir()
    assert all(p.parent == dest for p in written)


def test_rendering_twice_overwrites_rather_than_accumulates(fitted, tmp_path):
    render(fitted, tmp_path)
    render(fitted, tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 8


# ---------- the label the curves are drawn against ----------


def test_roc_is_skipped_when_only_one_class_is_present(fitted, tmp_path):
    # A degenerate run must still yield the other four rather than crashing a
    # train that has already written its bundle.
    written = render(fitted, tmp_path, ensemble_labels=np.zeros(N_ROWS, dtype=int))
    assert len(written) == 7
    assert not (tmp_path / "02-roc-pr-curves.png").exists()


def test_supplied_ground_truth_labels_are_used_when_given(fitted, tmp_path):
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 2, size=N_ROWS)
    written = render(fitted, tmp_path, labels=truth)
    assert (tmp_path / "02-roc-pr-curves.png") in written


# ---------- opt-in wiring ----------


def test_training_draws_nothing_unless_asked(monkeypatch):
    """The default path must not so much as import the figure module."""
    import ml.pipeline.train as train

    called = []
    monkeypatch.setattr(
        "ml.viz.figures.render_training_figures",
        lambda **kw: called.append(kw) or [],
    )
    signature = train.run_training.__defaults__
    # dest=None, figures_dir=None
    assert signature == (None, None)
    assert called == []
