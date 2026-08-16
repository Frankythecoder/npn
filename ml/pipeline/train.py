"""Offline training run: fit everything, report, and write the artifact bundle."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.registry import build_detectors, live_detectors
from ml.ensemble.voting import combine_matrix, votes_required
from ml.explain.shap_explainer import build_explainer_state
from ml.explain.surrogate import train_surrogate
from ml.features.engineer import build_training_frame
from ml.storage.artifacts import ArtifactBundle, save_bundle


@dataclass
class TrainingReport:
    rate_table: dict[str, float]
    flag_counts: dict[str, int]
    agreement: dict[str, float]
    vote_histogram: dict[int, int]
    ensemble_rate: float
    ensemble_flagged: int
    threshold_sweep: dict[int, int]
    surrogate_auc: float
    surrogate_agreement: float
    dbscan_native_noise_rate: float
    n_rows: int
    warnings: list[str] = field(default_factory=list)


def _build_frames(X: pd.DataFrame, continuous_columns: list[str]) -> tuple[dict, dict]:
    """Return (scalers, scaled frames) keyed by the detector `scaler` attribute."""
    standard = StandardScaler().fit(X)
    robust = RobustScaler().fit(X)
    continuous = StandardScaler().fit(X[continuous_columns])

    scalers = {"standard": standard, "robust": robust, "continuous": continuous}
    frames = {
        "standard": pd.DataFrame(standard.transform(X), columns=X.columns),
        "robust": pd.DataFrame(robust.transform(X), columns=X.columns),
        "continuous": pd.DataFrame(
            continuous.transform(X[continuous_columns]), columns=continuous_columns
        ),
    }
    return scalers, frames


def run_training(cfg: Config, dest: str | Path | None = None) -> TrainingReport:
    """Fit every detector, build the ensemble and surrogate, and save the bundle."""
    dest = Path(dest) if dest is not None else Path(cfg.get("storage.local_dir"))

    raw = load_raw(cfg.get("data.csv_path"))
    X, feature_artifacts, profile_store = build_training_frame(raw)
    scalers, frames = _build_frames(X, feature_artifacts.continuous_columns)

    detectors = build_detectors(cfg)
    for detector in detectors:
        detector.fit(frames[detector.scaler])

    rate_table = {d.name: float(d.fit_flags_.mean()) for d in detectors}
    flag_counts = {d.name: int(d.fit_flags_.sum()) for d in detectors}

    low, high = cfg.get("validation.sane_band")
    warnings = [
        f"{name} flagged {rate:.2%}, outside the sane band "
        f"[{low:.0%}, {high:.0%}]"
        for name, rate in rate_table.items()
        if not (low <= rate <= high)
    ]

    live = live_detectors(detectors)
    live_flags = {d.name: d.fit_flags_ for d in live}

    agreement = {}
    for a, b in itertools.combinations(live_flags, 2):
        intersection = int(((live_flags[a] == 1) & (live_flags[b] == 1)).sum())
        union = int(((live_flags[a] == 1) | (live_flags[b] == 1)).sum())
        agreement[f"{a}|{b}"] = float(intersection / union) if union else 0.0

    threshold = cfg.get("ensemble.threshold")
    votes, ensemble_labels = combine_matrix(live_flags, threshold)

    vote_histogram = {v: int((votes == v).sum()) for v in range(len(live) + 1)}
    threshold_sweep = {
        required: int((votes >= required).sum())
        for required in range(1, len(live) + 1)
    }

    surrogate = train_surrogate(
        X,
        ensemble_labels,
        test_size=cfg.get("surrogate.test_size"),
        random_state=cfg.get("detectors.random_state"),
    )
    explainer_state = build_explainer_state(X)

    dbscan = next(d for d in detectors if d.name == "dbscan")

    report = TrainingReport(
        rate_table=rate_table,
        flag_counts=flag_counts,
        agreement=agreement,
        vote_histogram=vote_histogram,
        ensemble_rate=float(ensemble_labels.mean()),
        ensemble_flagged=int(ensemble_labels.sum()),
        threshold_sweep=threshold_sweep,
        surrogate_auc=surrogate.auc,
        surrogate_agreement=surrogate.agreement,
        dbscan_native_noise_rate=float(dbscan.native_noise_rate_),
        n_rows=len(X),
        warnings=warnings,
    )

    manifest = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(X),
        "contamination": cfg.get("detectors.contamination"),
        "ensemble_threshold": threshold,
        "votes_required": votes_required(len(live), threshold),
        "live_detectors": [d.name for d in live],
        "rate_table": rate_table,
        "flag_counts": flag_counts,
        "vote_histogram": vote_histogram,
        "ensemble_rate": report.ensemble_rate,
        "surrogate_auc": surrogate.auc,
        "surrogate_agreement": surrogate.agreement,
        "dbscan_native_noise_rate": report.dbscan_native_noise_rate,
        "warnings": warnings,
    }

    save_bundle(
        ArtifactBundle(
            manifest=manifest,
            scalers=scalers,
            feature_artifacts=feature_artifacts,
            profile_store=profile_store,
            detectors={d.name: d for d in detectors},
            surrogate=surrogate.model,
            explainer_state=explainer_state,
        ),
        dest,
    )

    return report


def format_report(report: TrainingReport) -> str:
    """Render the validation tables required before any serving work."""
    lines: list[str] = []
    n = report.n_rows

    lines.append(f"=== PER-DETECTOR ANOMALY RATES (n = {n}) ===")
    for name, rate in report.rate_table.items():
        lines.append(f"  {name:<20} {report.flag_counts[name]:>5}  {rate:6.2%}")
    lines.append(
        f"  dbscan native noise rate: {report.dbscan_native_noise_rate:.2%}"
    )

    lines.append("")
    lines.append("=== PAIRWISE AGREEMENT (Jaccard, live detectors) ===")
    for pair, jaccard in report.agreement.items():
        a, b = pair.split("|")
        lines.append(f"  {a:<20} vs {b:<20} {jaccard:.3f}")

    lines.append("")
    lines.append("=== VOTE HISTOGRAM ===")
    total_live = max(report.vote_histogram)
    for votes, count in sorted(report.vote_histogram.items()):
        lines.append(
            f"  {votes}/{total_live} votes: {count:>5} rows  {count / n:6.2%}"
        )

    lines.append("")
    lines.append("=== ENSEMBLE RATE BY THRESHOLD ===")
    for required, count in sorted(report.threshold_sweep.items()):
        lines.append(
            f"  >={required} of {total_live}: {count:>5} rows  {count / n:6.2%}"
        )
    lines.append(
        f"  selected: {report.ensemble_flagged} rows  {report.ensemble_rate:.2%}"
    )

    lines.append("")
    lines.append("=== SURROGATE FIDELITY TO THE ENSEMBLE ===")
    lines.append(f"  held-out AUC:       {report.surrogate_auc:.4f}")
    lines.append(f"  held-out agreement: {report.surrogate_agreement:.4f}")

    if report.warnings:
        lines.append("")
        lines.append("=== WARNINGS ===")
        lines.extend(f"  {w}" for w in report.warnings)

    return "\n".join(lines)


def main() -> None:
    cfg = Config.load()
    report = run_training(cfg)
    print(format_report(report))


if __name__ == "__main__":
    main()
