"""Generic majority-vote combiner.

Nothing here is hardcoded per detector: the combiner reads however many flags it
is given and derives the requirement from the configured threshold.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class EnsembleResult:
    is_anomaly: bool
    votes_for: int
    votes_total: int
    votes_required: int
    threshold: float

    def as_dict(self) -> dict:
        return asdict(self)


def votes_required(n_detectors: int, threshold: float) -> int:
    """ceil(n * threshold), floored at 1 so a threshold of 0 cannot flag everything."""
    if n_detectors < 1:
        raise ValueError("votes_required needs at least one detector")
    return max(1, math.ceil(n_detectors * threshold))


def combine_one(flags: dict[str, int], threshold: float) -> EnsembleResult:
    """Combine one row's per-detector flags into a verdict."""
    if not flags:
        raise ValueError("combine_one needs at least one detector flag")
    total = len(flags)
    required = votes_required(total, threshold)
    votes_for = int(sum(int(v) for v in flags.values()))
    return EnsembleResult(
        is_anomaly=bool(votes_for >= required),
        votes_for=votes_for,
        votes_total=total,
        votes_required=required,
        threshold=float(threshold),
    )


def combine_matrix(
    flags: dict[str, np.ndarray], threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised combine over a whole dataset. Returns (votes_for, is_anomaly)."""
    if not flags:
        raise ValueError("combine_matrix needs at least one detector flag array")
    required = votes_required(len(flags), threshold)
    stacked = np.vstack([np.asarray(v, dtype=int) for v in flags.values()])
    votes = stacked.sum(axis=0)
    return votes, (votes >= required).astype(int)
