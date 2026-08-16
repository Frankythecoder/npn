# Plan A — ML Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `/ml` Python package so that `score_transaction(raw_txn: dict) -> dict` works standalone on a hand-crafted transaction, and `train.py` reproduces the validation tables in spec §6.

**Architecture:** A single feature-engineering module serves both training and serving. Eight anomaly detectors share one Protocol and derive their binary flag by comparing a continuous score against a threshold frozen at fit time. Four of them vote; an XGBoost surrogate trained on that vote provides SHAP explanations. All fitted state is saved as one versioned artifact bundle that the scorer loads once.

**Tech Stack:** Python 3.11, pandas 2.2.3, numpy 2.0.2, scikit-learn 1.6.0, xgboost 3.2.0, shap 0.51.0, joblib 1.4.2, PyYAML 6.0.2, pytest 9.0.2.

**Spec:** `docs/superpowers/specs/2026-08-16-anomaly-detection-design.md`

## Global Constraints

- **`final.ipynb` is read-only.** Do not add, edit, remove or re-execute any cell. Do not copy code out of it. (spec §13)
- **`original.csv` is read-only.** Never write to it.
- **Scope is `/ml` and `/tests` only.** No FastAPI, no Docker, no `gcloud`, no GCS, no Firestore, no React. Those are Plans B and C.
- **Pin these exact versions** in `requirements.txt`: `pandas==2.2.3`, `numpy==2.0.2`, `scikit-learn==1.6.0`, `xgboost==3.2.0`, `shap==0.51.0`, `joblib==1.4.2`, `PyYAML==6.0.2`, `pytest==9.0.2`. Detectors are joblib pickles and a scikit-learn minor-version drift breaks unpickling. (spec §9)
- **`random_state=42`** everywhere one is accepted. Two training runs must produce identical rate tables. (spec §14)
- **Every detector score is oriented so higher = more anomalous.** (spec §4.3)
- **Detectors never scale or select columns.** The pipeline hands each detector an already-scaled, already-view-selected frame. Detectors only fit, score and threshold.
- **No "Phase 2", roadmap, rules-engine or case-management references** in any code, comment, docstring or output string.

## File Structure

| File | Responsibility |
|---|---|
| `ml/config.yaml` | All tunable values: contamination, voting threshold, detector hyperparameters, validation band |
| `ml/config.py` | Loads and validates `config.yaml`; dotted-path access |
| `ml/data/loader.py` | Read `original.csv`, parse dates, validate the 16 raw columns |
| `ml/features/engineer.py` | `FeatureArtifacts`, `ProfileStore`, `build_training_frame`, `transform_one` |
| `ml/detectors/base.py` | `AnomalyDetector` Protocol and `BaseDetector` threshold-transfer base class |
| `ml/detectors/isolation_forest.py` | Isolation Forest |
| `ml/detectors/one_class_svm.py` | One-Class SVM |
| `ml/detectors/lof.py` | LOF, two fitted objects (spec §4.5) |
| `ml/detectors/dbscan.py` | DBSCAN with nearest-core-sample live scoring (spec §4.6) |
| `ml/detectors/mcd.py` | MCD on the continuous view |
| `ml/detectors/gmm.py` | Gaussian Mixture |
| `ml/detectors/kmeans.py` | K-Means distance-to-centroid |
| `ml/detectors/pca_reconstruction.py` | PCA reconstruction error |
| `ml/detectors/registry.py` | Builds the detector roster from config |
| `ml/ensemble/voting.py` | Generic majority-vote combiner |
| `ml/explain/surrogate.py` | Trains XGBoost on ensemble labels, reports fidelity |
| `ml/explain/shap_explainer.py` | SHAP wrapper, top features, plain-English sentence |
| `ml/storage/artifacts.py` | `ArtifactBundle`, `save_bundle`, `load_bundle` |
| `ml/pipeline/train.py` | Full offline run; writes the bundle; prints the §6 tables |
| `ml/pipeline/score.py` | `score_transaction()` and the loaded-once singleton |

`ml/config.py` and `ml/detectors/registry.py` are not in spec §13. They are added because `config.yaml` needs a typed loader, and because the roster must be built in one place so `train.py` and `score.py` cannot disagree about it.

## Design Decisions Fixed Here

**1. Daily counts use a running count, not a whole-day count.** Spec §3.3 requires the live rule `day_counts.get(key, 0) + 1`. The notebook's training-side `transform('count')` gives every row the entire day's total, which is look-ahead bias and can never match the serving rule. Both sides use `cumcount() + 1`. This moves 37 rows of `DailyAccountVolume` and 24 rows of `DailyDeviceVelocity` from 2 to 1, so §6 numbers may shift by a few rows — see the acceptance tolerances in Task 16.

**2. `location_freq_default` is the minimum observed training frequency.** An unseen city maps to "as rare as the rarest known city", keeping the value in-distribution.

**3. `votes_required = max(1, ceil(n * threshold))`.** The floor of 1 prevents a threshold of 0 flagging every row.

**4. `ml/storage/artifacts.py` takes a filesystem `Path` only.** Plan B adds Cloud Storage by downloading to a temp directory and calling `load_bundle` on it. No storage abstraction is built now.

---

### Task 1: Repository scaffolding, config, and dependencies

**Files:**
- Create: `.gitignore`, `requirements.txt`, `pytest.ini`
- Create: `ml/__init__.py`, `ml/config.py`, `ml/config.yaml`
- Create: `ml/data/__init__.py`, `ml/features/__init__.py`, `ml/detectors/__init__.py`, `ml/ensemble/__init__.py`, `ml/explain/__init__.py`, `ml/storage/__init__.py`, `ml/pipeline/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ml.config.Config` with `Config.load(path=None) -> Config`, `Config.get(dotted: str, default=_UNSET) -> Any`, and module constant `CONFIG_PATH: Path`

- [ ] **Step 1: Initialise the repository**

This directory is not yet a git repository. Run:

```bash
cd /c/Users/Frank/npn
git init
git add original.csv final.ipynb docs
git commit -m "chore: import source dataset, reference notebook and design spec"
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
.env
artifacts/
*.egg-info/
.coverage
```

- [ ] **Step 3: Write `requirements.txt`**

```
pandas==2.2.3
numpy==2.0.2
scikit-learn==1.6.0
xgboost==3.2.0
shap==0.51.0
joblib==1.4.2
PyYAML==6.0.2
pytest==9.0.2
```

- [ ] **Step 4: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short
filterwarnings =
    ignore::DeprecationWarning
```

- [ ] **Step 5: Create the empty package files**

Create each of these as an empty file: `ml/__init__.py`, `ml/data/__init__.py`, `ml/features/__init__.py`, `ml/detectors/__init__.py`, `ml/ensemble/__init__.py`, `ml/explain/__init__.py`, `ml/storage/__init__.py`, `ml/pipeline/__init__.py`, `tests/__init__.py`.

- [ ] **Step 6: Write `ml/config.yaml`**

```yaml
data:
  csv_path: original.csv

detectors:
  contamination: 0.05
  random_state: 42
  isolation_forest:
    n_estimators: 200
  lof:
    n_neighbors: 20
  one_class_svm:
    kernel: rbf
    gamma: scale
    nu: 0.05
  dbscan:
    eps: 3.0
    min_samples: 5
  mcd: {}
  gmm:
    n_components: 5
    covariance_type: full
  kmeans:
    n_clusters: 8
    n_init: 10
  pca_reconstruction:
    n_components: 0.95

ensemble:
  threshold: 0.5

surrogate:
  min_auc: 0.95
  test_size: 0.25

validation:
  sane_band: [0.03, 0.07]

storage:
  local_dir: artifacts
```

- [ ] **Step 7: Write the failing test**

`tests/test_config.py`:

```python
import pytest
from ml.config import Config, CONFIG_PATH


def test_config_file_exists():
    assert CONFIG_PATH.exists()


def test_load_returns_expected_values():
    cfg = Config.load()
    assert cfg.get("detectors.contamination") == 0.05
    assert cfg.get("ensemble.threshold") == 0.5
    assert cfg.get("detectors.dbscan.eps") == 3.0
    assert cfg.get("detectors.dbscan.min_samples") == 5
    assert cfg.get("detectors.random_state") == 42
    assert cfg.get("validation.sane_band") == [0.03, 0.07]


def test_get_with_default_for_missing_key():
    cfg = Config.load()
    assert cfg.get("detectors.nonexistent", "fallback") == "fallback"


def test_get_raises_on_missing_key_without_default():
    cfg = Config.load()
    with pytest.raises(KeyError):
        cfg.get("detectors.nonexistent")


def test_missing_required_key_rejected_at_load():
    with pytest.raises(ValueError, match="missing required key"):
        Config({"data": {}})
```

- [ ] **Step 8: Run the test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.config'`

- [ ] **Step 9: Write `ml/config.py`**

```python
"""Loads and validates ml/config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"

_UNSET = object()

REQUIRED_KEYS = (
    "data.csv_path",
    "detectors.contamination",
    "detectors.random_state",
    "ensemble.threshold",
    "surrogate.min_auc",
    "validation.sane_band",
    "storage.local_dir",
)


class Config:
    """Dotted-path access over the parsed config.yaml tree."""

    def __init__(self, data: dict) -> None:
        self._data = data
        self._validate()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path is not None else CONFIG_PATH
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def get(self, dotted: str, default: Any = _UNSET) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _UNSET:
                    raise KeyError(dotted)
                return default
            node = node[part]
        return node

    def _validate(self) -> None:
        for key in REQUIRED_KEYS:
            if self.get(key, _UNSET) is _UNSET:
                raise ValueError(f"config missing required key: {key}")
```

- [ ] **Step 10: Run the test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 11: Commit**

```bash
git add .gitignore requirements.txt pytest.ini ml tests
git commit -m "feat: scaffold ml package with validated config loader"
```

---

### Task 2: Raw data loader

**Files:**
- Create: `ml/data/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `ml.config.Config`
- Produces: `RAW_COLUMNS: list[str]` (16 names), `load_raw(csv_path: str | Path) -> pd.DataFrame` with `TransactionDate` and `PreviousTransactionDate` as `datetime64[ns]`

- [ ] **Step 1: Write the failing test**

`tests/test_loader.py`:

```python
import pandas as pd
import pytest

from ml.config import Config
from ml.data.loader import RAW_COLUMNS, load_raw


@pytest.fixture(scope="module")
def df():
    return load_raw(Config.load().get("data.csv_path"))


def test_row_and_column_count(df):
    assert df.shape == (2512, 16)


def test_all_raw_columns_present(df):
    assert list(df.columns) == RAW_COLUMNS


def test_dates_parsed_to_datetime(df):
    assert pd.api.types.is_datetime64_any_dtype(df["TransactionDate"])
    assert pd.api.types.is_datetime64_any_dtype(df["PreviousTransactionDate"])


def test_no_missing_values(df):
    assert df.isnull().sum().sum() == 0


def test_index_is_a_clean_range(df):
    assert df.index.tolist() == list(range(2512))


def test_missing_column_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("TransactionID,AccountID\nTX1,AC1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_raw(bad)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.data.loader'`

- [ ] **Step 3: Write `ml/data/loader.py`**

```python
"""Loads and validates the raw transaction CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_COLUMNS = [
    "TransactionID",
    "AccountID",
    "TransactionAmount",
    "TransactionDate",
    "TransactionType",
    "Location",
    "DeviceID",
    "IP Address",
    "MerchantID",
    "Channel",
    "CustomerAge",
    "CustomerOccupation",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "PreviousTransactionDate",
]

DATE_COLUMNS = ["TransactionDate", "PreviousTransactionDate"]


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw CSV, verify its schema, and parse the date columns."""
    df = pd.read_csv(csv_path)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    df = df[RAW_COLUMNS].copy()
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="raise")

    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_loader.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ml/data/loader.py tests/test_loader.py
git commit -m "feat: add raw CSV loader with schema validation"
```

---

### Task 3: Feature artifacts, profile store, and the training frame

**Files:**
- Create: `ml/features/engineer.py`
- Test: `tests/test_engineer_training.py`

**Interfaces:**
- Consumes: `ml.data.loader.load_raw`
- Produces:
  - `FEATURE_COLUMNS: list[str]` (19, explicit order)
  - `CONTINUOUS_COLUMNS: list[str]` (7)
  - `CATEGORICAL_LEVELS: dict[str, list[str]]`
  - `FeatureArtifacts` dataclass with fields `location_freq`, `location_freq_default`, `feature_columns`, `continuous_columns`, `categorical_levels`, `time_since_last_tx_median`
  - `ProfileStore` with `account_last_tx: dict[str, pd.Timestamp]`, `account_day_counts: dict[tuple[str, str], int]`, `device_day_counts: dict[tuple[str, str], int]`, and method `observe(account_id: str, device_id: str, txn_date: pd.Timestamp) -> None`
  - `build_training_frame(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, FeatureArtifacts, ProfileStore]`

- [ ] **Step 1: Write the failing test**

`tests/test_engineer_training.py`:

```python
import numpy as np
import pandas as pd
import pytest

from ml.config import Config
from ml.data.loader import load_raw
from ml.features.engineer import (
    CATEGORICAL_LEVELS,
    CONTINUOUS_COLUMNS,
    FEATURE_COLUMNS,
    build_training_frame,
)


@pytest.fixture(scope="module")
def built():
    return build_training_frame(load_raw(Config.load().get("data.csv_path")))


def test_frame_shape_and_column_order(built):
    X, _, _ = built
    assert X.shape == (2512, 19)
    assert list(X.columns) == FEATURE_COLUMNS


def test_continuous_columns_are_a_subset_of_feature_columns():
    assert len(CONTINUOUS_COLUMNS) == 7
    assert set(CONTINUOUS_COLUMNS).issubset(set(FEATURE_COLUMNS))


def test_all_values_numeric_and_finite(built):
    X, _, _ = built
    assert X.select_dtypes(include=[np.number]).shape[1] == 19
    assert np.isfinite(X.to_numpy()).all()


def test_time_since_last_tx_is_positive_and_sane(built):
    X, artifacts, _ = built
    gaps = X["TimeSinceLastTx_Hours"]
    assert (gaps > 0).all(), "the rebuilt gap must never be negative (spec 2.1)"
    assert 900 < artifacts.time_since_last_tx_median < 980


def test_daily_counts_are_running_not_whole_day(built):
    X, _, _ = built
    assert X["DailyAccountVolume"].value_counts().to_dict() == {1: 2475, 2: 37}
    assert X["DailyDeviceVelocity"].value_counts().to_dict() == {1: 2488, 2: 24}


def test_one_hot_groups_each_sum_to_one(built):
    X, _, _ = built
    for prefix, levels in CATEGORICAL_LEVELS.items():
        cols = [f"{prefix}_{lvl}" for lvl in levels]
        assert (X[cols].sum(axis=1) == 1).all()


def test_utilization_ratio_matches_definition(built):
    X, _, _ = built
    raw = load_raw(Config.load().get("data.csv_path"))
    expected = raw["TransactionAmount"] / raw["AccountBalance"]
    assert np.allclose(X["UtilizationRatio"], expected)


def test_artifacts_capture_location_frequencies(built):
    _, artifacts, _ = built
    assert len(artifacts.location_freq) == 43
    assert artifacts.location_freq_default == min(artifacts.location_freq.values())
    assert artifacts.feature_columns == FEATURE_COLUMNS


def test_profile_store_covers_every_account_and_device(built):
    _, _, profiles = built
    raw = load_raw(Config.load().get("data.csv_path"))
    assert len(profiles.account_last_tx) == raw["AccountID"].nunique()
    assert sum(profiles.account_day_counts.values()) == 2512
    assert sum(profiles.device_day_counts.values()) == 2512


def test_profile_last_tx_is_the_accounts_latest(built):
    _, _, profiles = built
    raw = load_raw(Config.load().get("data.csv_path"))
    expected = raw.groupby("AccountID")["TransactionDate"].max()
    for acct, ts in expected.items():
        assert profiles.account_last_tx[acct] == ts
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_engineer_training.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.features.engineer'`

- [ ] **Step 3: Write `ml/features/engineer.py`**

```python
"""Feature engineering shared by training and serving.

Both entry points emit the same 19 columns in the same frozen order. The column
order is persisted in FeatureArtifacts so any drift between training and serving
fails loudly instead of silently misaligning values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

CATEGORICAL_LEVELS: dict[str, list[str]] = {
    "TransactionType": ["Credit", "Debit"],
    "Channel": ["ATM", "Branch", "Online"],
    "CustomerOccupation": ["Doctor", "Engineer", "Retired", "Student"],
}

NUMERIC_COLUMNS = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "TimeSinceLastTx_Hours",
    "DailyAccountVolume",
    "UtilizationRatio",
    "DailyDeviceVelocity",
    "Location_Freq",
]

ONE_HOT_COLUMNS = [
    f"{prefix}_{level}"
    for prefix, levels in CATEGORICAL_LEVELS.items()
    for level in levels
]

# The frozen 19, built explicitly rather than relying on get_dummies ordering.
FEATURE_COLUMNS = NUMERIC_COLUMNS + ONE_HOT_COLUMNS

# MCD is a Gaussian elliptical estimator and cannot be fitted on binary dummies
# or on the 95-98% single-valued counters (spec 2.2). It is the only consumer.
CONTINUOUS_COLUMNS = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "AccountBalance",
    "TimeSinceLastTx_Hours",
    "UtilizationRatio",
    "Location_Freq",
]


@dataclass
class FeatureArtifacts:
    """Training-derived state a single incoming row cannot supply."""

    location_freq: dict[str, int]
    location_freq_default: int
    feature_columns: list[str]
    continuous_columns: list[str]
    categorical_levels: dict[str, list[str]]
    time_since_last_tx_median: float


@dataclass
class ProfileStore:
    """Per-account and per-device history for the three history-dependent features."""

    account_last_tx: dict[str, pd.Timestamp] = field(default_factory=dict)
    account_day_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    device_day_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    @staticmethod
    def _day_key(txn_date: pd.Timestamp) -> str:
        return pd.Timestamp(txn_date).date().isoformat()

    def account_day_count(self, account_id: str, txn_date: pd.Timestamp) -> int:
        return self.account_day_counts.get((account_id, self._day_key(txn_date)), 0)

    def device_day_count(self, device_id: str, txn_date: pd.Timestamp) -> int:
        return self.device_day_counts.get((device_id, self._day_key(txn_date)), 0)

    def gap_hours(
        self, account_id: str, txn_date: pd.Timestamp, default: float
    ) -> tuple[float, bool]:
        """Return (gap in hours, seen_before). Falls back to `default` if unseen."""
        previous = self.account_last_tx.get(account_id)
        if previous is None:
            return default, False
        delta = (pd.Timestamp(txn_date) - previous).total_seconds() / 3600.0
        return delta, True

    def observe(
        self, account_id: str, device_id: str, txn_date: pd.Timestamp
    ) -> None:
        """Record a transaction so subsequent scores see it as history."""
        txn_date = pd.Timestamp(txn_date)
        day = self._day_key(txn_date)
        previous = self.account_last_tx.get(account_id)
        if previous is None or txn_date > previous:
            self.account_last_tx[account_id] = txn_date
        self.account_day_counts[(account_id, day)] = (
            self.account_day_counts.get((account_id, day), 0) + 1
        )
        self.device_day_counts[(device_id, day)] = (
            self.device_day_counts.get((device_id, day), 0) + 1
        )


def _chronological_running_count(
    df: pd.DataFrame, key: str, day: pd.Series
) -> pd.Series:
    """Count of prior same-day transactions for `key`, inclusive of this row.

    A running count, not a whole-day total: at the first transaction of a day it
    is not knowable that a second will follow. This also makes the training value
    identical to the serving rule in spec 3.3.
    """
    ordering = df.sort_values([key, "TransactionDate"]).index
    counts = (
        df.loc[ordering]
        .assign(_day=day.loc[ordering])
        .groupby([key, "_day"])
        .cumcount()
        + 1
    )
    return counts.reindex(df.index)


def build_training_frame(
    df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, FeatureArtifacts, ProfileStore]:
    """Engineer the full 19-column training matrix and the artifacts serving needs."""
    df = df_raw.copy()
    day = df["TransactionDate"].dt.date

    # Gap to this account's own previous transaction. PreviousTransactionDate is an
    # ingest timestamp and is deliberately unused (spec 2.1).
    ordering = df.sort_values(["AccountID", "TransactionDate"]).index
    gap = (
        df.loc[ordering]
        .groupby("AccountID")["TransactionDate"]
        .diff()
        .dt.total_seconds()
        / 3600.0
    ).reindex(df.index)
    gap_median = float(gap.median())
    df["TimeSinceLastTx_Hours"] = gap.fillna(gap_median)

    df["DailyAccountVolume"] = _chronological_running_count(df, "AccountID", day)
    df["DailyDeviceVelocity"] = _chronological_running_count(df, "DeviceID", day)
    df["UtilizationRatio"] = df["TransactionAmount"] / df["AccountBalance"]

    location_freq = df["Location"].value_counts().to_dict()
    df["Location_Freq"] = df["Location"].map(location_freq)

    frame = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLUMNS:
        frame[col] = df[col].astype(float)
    for prefix, levels in CATEGORICAL_LEVELS.items():
        for level in levels:
            frame[f"{prefix}_{level}"] = (df[prefix] == level).astype(int)
    frame = frame[FEATURE_COLUMNS]

    artifacts = FeatureArtifacts(
        location_freq={str(k): int(v) for k, v in location_freq.items()},
        location_freq_default=int(min(location_freq.values())),
        feature_columns=list(FEATURE_COLUMNS),
        continuous_columns=list(CONTINUOUS_COLUMNS),
        categorical_levels={k: list(v) for k, v in CATEGORICAL_LEVELS.items()},
        time_since_last_tx_median=gap_median,
    )

    profiles = ProfileStore()
    for row in df.sort_values("TransactionDate").itertuples(index=False):
        profiles.observe(row.AccountID, row.DeviceID, row.TransactionDate)

    return frame, artifacts, profiles
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_engineer_training.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add ml/features/engineer.py tests/test_engineer_training.py
git commit -m "feat: build training feature matrix, artifacts and profile store"
```

---

### Task 4: Single-transaction transform and train/serve parity

**Files:**
- Modify: `ml/features/engineer.py` (append `transform_one` and `RAW_INPUT_FIELDS`)
- Test: `tests/test_engineer_parity.py`

**Interfaces:**
- Consumes: `FeatureArtifacts`, `ProfileStore`, `FEATURE_COLUMNS` from Task 3
- Produces: `transform_one(raw_txn: dict, artifacts: FeatureArtifacts, profiles: ProfileStore) -> tuple[pd.DataFrame, list[str]]` returning a `(1, 19)` frame and a list of warning strings; `RAW_INPUT_FIELDS: list[str]`

- [ ] **Step 1: Write the failing test**

`tests/test_engineer_parity.py`:

```python
import numpy as np
import pandas as pd
import pytest

from ml.config import Config
from ml.data.loader import load_raw
from ml.features.engineer import (
    FEATURE_COLUMNS,
    ProfileStore,
    build_training_frame,
    transform_one,
)


@pytest.fixture(scope="module")
def raw():
    return load_raw(Config.load().get("data.csv_path"))


@pytest.fixture(scope="module")
def built(raw):
    return build_training_frame(raw)


def _profiles_up_to(raw: pd.DataFrame, cutoff: pd.Timestamp) -> ProfileStore:
    """A profile store holding only transactions strictly before `cutoff`."""
    store = ProfileStore()
    earlier = raw[raw["TransactionDate"] < cutoff].sort_values("TransactionDate")
    for row in earlier.itertuples(index=False):
        store.observe(row.AccountID, row.DeviceID, row.TransactionDate)
    return store


def test_transform_one_returns_the_frozen_columns(raw, built):
    _, artifacts, profiles = built
    frame, _ = transform_one(raw.iloc[0].to_dict(), artifacts, profiles)
    assert frame.shape == (1, 19)
    assert list(frame.columns) == FEATURE_COLUMNS


def test_train_serve_parity_on_a_repeat_account(raw, built):
    """The critical test: rewind the profile store and reproduce a training row."""
    X, artifacts, _ = built
    repeat_accounts = raw["AccountID"].value_counts()
    account = repeat_accounts[repeat_accounts >= 2].index[0]
    rows = raw[raw["AccountID"] == account].sort_values("TransactionDate")
    target = rows.iloc[1]

    profiles = _profiles_up_to(raw, target["TransactionDate"])
    frame, warnings = transform_one(target.to_dict(), artifacts, profiles)

    expected = X.loc[target.name]
    for col in FEATURE_COLUMNS:
        assert frame.iloc[0][col] == pytest.approx(expected[col], rel=1e-9), col
    assert warnings == []


def test_train_serve_parity_across_many_rows(raw, built):
    X, artifacts, _ = built
    sample = raw.sample(n=25, random_state=42)
    for _, target in sample.iterrows():
        profiles = _profiles_up_to(raw, target["TransactionDate"])
        frame, _ = transform_one(target.to_dict(), artifacts, profiles)
        expected = X.loc[target.name]
        assert np.allclose(
            frame.iloc[0].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=1e-9,
        ), f"parity failed for row {target.name}"


def test_unseen_account_uses_the_median_gap_and_warns(raw, built):
    _, artifacts, profiles = built
    txn = raw.iloc[0].to_dict()
    txn["AccountID"] = "AC99999"
    frame, warnings = transform_one(txn, artifacts, profiles)
    assert frame.iloc[0]["TimeSinceLastTx_Hours"] == pytest.approx(
        artifacts.time_since_last_tx_median
    )
    assert any("unseen account" in w for w in warnings)


def test_unseen_city_uses_the_default_frequency_and_warns(raw, built):
    _, artifacts, profiles = built
    txn = raw.iloc[0].to_dict()
    txn["Location"] = "Atlantis"
    frame, warnings = transform_one(txn, artifacts, profiles)
    assert frame.iloc[0]["Location_Freq"] == artifacts.location_freq_default
    assert any("unseen location" in w for w in warnings)


def test_unseen_categorical_level_yields_all_zeros_and_warns(raw, built):
    _, artifacts, profiles = built
    txn = raw.iloc[0].to_dict()
    txn["Channel"] = "Carrier Pigeon"
    frame, warnings = transform_one(txn, artifacts, profiles)
    assert frame.iloc[0][["Channel_ATM", "Channel_Branch", "Channel_Online"]].sum() == 0
    assert any("unseen Channel" in w for w in warnings)


def test_daily_counts_are_self_inclusive(raw, built):
    """A lone transaction gets 1, matching how a lone training row gets 1."""
    _, artifacts, _ = built
    txn = raw.iloc[0].to_dict()
    txn["AccountID"] = "AC99999"
    txn["DeviceID"] = "D99999"
    frame, _ = transform_one(txn, artifacts, ProfileStore())
    assert frame.iloc[0]["DailyAccountVolume"] == 1
    assert frame.iloc[0]["DailyDeviceVelocity"] == 1


def test_missing_required_field_is_rejected(built):
    _, artifacts, profiles = built
    with pytest.raises(ValueError, match="missing required fields"):
        transform_one({"TransactionAmount": 10.0}, artifacts, profiles)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_engineer_parity.py -v`
Expected: FAIL with `ImportError: cannot import name 'transform_one'`

- [ ] **Step 3: Append `transform_one` to `ml/features/engineer.py`**

```python
RAW_INPUT_FIELDS = [
    "AccountID",
    "DeviceID",
    "Location",
    "TransactionDate",
    "TransactionAmount",
    "AccountBalance",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "TransactionType",
    "Channel",
    "CustomerOccupation",
]


def transform_one(
    raw_txn: dict,
    artifacts: FeatureArtifacts,
    profiles: ProfileStore,
) -> tuple[pd.DataFrame, list[str]]:
    """Engineer a single incoming transaction into the frozen 19-column frame.

    Returns the frame and a list of human-readable warnings. Unseen accounts,
    cities and categorical levels degrade to documented defaults rather than
    raising, so an unexpected input is visible instead of fatal.
    """
    missing = [f for f in RAW_INPUT_FIELDS if f not in raw_txn]
    if missing:
        raise ValueError(f"transform_one: missing required fields {missing}")

    warnings: list[str] = []
    txn_date = pd.Timestamp(raw_txn["TransactionDate"])
    account_id = str(raw_txn["AccountID"])
    device_id = str(raw_txn["DeviceID"])

    gap, seen = profiles.gap_hours(
        account_id, txn_date, artifacts.time_since_last_tx_median
    )
    if not seen:
        warnings.append(
            f"unseen account {account_id}: TimeSinceLastTx_Hours filled with the "
            f"training median ({artifacts.time_since_last_tx_median:.2f}h)"
        )

    city = str(raw_txn["Location"])
    if city in artifacts.location_freq:
        location_freq = artifacts.location_freq[city]
    else:
        location_freq = artifacts.location_freq_default
        warnings.append(
            f"unseen location {city!r}: Location_Freq set to the training minimum "
            f"({artifacts.location_freq_default})"
        )

    balance = float(raw_txn["AccountBalance"])
    if balance == 0:
        raise ValueError("transform_one: AccountBalance must be non-zero")

    values: dict[str, float] = {
        "TransactionAmount": float(raw_txn["TransactionAmount"]),
        "CustomerAge": float(raw_txn["CustomerAge"]),
        "TransactionDuration": float(raw_txn["TransactionDuration"]),
        "LoginAttempts": float(raw_txn["LoginAttempts"]),
        "AccountBalance": balance,
        "TimeSinceLastTx_Hours": float(gap),
        # Self-inclusive: this transaction counts itself, exactly as a lone
        # training row is counted once.
        "DailyAccountVolume": float(profiles.account_day_count(account_id, txn_date) + 1),
        "UtilizationRatio": float(raw_txn["TransactionAmount"]) / balance,
        "DailyDeviceVelocity": float(profiles.device_day_count(device_id, txn_date) + 1),
        "Location_Freq": float(location_freq),
    }

    for prefix, levels in artifacts.categorical_levels.items():
        supplied = raw_txn[prefix]
        if supplied not in levels:
            warnings.append(
                f"unseen {prefix} {supplied!r}: all {prefix} indicators set to 0"
            )
        for level in levels:
            values[f"{prefix}_{level}"] = float(supplied == level)

    frame = pd.DataFrame([values])[artifacts.feature_columns]
    return frame, warnings
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_engineer_parity.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the whole suite to check nothing regressed**

Run: `python -m pytest -v`
Expected: 29 passed

- [ ] **Step 6: Commit**

```bash
git add ml/features/engineer.py tests/test_engineer_parity.py
git commit -m "feat: add single-transaction transform with train/serve parity"
```

---

### Task 5: Detector Protocol and threshold-transfer base class

**Files:**
- Create: `ml/detectors/base.py`
- Test: `tests/test_detector_base.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `AnomalyDetector` Protocol with `name`, `view`, `scaler`, `live_scorable`, `fit`, `score`, `flag`
  - `BaseDetector` ABC with `fit(X) -> Self`, `score(X) -> np.ndarray`, `flag(X) -> np.ndarray`, `score_percentile(value: float) -> float`, and post-fit attributes `threshold_: float`, `live_threshold_: float`, `train_scores_: np.ndarray` (sorted), `fit_flags_: np.ndarray`
  - Subclasses implement `_fit(X: np.ndarray) -> None` and `_score(X: np.ndarray) -> np.ndarray`, and may override `_training_scores(X: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

`tests/test_detector_base.py`:

```python
import numpy as np
import pandas as pd
import pytest

from ml.detectors.base import BaseDetector


class FakeDetector(BaseDetector):
    """Scores each row by its first column, so expectations are exact."""

    name = "fake"
    view = "full"
    scaler = "standard"
    live_scorable = True

    def _fit(self, X: np.ndarray) -> None:
        self._fitted = True

    def _score(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0].astype(float)


@pytest.fixture
def frame():
    return pd.DataFrame({"a": np.arange(100.0), "b": np.zeros(100)})


def test_fit_returns_self(frame):
    det = FakeDetector(contamination=0.05)
    assert det.fit(frame) is det


def test_threshold_is_the_contamination_percentile(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    assert det.threshold_ == pytest.approx(np.percentile(np.arange(100.0), 95))


def test_flag_rate_matches_contamination_on_training_data(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    flags = det.flag(frame)
    assert flags.shape == (100,)
    assert set(np.unique(flags)).issubset({0, 1})
    assert abs(flags.mean() - 0.05) <= 0.01


def test_fit_flags_matches_flag_on_training_data(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    assert np.array_equal(det.fit_flags_, det.flag(frame))


def test_flag_agrees_with_thresholding_score(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    expected = (det.score(frame) >= det.live_threshold_).astype(int)
    assert np.array_equal(det.flag(frame), expected)


def test_train_scores_are_sorted(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    assert np.array_equal(det.train_scores_, np.sort(det.train_scores_))
    assert len(det.train_scores_) == 100


def test_score_percentile_bounds(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    assert det.score_percentile(-1.0) == pytest.approx(0.0)
    assert det.score_percentile(999.0) == pytest.approx(100.0)
    assert 45.0 <= det.score_percentile(50.0) <= 55.0


def test_score_before_fit_raises(frame):
    det = FakeDetector(contamination=0.05)
    with pytest.raises(RuntimeError, match="not fitted"):
        det.flag(frame)


def test_column_mismatch_is_rejected(frame):
    det = FakeDetector(contamination=0.05).fit(frame)
    with pytest.raises(ValueError, match="column mismatch"):
        det.score(pd.DataFrame({"a": [1.0], "z": [2.0]}))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_detector_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.detectors.base'`

- [ ] **Step 3: Write `ml/detectors/base.py`**

```python
"""Shared detector contract and the threshold-transfer base class.

Every detector derives its binary flag by comparing a continuous score against a
threshold frozen at fit time. That erases the difference between detectors that
expose a native predict() and those that do not, guarantees each one flags the
contamination rate on its training data, and judges new rows against a fixed
boundary, which is the correct live-scoring semantics.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class AnomalyDetector(Protocol):
    name: str
    view: Literal["full", "continuous"]
    scaler: Literal["standard", "robust", "continuous"]
    live_scorable: bool

    def fit(self, X: pd.DataFrame) -> "AnomalyDetector": ...
    def score(self, X: pd.DataFrame) -> np.ndarray: ...
    def flag(self, X: pd.DataFrame) -> np.ndarray: ...


class BaseDetector(ABC):
    """Implements fit/score/flag on top of subclass _fit and _score hooks."""

    name: str = "base"
    view: str = "full"
    scaler: str = "standard"
    live_scorable: bool = True

    def __init__(self, contamination: float = 0.05, **params) -> None:
        self.contamination = float(contamination)
        self.params = params
        self.threshold_: float | None = None
        self.live_threshold_: float | None = None
        self.train_scores_: np.ndarray | None = None
        self.fit_flags_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None

    # --- subclass hooks -------------------------------------------------

    @abstractmethod
    def _fit(self, X: np.ndarray) -> None:
        """Fit the underlying estimator."""

    @abstractmethod
    def _score(self, X: np.ndarray) -> np.ndarray:
        """Score arbitrary rows. Higher must mean more anomalous."""

    def _training_scores(self, X: np.ndarray) -> np.ndarray:
        """Scores used to set the threshold. LOF overrides this (spec 4.5)."""
        return self._score(X)

    # --- public API -----------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "BaseDetector":
        self.feature_names_ = list(X.columns)
        values = X.to_numpy(dtype=float)
        self._fit(values)

        scores = np.asarray(self._training_scores(values), dtype=float)
        self.threshold_ = float(
            np.percentile(scores, 100.0 * (1.0 - self.contamination))
        )
        self.live_threshold_ = self.threshold_
        self.train_scores_ = np.sort(scores)
        self.fit_flags_ = (scores >= self.threshold_).astype(int)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        self._check_ready(X)
        return np.asarray(self._score(X.to_numpy(dtype=float)), dtype=float)

    def flag(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) >= self.live_threshold_).astype(int)

    def score_percentile(self, value: float) -> float:
        """Where `value` falls in the training score distribution, as 0-100."""
        if self.train_scores_ is None:
            raise RuntimeError(f"{self.name} is not fitted")
        position = float(np.searchsorted(self.train_scores_, value, side="right"))
        return 100.0 * position / len(self.train_scores_)

    # --- internals ------------------------------------------------------

    def _check_ready(self, X: pd.DataFrame) -> None:
        if self.threshold_ is None:
            raise RuntimeError(f"{self.name} is not fitted")
        if list(X.columns) != self.feature_names_:
            raise ValueError(
                f"{self.name}: column mismatch. "
                f"expected {self.feature_names_}, got {list(X.columns)}"
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_detector_base.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add ml/detectors/base.py tests/test_detector_base.py
git commit -m "feat: add detector protocol and threshold-transfer base class"
```

---

### Task 6: Isolation Forest and One-Class SVM

**Files:**
- Create: `ml/detectors/isolation_forest.py`, `ml/detectors/one_class_svm.py`
- Test: `tests/test_detectors_simple.py`

**Interfaces:**
- Consumes: `ml.detectors.base.BaseDetector`
- Produces: `IsolationForestDetector(contamination, n_estimators, random_state)`, `OneClassSVMDetector(contamination, kernel, gamma, nu)`

- [ ] **Step 1: Write the failing test**

`tests/test_detectors_simple.py`:

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.isolation_forest import IsolationForestDetector
from ml.detectors.one_class_svm import OneClassSVMDetector
from ml.features.engineer import build_training_frame


@pytest.fixture(scope="module")
def scaled():
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)


def test_isolation_forest_flags_the_contamination_rate(scaled):
    det = IsolationForestDetector(contamination=0.05, n_estimators=200, random_state=42)
    det.fit(scaled)
    assert det.fit_flags_.sum() == 126
    assert det.name == "isolation_forest"
    assert det.live_scorable is True
    assert det.scaler == "standard"


def test_one_class_svm_flags_the_contamination_rate(scaled):
    det = OneClassSVMDetector(contamination=0.05, kernel="rbf", gamma="scale", nu=0.05)
    det.fit(scaled)
    assert det.fit_flags_.sum() == 126
    assert det.name == "one_class_svm"
    assert det.live_scorable is True


@pytest.mark.parametrize(
    "factory",
    [
        lambda: IsolationForestDetector(
            contamination=0.05, n_estimators=200, random_state=42
        ),
        lambda: OneClassSVMDetector(
            contamination=0.05, kernel="rbf", gamma="scale", nu=0.05
        ),
    ],
)
def test_scores_a_single_unseen_row(scaled, factory):
    det = factory().fit(scaled)
    row = scaled.iloc[[0]]
    assert det.score(row).shape == (1,)
    assert det.flag(row).shape == (1,)


def test_isolation_forest_is_deterministic(scaled):
    a = IsolationForestDetector(
        contamination=0.05, n_estimators=200, random_state=42
    ).fit(scaled)
    b = IsolationForestDetector(
        contamination=0.05, n_estimators=200, random_state=42
    ).fit(scaled)
    assert np.array_equal(a.fit_flags_, b.fit_flags_)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_detectors_simple.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.detectors.isolation_forest'`

- [ ] **Step 3: Write `ml/detectors/isolation_forest.py`**

```python
"""Isolation Forest detector."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from ml.detectors.base import BaseDetector


class IsolationForestDetector(BaseDetector):
    name = "isolation_forest"
    view = "full"
    scaler = "standard"
    live_scorable = True

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray) -> None:
        self._model = IsolationForest(
            n_estimators=self.params["n_estimators"],
            contamination=self.contamination,
            max_samples="auto",
            random_state=self.params["random_state"],
            n_jobs=-1,
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # decision_function is higher for inliers, so negate it.
        return -self._model.decision_function(X)
```

- [ ] **Step 4: Write `ml/detectors/one_class_svm.py`**

```python
"""One-Class SVM detector."""
from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from ml.detectors.base import BaseDetector


class OneClassSVMDetector(BaseDetector):
    name = "one_class_svm"
    view = "full"
    scaler = "standard"
    live_scorable = True

    def __init__(
        self,
        contamination: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        nu: float = 0.05,
    ) -> None:
        super().__init__(
            contamination=contamination, kernel=kernel, gamma=gamma, nu=nu
        )

    def _fit(self, X: np.ndarray) -> None:
        self._model = OneClassSVM(
            kernel=self.params["kernel"],
            gamma=self.params["gamma"],
            nu=self.params["nu"],
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # decision_function is higher for inliers, so negate it.
        return -self._model.decision_function(X)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_detectors_simple.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add ml/detectors/isolation_forest.py ml/detectors/one_class_svm.py tests/test_detectors_simple.py
git commit -m "feat: add isolation forest and one-class svm detectors"
```

---

### Task 7: LOF with two fitted objects

**Files:**
- Create: `ml/detectors/lof.py`
- Test: `tests/test_detector_lof.py`

**Interfaces:**
- Consumes: `ml.detectors.base.BaseDetector`
- Produces: `LOFDetector(contamination, n_neighbors)` with extra post-fit attributes `live_threshold_` (distinct from `threshold_`) and `live_train_scores_: np.ndarray`

- [ ] **Step 1: Write the failing test**

`tests/test_detector_lof.py`:

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.lof import LOFDetector
from ml.features.engineer import build_training_frame


@pytest.fixture(scope="module")
def scaled():
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    return pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)


@pytest.fixture(scope="module")
def fitted(scaled):
    return LOFDetector(contamination=0.05, n_neighbors=20).fit(scaled)


def test_uses_the_robust_scaler(fitted):
    assert fitted.scaler == "robust"
    assert fitted.name == "lof"
    assert fitted.live_scorable is True


def test_flags_the_contamination_rate(fitted):
    assert fitted.fit_flags_.sum() == 126


def test_two_distinct_thresholds_are_persisted(fitted):
    assert fitted.threshold_ is not None
    assert fitted.live_threshold_ is not None
    assert fitted.threshold_ != fitted.live_threshold_, (
        "the novelty=False and novelty=True fits are different quantities "
        "and must carry separate thresholds (spec 4.5)"
    )


def test_both_score_distributions_are_persisted(fitted):
    assert len(fitted.train_scores_) == 2512
    assert len(fitted.live_train_scores_) == 2512


def test_novelty_copy_scores_a_single_unseen_row(fitted, scaled):
    row = scaled.iloc[[0]]
    assert fitted.score(row).shape == (1,)
    assert fitted.flag(row).shape == (1,)


def test_live_flag_rate_on_training_data_is_near_contamination(fitted, scaled):
    rate = fitted.flag(scaled).mean()
    assert 0.03 <= rate <= 0.07
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_detector_lof.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.detectors.lof'`

- [ ] **Step 3: Write `ml/detectors/lof.py`**

```python
"""Local Outlier Factor, fitted twice.

scikit-learn's novelty=False and novelty=True modes are not interchangeable. The
former exposes negative_outlier_factor_ but has no predict(); the latter has
predict() but its training-data scores are a different quantity. Both are fitted
on the same data and both thresholds are persisted (spec 4.5):

  - the novelty=False fit produces training labels and the rate-table entry
  - the novelty=True fit serves live scoring, calibrated on its own scale
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ml.detectors.base import BaseDetector


class LOFDetector(BaseDetector):
    name = "lof"
    view = "full"
    scaler = "robust"
    live_scorable = True

    def __init__(self, contamination: float = 0.05, n_neighbors: int = 20) -> None:
        super().__init__(contamination=contamination, n_neighbors=n_neighbors)
        self.live_train_scores_: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        # Training-time fit: produces negative_outlier_factor_ for the labels.
        self._train_model = LocalOutlierFactor(
            n_neighbors=self.params["n_neighbors"],
            contamination=self.contamination,
            metric="euclidean",
            n_jobs=-1,
        )
        self._train_model.fit_predict(X)

        # Serving fit: the only copy that can score unseen rows.
        self._live_model = LocalOutlierFactor(
            n_neighbors=self.params["n_neighbors"],
            contamination=self.contamination,
            metric="euclidean",
            novelty=True,
            n_jobs=-1,
        ).fit(X)

    def _training_scores(self, X: np.ndarray) -> np.ndarray:
        # negative_outlier_factor_ is lower for outliers, so negate it. This
        # attribute exists only for the data the model was fitted on.
        return -self._train_model.negative_outlier_factor_

    def _score(self, X: np.ndarray) -> np.ndarray:
        # score_samples is higher for inliers, so negate it.
        return -self._live_model.score_samples(X)

    def fit(self, X) -> "LOFDetector":
        super().fit(X)
        # The novelty copy lives on a different scale, so calibrate it separately.
        live_scores = np.asarray(
            self._score(X.to_numpy(dtype=float)), dtype=float
        )
        self.live_threshold_ = float(
            np.percentile(live_scores, 100.0 * (1.0 - self.contamination))
        )
        self.live_train_scores_ = np.sort(live_scores)
        return self

    def score_percentile(self, value: float) -> float:
        """Percentile against the live score distribution, which is what score() emits."""
        if self.live_train_scores_ is None:
            raise RuntimeError("lof is not fitted")
        position = float(
            np.searchsorted(self.live_train_scores_, value, side="right")
        )
        return 100.0 * position / len(self.live_train_scores_)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_detector_lof.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ml/detectors/lof.py tests/test_detector_lof.py
git commit -m "feat: add LOF detector with separate training and novelty fits"
```

---

### Task 8: DBSCAN with nearest-core-sample live scoring

**Files:**
- Create: `ml/detectors/dbscan.py`
- Test: `tests/test_detector_dbscan.py`

**Interfaces:**
- Consumes: `ml.detectors.base.BaseDetector`
- Produces: `DBSCANDetector(contamination, eps, min_samples)` with post-fit attributes `core_samples_: np.ndarray`, `native_noise_rate_: float`, `n_clusters_: int`

- [ ] **Step 1: Write the failing test**

`tests/test_detector_dbscan.py`:

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.dbscan import DBSCANDetector
from ml.features.engineer import build_training_frame


@pytest.fixture(scope="module")
def scaled():
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)


@pytest.fixture(scope="module")
def fitted(scaled):
    return DBSCANDetector(contamination=0.05, eps=3.0, min_samples=5).fit(scaled)


def test_flags_the_contamination_rate(fitted):
    assert fitted.fit_flags_.sum() == 126
    assert fitted.name == "dbscan"
    assert fitted.live_scorable is True


def test_core_samples_are_retained_for_serving(fitted, scaled):
    assert fitted.core_samples_.ndim == 2
    assert fitted.core_samples_.shape[1] == scaled.shape[1]
    assert fitted.core_samples_.shape[0] > 1000


def test_native_noise_rate_is_reported(fitted):
    assert 0.0 < fitted.native_noise_rate_ < 0.20
    assert fitted.n_clusters_ > 1


def test_calibrated_flags_are_a_subset_of_native_noise(fitted, scaled):
    """The calibrated flag tightens DBSCAN's own noise definition (spec 4.6)."""
    native_noise = fitted._model.labels_ == -1
    calibrated = fitted.fit_flags_ == 1
    assert calibrated.sum() <= native_noise.sum()
    assert np.all(native_noise[calibrated]), (
        "every calibrated flag must also be a native DBSCAN noise point"
    )


def test_scores_a_single_unseen_row(fitted, scaled):
    row = scaled.iloc[[0]]
    assert fitted.score(row).shape == (1,)
    assert fitted.flag(row).shape == (1,)


def test_score_is_a_non_negative_distance(fitted, scaled):
    assert (fitted.score(scaled) >= 0).all()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_detector_dbscan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.detectors.dbscan'`

- [ ] **Step 3: Write `ml/detectors/dbscan.py`**

```python
"""DBSCAN with nearest-core-sample scoring for unseen rows.

scikit-learn's DBSCAN exposes only fit_predict and cannot classify new points. The
core samples are retained at fit time and a new row is scored by its distance to
the nearest one, which is the natural extension of DBSCAN's own rule: a point
within reach of a core sample joins that cluster, a point far from every core
sample is noise (spec 4.6).

eps shapes the cluster structure; the anomaly rate is set by threshold transfer,
not by eps. native_noise_rate_ is reported so a badly chosen eps stays visible.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from ml.detectors.base import BaseDetector


class DBSCANDetector(BaseDetector):
    name = "dbscan"
    view = "full"
    scaler = "standard"
    live_scorable = True

    def __init__(
        self,
        contamination: float = 0.05,
        eps: float = 3.0,
        min_samples: int = 5,
    ) -> None:
        super().__init__(
            contamination=contamination, eps=eps, min_samples=min_samples
        )
        self.core_samples_: np.ndarray | None = None
        self.native_noise_rate_: float | None = None
        self.n_clusters_: int | None = None

    def _fit(self, X: np.ndarray) -> None:
        self._model = DBSCAN(
            eps=self.params["eps"],
            min_samples=self.params["min_samples"],
            n_jobs=-1,
        ).fit(X)

        labels = self._model.labels_
        if len(self._model.core_sample_indices_) == 0:
            raise ValueError(
                f"dbscan: eps={self.params['eps']} produced no core samples; "
                "increase eps or lower min_samples"
            )

        self.core_samples_ = X[self._model.core_sample_indices_]
        self.native_noise_rate_ = float((labels == -1).mean())
        self.n_clusters_ = int(len(set(labels)) - (1 if -1 in labels else 0))
        self._nn = NearestNeighbors(n_neighbors=1).fit(self.core_samples_)

    def _score(self, X: np.ndarray) -> np.ndarray:
        distances, _ = self._nn.kneighbors(X)
        return distances.ravel()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_detector_dbscan.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ml/detectors/dbscan.py tests/test_detector_dbscan.py
git commit -m "feat: add DBSCAN detector with nearest-core-sample live scoring"
```

---

### Task 9: The four training-time detectors

**Files:**
- Create: `ml/detectors/mcd.py`, `ml/detectors/gmm.py`, `ml/detectors/kmeans.py`, `ml/detectors/pca_reconstruction.py`
- Test: `tests/test_detectors_train_only.py`

**Interfaces:**
- Consumes: `ml.detectors.base.BaseDetector`
- Produces: `MCDDetector(contamination, random_state)` with `view="continuous"` and `scaler="continuous"`; `GMMDetector(contamination, n_components, covariance_type, random_state)`; `KMeansDetector(contamination, n_clusters, n_init, random_state)`; `PCAReconstructionDetector(contamination, n_components, random_state)`. All four have `live_scorable = False`.

- [ ] **Step 1: Write the failing test**

`tests/test_detectors_train_only.py`:

```python
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.gmm import GMMDetector
from ml.detectors.kmeans import KMeansDetector
from ml.detectors.mcd import MCDDetector
from ml.detectors.pca_reconstruction import PCAReconstructionDetector
from ml.features.engineer import CONTINUOUS_COLUMNS, build_training_frame


@pytest.fixture(scope="module")
def frames():
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    full = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    cont_raw = X[CONTINUOUS_COLUMNS]
    cont = pd.DataFrame(
        StandardScaler().fit_transform(cont_raw), columns=CONTINUOUS_COLUMNS
    )
    return full, cont


def test_all_four_are_marked_train_only():
    for cls in (MCDDetector, GMMDetector, KMeansDetector, PCAReconstructionDetector):
        assert cls.live_scorable is False, cls.__name__


def test_mcd_uses_the_continuous_view():
    assert MCDDetector.view == "continuous"
    assert MCDDetector.scaler == "continuous"


def test_mcd_fits_cleanly_on_the_continuous_view(frames):
    """On the full 19 columns MCD fails to converge (spec 2.2)."""
    _, cont = frames
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        det = MCDDetector(contamination=0.05, random_state=42).fit(cont)
    convergence = [w for w in caught if "Determinant has increased" in str(w.message)]
    assert convergence == [], "MCD must converge without determinant warnings"
    assert det.fit_flags_.sum() == 126


def test_gmm_converges_and_flags_the_rate(frames):
    full, _ = frames
    det = GMMDetector(
        contamination=0.05, n_components=5, covariance_type="full", random_state=42
    ).fit(full)
    assert det._model.converged_ is True
    assert det.fit_flags_.sum() == 126


def test_kmeans_flags_the_rate(frames):
    full, _ = frames
    det = KMeansDetector(
        contamination=0.05, n_clusters=8, n_init=10, random_state=42
    ).fit(full)
    assert det.fit_flags_.sum() == 126
    assert (det.score(full) >= 0).all()


def test_pca_reconstruction_flags_the_rate(frames):
    full, _ = frames
    det = PCAReconstructionDetector(
        contamination=0.05, n_components=0.95, random_state=42
    ).fit(full)
    assert det.fit_flags_.sum() == 126
    assert (det.score(full) >= 0).all()
    assert det._model.n_components_ < full.shape[1]


def test_all_four_score_a_single_row(frames):
    full, cont = frames
    pairs = [
        (MCDDetector(contamination=0.05, random_state=42), cont),
        (
            GMMDetector(
                contamination=0.05,
                n_components=5,
                covariance_type="full",
                random_state=42,
            ),
            full,
        ),
        (
            KMeansDetector(
                contamination=0.05, n_clusters=8, n_init=10, random_state=42
            ),
            full,
        ),
        (
            PCAReconstructionDetector(
                contamination=0.05, n_components=0.95, random_state=42
            ),
            full,
        ),
    ]
    for det, frame in pairs:
        det.fit(frame)
        assert det.score(frame.iloc[[0]]).shape == (1,), det.name
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_detectors_train_only.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.detectors.mcd'`

- [ ] **Step 3: Write `ml/detectors/mcd.py`**

```python
"""Minimum Covariance Determinant detector.

Fitted on the continuous view only. On the full 19-column matrix the covariance is
rank-deficient and, more fundamentally, LoginAttempts, DailyAccountVolume and
DailyDeviceVelocity are 95-98% single-valued and become exactly constant inside
MCD's central support subset, making that subset's covariance singular (spec 2.2).

Not live-scorable: it is fitted and reported, but takes no part in the vote.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import EllipticEnvelope

from ml.detectors.base import BaseDetector


class MCDDetector(BaseDetector):
    name = "mcd"
    view = "continuous"
    scaler = "continuous"
    live_scorable = False

    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        super().__init__(contamination=contamination, random_state=random_state)

    def _fit(self, X: np.ndarray) -> None:
        self._model = EllipticEnvelope(
            contamination=self.contamination,
            random_state=self.params["random_state"],
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # Mahalanobis distance is already higher for anomalies.
        return self._model.mahalanobis(X)
```

- [ ] **Step 4: Write `ml/detectors/gmm.py`**

```python
"""Gaussian Mixture detector: flags the lowest-log-likelihood rows.

Not live-scorable: fitted and reported, but takes no part in the vote.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from ml.detectors.base import BaseDetector


class GMMDetector(BaseDetector):
    name = "gmm"
    view = "full"
    scaler = "standard"
    live_scorable = False

    def __init__(
        self,
        contamination: float = 0.05,
        n_components: int = 5,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        super().__init__(
            contamination=contamination,
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray) -> None:
        self._model = GaussianMixture(
            n_components=self.params["n_components"],
            covariance_type=self.params["covariance_type"],
            random_state=self.params["random_state"],
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # score_samples is a log-likelihood, higher for typical rows, so negate.
        return -self._model.score_samples(X)
```

- [ ] **Step 5: Write `ml/detectors/kmeans.py`**

```python
"""K-Means detector: flags the largest distance to the nearest centroid.

Not live-scorable: fitted and reported, but takes no part in the vote.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ml.detectors.base import BaseDetector


class KMeansDetector(BaseDetector):
    name = "kmeans"
    view = "full"
    scaler = "standard"
    live_scorable = False

    def __init__(
        self,
        contamination: float = 0.05,
        n_clusters: int = 8,
        n_init: int = 10,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            contamination=contamination,
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray) -> None:
        self._model = KMeans(
            n_clusters=self.params["n_clusters"],
            n_init=self.params["n_init"],
            random_state=self.params["random_state"],
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # transform() returns distances to every centroid; take the nearest.
        return self._model.transform(X).min(axis=1)
```

- [ ] **Step 6: Write `ml/detectors/pca_reconstruction.py`**

```python
"""PCA reconstruction-error detector.

This PCA is fitted solely for reconstruction error and is persisted independently
of any PCA used for visualisation (spec 4.7).

Not live-scorable: fitted and reported, but takes no part in the vote.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ml.detectors.base import BaseDetector


class PCAReconstructionDetector(BaseDetector):
    name = "pca_reconstruction"
    view = "full"
    scaler = "standard"
    live_scorable = False

    def __init__(
        self,
        contamination: float = 0.05,
        n_components: float = 0.95,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            contamination=contamination,
            n_components=n_components,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray) -> None:
        self._model = PCA(
            n_components=self.params["n_components"],
            random_state=self.params["random_state"],
        ).fit(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        reconstructed = self._model.inverse_transform(self._model.transform(X))
        return ((X - reconstructed) ** 2).mean(axis=1)
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `python -m pytest tests/test_detectors_train_only.py -v`
Expected: 7 passed

- [ ] **Step 8: Commit**

```bash
git add ml/detectors/mcd.py ml/detectors/gmm.py ml/detectors/kmeans.py ml/detectors/pca_reconstruction.py tests/test_detectors_train_only.py
git commit -m "feat: add MCD, GMM, k-means and PCA-reconstruction detectors"
```

---

### Task 10: Detector registry and the voting ensemble

**Files:**
- Create: `ml/detectors/registry.py`, `ml/ensemble/voting.py`
- Test: `tests/test_registry.py`, `tests/test_voting.py`

**Interfaces:**
- Consumes: all eight detector classes, `ml.config.Config`
- Produces:
  - `build_detectors(cfg: Config) -> list[BaseDetector]` (8, in a fixed order)
  - `live_detectors(detectors: list[BaseDetector]) -> list[BaseDetector]` (4)
  - `EnsembleResult` dataclass with `is_anomaly: bool`, `votes_for: int`, `votes_total: int`, `votes_required: int`, `threshold: float`, and `as_dict() -> dict`
  - `votes_required(n_detectors: int, threshold: float) -> int`
  - `combine_one(flags: dict[str, int], threshold: float) -> EnsembleResult`
  - `combine_matrix(flags: dict[str, np.ndarray], threshold: float) -> tuple[np.ndarray, np.ndarray]` returning `(votes_for, is_anomaly)`

- [ ] **Step 1: Write the failing tests**

`tests/test_registry.py`:

```python
from ml.config import Config
from ml.detectors.registry import DETECTOR_ORDER, build_detectors, live_detectors


def test_registry_builds_eight_detectors():
    detectors = build_detectors(Config.load())
    assert len(detectors) == 8
    assert [d.name for d in detectors] == DETECTOR_ORDER


def test_exactly_four_are_live_scorable():
    live = live_detectors(build_detectors(Config.load()))
    assert [d.name for d in live] == [
        "isolation_forest",
        "lof",
        "one_class_svm",
        "dbscan",
    ]


def test_detectors_receive_config_hyperparameters():
    by_name = {d.name: d for d in build_detectors(Config.load())}
    assert by_name["dbscan"].params["eps"] == 3.0
    assert by_name["dbscan"].params["min_samples"] == 5
    assert by_name["lof"].params["n_neighbors"] == 20
    assert by_name["kmeans"].params["n_clusters"] == 8
    assert all(d.contamination == 0.05 for d in by_name.values())


def test_every_detector_declares_a_known_scaler():
    for det in build_detectors(Config.load()):
        assert det.scaler in {"standard", "robust", "continuous"}, det.name
```

`tests/test_voting.py`:

```python
import numpy as np
import pytest

from ml.ensemble.voting import (
    EnsembleResult,
    combine_matrix,
    combine_one,
    votes_required,
)


@pytest.mark.parametrize(
    "n,threshold,expected",
    [
        (4, 0.5, 2),
        (4, 0.75, 3),
        (4, 1.0, 4),
        (4, 0.25, 1),
        (4, 0.0, 1),
        (7, 0.5, 4),
        (3, 0.5, 2),
        (1, 0.5, 1),
    ],
)
def test_votes_required_uses_ceiling_with_a_floor_of_one(n, threshold, expected):
    assert votes_required(n, threshold) == expected


def test_combine_one_flags_at_the_threshold():
    result = combine_one(
        {"a": 1, "b": 1, "c": 0, "d": 0}, threshold=0.5
    )
    assert isinstance(result, EnsembleResult)
    assert result.is_anomaly is True
    assert result.votes_for == 2
    assert result.votes_total == 4
    assert result.votes_required == 2


def test_combine_one_does_not_flag_below_the_threshold():
    result = combine_one({"a": 1, "b": 0, "c": 0, "d": 0}, threshold=0.5)
    assert result.is_anomaly is False
    assert result.votes_for == 1


def test_combine_one_rejects_an_empty_roster():
    with pytest.raises(ValueError, match="at least one detector"):
        combine_one({}, threshold=0.5)


def test_as_dict_matches_the_result_contract():
    keys = set(combine_one({"a": 1, "b": 1}, threshold=0.5).as_dict())
    assert keys == {
        "is_anomaly",
        "votes_for",
        "votes_total",
        "votes_required",
        "threshold",
    }


def test_combine_matrix_matches_combine_one_row_by_row():
    flags = {
        "a": np.array([1, 0, 1, 0]),
        "b": np.array([1, 1, 0, 0]),
        "c": np.array([0, 1, 0, 0]),
        "d": np.array([0, 0, 0, 1]),
    }
    votes, is_anomaly = combine_matrix(flags, threshold=0.5)
    assert votes.tolist() == [2, 2, 1, 1]
    assert is_anomaly.tolist() == [1, 1, 0, 0]
    for i in range(4):
        single = combine_one({k: int(v[i]) for k, v in flags.items()}, 0.5)
        assert single.votes_for == votes[i]
        assert int(single.is_anomaly) == is_anomaly[i]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_registry.py tests/test_voting.py -v`
Expected: FAIL with `ModuleNotFoundError` for `ml.detectors.registry` and `ml.ensemble.voting`

- [ ] **Step 3: Write `ml/detectors/registry.py`**

```python
"""Builds the detector roster from config, in one place.

train.py and score.py both call this, so they cannot disagree about which
detectors exist or which of them vote.
"""
from __future__ import annotations

from ml.config import Config
from ml.detectors.base import BaseDetector
from ml.detectors.dbscan import DBSCANDetector
from ml.detectors.gmm import GMMDetector
from ml.detectors.isolation_forest import IsolationForestDetector
from ml.detectors.kmeans import KMeansDetector
from ml.detectors.lof import LOFDetector
from ml.detectors.mcd import MCDDetector
from ml.detectors.one_class_svm import OneClassSVMDetector
from ml.detectors.pca_reconstruction import PCAReconstructionDetector

DETECTOR_ORDER = [
    "isolation_forest",
    "lof",
    "one_class_svm",
    "dbscan",
    "mcd",
    "gmm",
    "kmeans",
    "pca_reconstruction",
]


def build_detectors(cfg: Config) -> list[BaseDetector]:
    """Instantiate all eight detectors with their configured hyperparameters."""
    contamination = cfg.get("detectors.contamination")
    random_state = cfg.get("detectors.random_state")

    return [
        IsolationForestDetector(
            contamination=contamination,
            n_estimators=cfg.get("detectors.isolation_forest.n_estimators"),
            random_state=random_state,
        ),
        LOFDetector(
            contamination=contamination,
            n_neighbors=cfg.get("detectors.lof.n_neighbors"),
        ),
        OneClassSVMDetector(
            contamination=contamination,
            kernel=cfg.get("detectors.one_class_svm.kernel"),
            gamma=cfg.get("detectors.one_class_svm.gamma"),
            nu=cfg.get("detectors.one_class_svm.nu"),
        ),
        DBSCANDetector(
            contamination=contamination,
            eps=cfg.get("detectors.dbscan.eps"),
            min_samples=cfg.get("detectors.dbscan.min_samples"),
        ),
        MCDDetector(contamination=contamination, random_state=random_state),
        GMMDetector(
            contamination=contamination,
            n_components=cfg.get("detectors.gmm.n_components"),
            covariance_type=cfg.get("detectors.gmm.covariance_type"),
            random_state=random_state,
        ),
        KMeansDetector(
            contamination=contamination,
            n_clusters=cfg.get("detectors.kmeans.n_clusters"),
            n_init=cfg.get("detectors.kmeans.n_init"),
            random_state=random_state,
        ),
        PCAReconstructionDetector(
            contamination=contamination,
            n_components=cfg.get("detectors.pca_reconstruction.n_components"),
            random_state=random_state,
        ),
    ]


def live_detectors(detectors: list[BaseDetector]) -> list[BaseDetector]:
    """The subset that can score an unseen row, and therefore the voting roster."""
    return [d for d in detectors if d.live_scorable]
```

- [ ] **Step 4: Write `ml/ensemble/voting.py`**

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_registry.py tests/test_voting.py -v`
Expected: 17 passed

- [ ] **Step 6: Commit**

```bash
git add ml/detectors/registry.py ml/ensemble/voting.py tests/test_registry.py tests/test_voting.py
git commit -m "feat: add detector registry and generic voting ensemble"
```

---

### Task 11: Artifact bundle save and load

**Files:**
- Create: `ml/storage/artifacts.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: `FeatureArtifacts`, `ProfileStore`, `BaseDetector`
- Produces: `ArtifactBundle` dataclass with fields `manifest: dict`, `scalers: dict[str, object]`, `feature_artifacts: FeatureArtifacts`, `profile_store: ProfileStore`, `detectors: dict[str, BaseDetector]`, `surrogate`, `explainer_state: dict`; plus `save_bundle(bundle: ArtifactBundle, dest: Path) -> None` and `load_bundle(src: Path) -> ArtifactBundle`

- [ ] **Step 1: Write the failing test**

`tests/test_artifacts.py`:

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.isolation_forest import IsolationForestDetector
from ml.features.engineer import build_training_frame
from ml.storage.artifacts import ArtifactBundle, load_bundle, save_bundle


@pytest.fixture(scope="module")
def bundle():
    X, artifacts, profiles = build_training_frame(
        load_raw(Config.load().get("data.csv_path"))
    )
    scaler = StandardScaler().fit(X)
    scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    det = IsolationForestDetector(
        contamination=0.05, n_estimators=200, random_state=42
    ).fit(scaled)
    return ArtifactBundle(
        manifest={"version": "test", "rate_table": {"isolation_forest": 0.0502}},
        scalers={"standard": scaler},
        feature_artifacts=artifacts,
        profile_store=profiles,
        detectors={"isolation_forest": det},
        surrogate=None,
        explainer_state={"feature_percentiles": {"TransactionAmount": [1.0, 2.0]}},
    )


def test_round_trip_preserves_the_manifest(tmp_path, bundle):
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    assert loaded.manifest == bundle.manifest


def test_round_trip_preserves_feature_artifacts(tmp_path, bundle):
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    assert loaded.feature_artifacts.feature_columns == (
        bundle.feature_artifacts.feature_columns
    )
    assert loaded.feature_artifacts.location_freq_default == (
        bundle.feature_artifacts.location_freq_default
    )
    assert loaded.feature_artifacts.time_since_last_tx_median == pytest.approx(
        bundle.feature_artifacts.time_since_last_tx_median
    )


def test_round_trip_preserves_the_profile_store(tmp_path, bundle):
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    assert loaded.profile_store.account_last_tx == bundle.profile_store.account_last_tx
    assert (
        loaded.profile_store.account_day_counts
        == bundle.profile_store.account_day_counts
    )


def test_round_trip_preserves_detector_thresholds(tmp_path, bundle):
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    original = bundle.detectors["isolation_forest"]
    restored = loaded.detectors["isolation_forest"]
    assert restored.threshold_ == pytest.approx(original.threshold_)
    assert np.array_equal(restored.train_scores_, original.train_scores_)
    assert np.array_equal(restored.fit_flags_, original.fit_flags_)


def test_restored_detector_scores_identically(tmp_path, bundle):
    X, _, _ = build_training_frame(load_raw(Config.load().get("data.csv_path")))
    scaled = pd.DataFrame(
        bundle.scalers["standard"].transform(X), columns=X.columns
    )
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    row = scaled.iloc[[0]]
    assert loaded.detectors["isolation_forest"].score(row) == pytest.approx(
        bundle.detectors["isolation_forest"].score(row)
    )


def test_manifest_is_human_readable_json(tmp_path, bundle):
    import json

    save_bundle(bundle, tmp_path)
    text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert json.loads(text)["version"] == "test"


def test_load_rejects_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "does-not-exist")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_artifacts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.storage.artifacts'`

- [ ] **Step 3: Write `ml/storage/artifacts.py`**

```python
"""Artifact bundle persistence.

Takes a filesystem path. Version directories are written whole and are treated as
immutable by readers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ml.features.engineer import FeatureArtifacts, ProfileStore

MANIFEST_NAME = "manifest.json"
DETECTOR_DIR = "detectors"


@dataclass
class ArtifactBundle:
    """Everything the scorer needs, loaded once."""

    manifest: dict
    scalers: dict[str, Any]
    feature_artifacts: FeatureArtifacts
    profile_store: ProfileStore
    detectors: dict[str, Any]
    surrogate: Any
    explainer_state: dict


def save_bundle(bundle: ArtifactBundle, dest: str | Path) -> None:
    """Write the bundle to `dest`, creating it if needed."""
    dest = Path(dest)
    (dest / DETECTOR_DIR).mkdir(parents=True, exist_ok=True)

    with open(dest / MANIFEST_NAME, "w", encoding="utf-8") as fh:
        json.dump(bundle.manifest, fh, indent=2, default=str)

    joblib.dump(bundle.scalers, dest / "scalers.pkl")
    joblib.dump(bundle.feature_artifacts, dest / "feature_artifacts.pkl")
    joblib.dump(bundle.profile_store, dest / "profile_store.pkl")
    joblib.dump(bundle.explainer_state, dest / "explainer_state.pkl")

    for name, detector in bundle.detectors.items():
        joblib.dump(detector, dest / DETECTOR_DIR / f"{name}.pkl")

    if bundle.surrogate is not None:
        bundle.surrogate.save_model(str(dest / "surrogate_xgb.json"))


def load_bundle(src: str | Path) -> ArtifactBundle:
    """Read a bundle written by save_bundle."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"artifact bundle not found: {src}")

    with open(src / MANIFEST_NAME, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    detectors = {
        path.stem: joblib.load(path)
        for path in sorted((src / DETECTOR_DIR).glob("*.pkl"))
    }

    surrogate = None
    surrogate_path = src / "surrogate_xgb.json"
    if surrogate_path.exists():
        from xgboost import XGBClassifier

        surrogate = XGBClassifier()
        surrogate.load_model(str(surrogate_path))

    return ArtifactBundle(
        manifest=manifest,
        scalers=joblib.load(src / "scalers.pkl"),
        feature_artifacts=joblib.load(src / "feature_artifacts.pkl"),
        profile_store=joblib.load(src / "profile_store.pkl"),
        detectors=detectors,
        surrogate=surrogate,
        explainer_state=joblib.load(src / "explainer_state.pkl"),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_artifacts.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add ml/storage/artifacts.py tests/test_artifacts.py
git commit -m "feat: add artifact bundle save and load"
```

---

### Task 12: XGBoost surrogate

**Files:**
- Create: `ml/explain/surrogate.py`
- Test: `tests/test_surrogate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks beyond a feature frame and a label array
- Produces: `SurrogateResult` dataclass with `model`, `auc: float`, `agreement: float`, `scale_pos_weight: float`; `train_surrogate(X: pd.DataFrame, y: np.ndarray, *, test_size: float, random_state: int) -> SurrogateResult`

- [ ] **Step 1: Write the failing test**

`tests/test_surrogate.py`:

```python
import numpy as np
import pandas as pd
import pytest

from ml.explain.surrogate import SurrogateResult, train_surrogate


@pytest.fixture(scope="module")
def separable():
    rng = np.random.default_rng(42)
    n = 2000
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    # A clean, learnable rule so fidelity should be high.
    y = ((X["a"] > 1.8) | (X["b"] > 2.0)).astype(int).to_numpy()
    return X, y


def test_returns_a_surrogate_result(separable):
    X, y = separable
    result = train_surrogate(X, y, test_size=0.25, random_state=42)
    assert isinstance(result, SurrogateResult)


def test_fidelity_is_high_on_a_learnable_label(separable):
    X, y = separable
    result = train_surrogate(X, y, test_size=0.25, random_state=42)
    # AUC is the fidelity measure that matters; scale_pos_weight deliberately
    # shifts the hard decision boundary, so agreement gets a looser floor.
    assert result.auc >= 0.95
    assert result.agreement >= 0.85


def test_scale_pos_weight_reflects_class_imbalance(separable):
    X, y = separable
    result = train_surrogate(X, y, test_size=0.25, random_state=42)
    expected = (len(y) - y.sum()) / y.sum()
    assert result.scale_pos_weight == pytest.approx(expected, rel=0.2)


def test_model_predicts_probabilities_for_a_single_row(separable):
    X, y = separable
    result = train_surrogate(X, y, test_size=0.25, random_state=42)
    proba = result.model.predict_proba(X.iloc[[0]])
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_is_deterministic(separable):
    X, y = separable
    a = train_surrogate(X, y, test_size=0.25, random_state=42)
    b = train_surrogate(X, y, test_size=0.25, random_state=42)
    assert a.auc == pytest.approx(b.auc)


def test_rejects_a_single_class_label(separable):
    X, _ = separable
    with pytest.raises(ValueError, match="both classes"):
        train_surrogate(X, np.zeros(len(X), dtype=int), test_size=0.25, random_state=42)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_surrogate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.explain.surrogate'`

- [ ] **Step 3: Write `ml/explain/surrogate.py`**

```python
"""XGBoost surrogate trained on the ensemble's decision.

The surrogate explains; it never decides. Its purpose is to make a single
transaction explainable via SHAP without attributing across four detectors.

It is therefore scored on FIDELITY to the ensemble - held-out AUC and agreement
against the ensemble label - not on detection accuracy. The labels are themselves
model output, so reporting "accuracy" against them would be a category error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


@dataclass
class SurrogateResult:
    model: Any
    auc: float
    agreement: float
    scale_pos_weight: float


def train_surrogate(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> SurrogateResult:
    """Fit the surrogate and measure how faithfully it reproduces the ensemble."""
    y = np.asarray(y, dtype=int)
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        raise ValueError(
            "train_surrogate needs both classes present in the ensemble label"
        )

    scale_pos_weight = float((len(y) - positives) / positives)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities))
    agreement = float((model.predict(X_test) == y_test).mean())

    # Refit on the full dataset so the shipped model has seen every row.
    final = XGBClassifier(**model.get_params())
    final.fit(X, y)

    return SurrogateResult(
        model=final,
        auc=auc,
        agreement=agreement,
        scale_pos_weight=scale_pos_weight,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_surrogate.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ml/explain/surrogate.py tests/test_surrogate.py
git commit -m "feat: add XGBoost surrogate with fidelity metrics"
```

---

### Task 13: SHAP explainer and plain-English rendering

**Files:**
- Create: `ml/explain/shap_explainer.py`
- Test: `tests/test_shap_explainer.py`

**Interfaces:**
- Consumes: a fitted XGBoost model, the training feature frame
- Produces:
  - `FEATURE_PHRASES: dict[str, str]`, `ONE_HOT_PHRASES: dict[str, str]`
  - `build_explainer_state(X_train: pd.DataFrame) -> dict`
  - `ShapExplainer(model, feature_columns: list[str], explainer_state: dict)` with `explain(row: pd.DataFrame, is_anomaly: bool, top_n: int = 3) -> dict` returning keys `top_features`, `plain_english`, `surrogate_probability`

- [ ] **Step 1: Write the failing test**

`tests/test_shap_explainer.py`:

```python
import numpy as np
import pandas as pd
import pytest

from ml.explain.shap_explainer import (
    FEATURE_PHRASES,
    ONE_HOT_PHRASES,
    ShapExplainer,
    build_explainer_state,
)
from ml.explain.surrogate import train_surrogate
from ml.features.engineer import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(42)
    n = 1500
    X = pd.DataFrame(
        {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    )
    X["UtilizationRatio"] = rng.uniform(0, 0.4, size=n)
    X.loc[: n // 20, "UtilizationRatio"] = rng.uniform(0.9, 1.5, size=n // 20 + 1)
    y = (X["UtilizationRatio"] > 0.8).astype(int).to_numpy()
    result = train_surrogate(X, y, test_size=0.25, random_state=42)
    state = build_explainer_state(X)
    return ShapExplainer(result.model, FEATURE_COLUMNS, state), X, y


def test_every_feature_column_has_a_phrase():
    for col in FEATURE_COLUMNS:
        assert col in FEATURE_PHRASES or col in ONE_HOT_PHRASES, col


def test_explain_returns_the_documented_keys(fitted):
    explainer, X, _ = fitted
    result = explainer.explain(X.iloc[[0]], is_anomaly=False)
    assert set(result) == {"top_features", "plain_english", "surrogate_probability"}


def test_top_features_have_the_documented_shape(fitted):
    explainer, X, _ = fitted
    result = explainer.explain(X.iloc[[0]], is_anomaly=False)
    assert 1 <= len(result["top_features"]) <= 3
    for item in result["top_features"]:
        assert set(item) == {
            "feature",
            "value",
            "shap_value",
            "direction",
            "percentile",
        }
        assert item["direction"] in {"increases", "decreases"}
        assert 0.0 <= item["percentile"] <= 100.0


def test_top_features_are_sorted_by_absolute_shap(fitted):
    explainer, X, y = fitted
    idx = int(np.argmax(y))
    result = explainer.explain(X.iloc[[idx]], is_anomaly=True)
    magnitudes = [abs(f["shap_value"]) for f in result["top_features"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_the_driving_feature_is_surfaced(fitted):
    explainer, X, y = fitted
    idx = int(np.argmax(y))
    result = explainer.explain(X.iloc[[idx]], is_anomaly=True)
    names = [f["feature"] for f in result["top_features"]]
    assert "UtilizationRatio" in names


def test_flagged_sentence_reads_as_a_flag(fitted):
    explainer, X, y = fitted
    idx = int(np.argmax(y))
    text = explainer.explain(X.iloc[[idx]], is_anomaly=True)["plain_english"]
    assert text.startswith("Flagged primarily due to")
    assert text.endswith(".")


def test_clean_sentence_uses_its_own_template(fitted):
    explainer, X, y = fitted
    idx = int(np.argmin(y))
    text = explainer.explain(X.iloc[[idx]], is_anomaly=False)["plain_english"]
    assert "No strong anomaly indicators" in text
    assert "Flagged" not in text


def test_surrogate_probability_is_a_probability(fitted):
    explainer, X, _ = fitted
    p = explainer.explain(X.iloc[[0]], is_anomaly=False)["surrogate_probability"]
    assert 0.0 <= p <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_shap_explainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.explain.shap_explainer'`

- [ ] **Step 3: Write `ml/explain/shap_explainer.py`**

```python
"""SHAP wrapper producing top features and a plain-English sentence.

"Unusually high" is a claim about the training distribution, so every magnitude
phrase is generated from a measured percentile rather than asserted.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

FEATURE_PHRASES: dict[str, str] = {
    "TransactionAmount": "transaction amount",
    "CustomerAge": "customer age",
    "TransactionDuration": "transaction duration",
    "LoginAttempts": "login attempts",
    "AccountBalance": "account balance",
    "TimeSinceLastTx_Hours": "gap since the account's last transaction",
    "DailyAccountVolume": "number of transactions on this account today",
    "UtilizationRatio": "share of the account balance drained",
    "DailyDeviceVelocity": "number of transactions from this device today",
    "Location_Freq": "how common this location is",
}

# One-hot features are facts, not magnitudes: "unusually high Channel_Online"
# would be meaningless.
ONE_HOT_PHRASES: dict[str, str] = {
    "TransactionType_Credit": "the transaction was a credit",
    "TransactionType_Debit": "the transaction was a debit",
    "Channel_ATM": "the transaction was made at an ATM",
    "Channel_Branch": "the transaction was made at a branch",
    "Channel_Online": "the transaction was made online",
    "CustomerOccupation_Doctor": "the customer is a doctor",
    "CustomerOccupation_Engineer": "the customer is an engineer",
    "CustomerOccupation_Retired": "the customer is retired",
    "CustomerOccupation_Student": "the customer is a student",
}

HIGH_PERCENTILE = 90.0
LOW_PERCENTILE = 10.0


def build_explainer_state(X_train: pd.DataFrame) -> dict:
    """Persist the sorted training values per column, for percentile lookups."""
    return {
        "feature_percentiles": {
            col: np.sort(X_train[col].to_numpy(dtype=float)).tolist()
            for col in X_train.columns
        }
    }


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


class ShapExplainer:
    """Explains a single row against the surrogate model."""

    def __init__(
        self,
        model: Any,
        feature_columns: list[str],
        explainer_state: dict,
    ) -> None:
        self.model = model
        self.feature_columns = list(feature_columns)
        self._percentiles = {
            col: np.asarray(values, dtype=float)
            for col, values in explainer_state["feature_percentiles"].items()
        }
        self._explainer = shap.TreeExplainer(model)

    def _percentile_of(self, column: str, value: float) -> float:
        reference = self._percentiles.get(column)
        if reference is None or len(reference) == 0:
            return 50.0
        position = float(np.searchsorted(reference, value, side="right"))
        return 100.0 * position / len(reference)

    def _phrase_for(self, column: str, value: float, percentile: float) -> str:
        if column in ONE_HOT_PHRASES:
            return ONE_HOT_PHRASES[column]
        noun = FEATURE_PHRASES.get(column, column)
        if percentile >= HIGH_PERCENTILE:
            return f"an unusually high {noun} ({percentile:.0f}th percentile)"
        if percentile <= LOW_PERCENTILE:
            return f"an unusually low {noun} ({percentile:.0f}th percentile)"
        return f"the {noun} ({percentile:.0f}th percentile)"

    def explain(
        self, row: pd.DataFrame, is_anomaly: bool, top_n: int = 3
    ) -> dict:
        """Return top contributing features and a rendered sentence."""
        frame = row[self.feature_columns]
        shap_values = np.asarray(self._explainer.shap_values(frame))
        if shap_values.ndim == 3:
            # Some SHAP versions return one matrix per class.
            shap_values = shap_values[:, :, -1]
        contributions = shap_values[0]

        order = np.argsort(np.abs(contributions))[::-1][:top_n]

        top_features = []
        phrases = []
        for idx in order:
            column = self.feature_columns[idx]
            value = float(frame.iloc[0][column])
            percentile = self._percentile_of(column, value)
            top_features.append(
                {
                    "feature": column,
                    "value": value,
                    "shap_value": float(contributions[idx]),
                    "direction": (
                        "increases" if contributions[idx] > 0 else "decreases"
                    ),
                    "percentile": percentile,
                }
            )
            phrases.append(self._phrase_for(column, value, percentile))

        if is_anomaly:
            sentence = f"Flagged primarily due to {_join(phrases[:2])}."
        else:
            sentence = (
                "No strong anomaly indicators. The closest contributors were "
                f"{_join(phrases[:2])}."
            )

        return {
            "top_features": top_features,
            "plain_english": sentence,
            "surrogate_probability": float(
                self.model.predict_proba(frame)[0, 1]
            ),
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_shap_explainer.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add ml/explain/shap_explainer.py tests/test_shap_explainer.py
git commit -m "feat: add SHAP explainer with plain-English rendering"
```

---

### Task 14: Training pipeline and the validation report

**Files:**
- Create: `ml/pipeline/train.py`
- Test: `tests/test_train_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1–13
- Produces:
  - `run_training(cfg: Config, dest: Path | None = None) -> TrainingReport`
  - `TrainingReport` dataclass with `rate_table: dict[str, float]`, `flag_counts: dict[str, int]`, `agreement: dict[str, float]`, `vote_histogram: dict[int, int]`, `ensemble_rate: float`, `ensemble_flagged: int`, `threshold_sweep: dict[int, int]`, `surrogate_auc: float`, `surrogate_agreement: float`, `dbscan_native_noise_rate: float`, `n_rows: int`, `warnings: list[str]`
  - `format_report(report: TrainingReport) -> str`
  - CLI entry point: `python -m ml.pipeline.train`

- [ ] **Step 1: Write the failing test**

`tests/test_train_pipeline.py`:

```python
import pytest

from ml.config import Config
from ml.pipeline.train import TrainingReport, format_report, run_training
from ml.storage.artifacts import load_bundle


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    dest = tmp_path_factory.mktemp("artifacts")
    report = run_training(Config.load(), dest=dest)
    return report, dest


def test_every_detector_flags_the_contamination_rate(trained):
    report, _ = trained
    assert len(report.rate_table) == 8
    for name, rate in report.rate_table.items():
        assert 0.045 <= rate <= 0.055, f"{name} flagged {rate:.4f}"
        assert report.flag_counts[name] == 126, name


def test_no_detector_falls_outside_the_sane_band(trained):
    report, _ = trained
    assert report.warnings == [], report.warnings


def test_ensemble_rate_is_in_the_target_band(trained):
    report, _ = trained
    assert 0.03 <= report.ensemble_rate <= 0.07
    assert 100 <= report.ensemble_flagged <= 160


def test_vote_histogram_covers_zero_through_four(trained):
    report, _ = trained
    assert set(report.vote_histogram) == {0, 1, 2, 3, 4}
    assert sum(report.vote_histogram.values()) == 2512


def test_threshold_sweep_is_monotonically_decreasing(trained):
    report, _ = trained
    counts = [report.threshold_sweep[k] for k in sorted(report.threshold_sweep)]
    assert counts == sorted(counts, reverse=True)


def test_pairwise_agreement_covers_every_live_pair(trained):
    report, _ = trained
    assert len(report.agreement) == 6
    for pair, jaccard in report.agreement.items():
        assert 0.0 <= jaccard <= 1.0, pair


def test_dbscan_native_noise_rate_is_reported(trained):
    report, _ = trained
    assert 0.0 < report.dbscan_native_noise_rate < 0.20


def test_surrogate_fidelity_meets_the_configured_floor(trained):
    report, _ = trained
    assert report.surrogate_auc >= Config.load().get("surrogate.min_auc")


def test_bundle_is_written_and_loadable(trained):
    _, dest = trained
    bundle = load_bundle(dest)
    assert len(bundle.detectors) == 8
    assert set(bundle.scalers) == {"standard", "robust", "continuous"}
    assert bundle.surrogate is not None
    assert bundle.manifest["rate_table"]
    assert bundle.feature_artifacts.feature_columns


def test_report_formats_without_error(trained):
    report, _ = trained
    text = format_report(report)
    assert "PER-DETECTOR ANOMALY RATES" in text
    assert "isolation_forest" in text
    assert "VOTE HISTOGRAM" in text


def test_training_is_deterministic(tmp_path):
    a = run_training(Config.load(), dest=tmp_path / "a")
    b = run_training(Config.load(), dest=tmp_path / "b")
    assert a.rate_table == b.rate_table
    assert a.vote_histogram == b.vote_histogram
    assert a.ensemble_flagged == b.ensemble_flagged
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_train_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.pipeline.train'`

- [ ] **Step 3: Write `ml/pipeline/train.py`**

```python
"""Offline training run: fit everything, report, and write the artifact bundle."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_train_pipeline.py -v`
Expected: 11 passed

- [ ] **Step 5: Run the training CLI and read the tables**

Run: `python -m ml.pipeline.train`
Expected: the four validation tables print, every detector shows 126 rows at 5.02%, the ensemble lands between 3% and 7%, and no warnings appear.

- [ ] **Step 6: Commit**

```bash
git add ml/pipeline/train.py tests/test_train_pipeline.py
git commit -m "feat: add training pipeline with validation report"
```

---

### Task 15: `score_transaction()`

**Files:**
- Create: `ml/pipeline/score.py`
- Test: `tests/test_score.py`

**Interfaces:**
- Consumes: everything from Tasks 1–14
- Produces:
  - `Scorer(bundle: ArtifactBundle, threshold: float)` with `score_transaction(raw_txn: dict) -> dict`
  - `get_scorer(artifact_dir: str | Path | None = None) -> Scorer` (loads once, module-level singleton)
  - `score_transaction(raw_txn: dict) -> dict` (module-level convenience)
  - `reset_scorer() -> None` (test hook)

- [ ] **Step 1: Write the failing test**

`tests/test_score.py`:

```python
import pytest

from ml.config import Config
from ml.pipeline.score import Scorer
from ml.pipeline.train import run_training
from ml.storage.artifacts import load_bundle


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    dest = tmp_path_factory.mktemp("artifacts")
    run_training(Config.load(), dest=dest)
    return dest


@pytest.fixture
def scorer(artifact_dir):
    """Function-scoped on purpose.

    score_transaction() mutates the profile store, so a module-scoped scorer
    would let repeated calls drive DailyAccountVolume far above its training
    maximum of 2 and make the assertions order-dependent. Training runs once;
    each test gets a fresh profile store.
    """
    return Scorer(
        load_bundle(artifact_dir), threshold=Config.load().get("ensemble.threshold")
    )


NORMAL_TXN = {
    "TransactionID": "TX900001",
    "AccountID": "AC00128",
    "DeviceID": "D000380",
    "Location": "San Diego",
    "TransactionDate": "2023-06-15 14:30:00",
    "TransactionAmount": 120.00,
    "AccountBalance": 8000.00,
    "CustomerAge": 45,
    "TransactionDuration": 90,
    "LoginAttempts": 1,
    "TransactionType": "Debit",
    "Channel": "ATM",
    "CustomerOccupation": "Engineer",
}

WEIRD_TXN = {
    **NORMAL_TXN,
    "TransactionID": "TX900002",
    "TransactionAmount": 4800.00,
    "AccountBalance": 5000.00,
    "LoginAttempts": 5,
    "Channel": "Online",
}


def test_result_matches_the_documented_contract(scorer):
    result = scorer.score_transaction(NORMAL_TXN)
    assert set(result) == {
        "transaction_id",
        "scored_at",
        "ensemble",
        "detectors",
        "explanation",
        "features",
        "raw",
        "warnings",
    }


def test_ensemble_block_has_the_documented_keys(scorer):
    ensemble = scorer.score_transaction(NORMAL_TXN)["ensemble"]
    assert set(ensemble) == {
        "is_anomaly",
        "votes_for",
        "votes_total",
        "votes_required",
        "threshold",
    }
    assert ensemble["votes_total"] == 4
    assert ensemble["votes_required"] == 2


def test_only_live_detectors_appear(scorer):
    detectors = scorer.score_transaction(NORMAL_TXN)["detectors"]
    assert len(detectors) == 4
    assert [d["name"] for d in detectors] == [
        "isolation_forest",
        "lof",
        "one_class_svm",
        "dbscan",
    ]
    for entry in detectors:
        assert set(entry) == {
            "name",
            "flag",
            "score",
            "score_percentile",
            "live_scored",
        }
        assert entry["live_scored"] is True
        assert entry["flag"] in (0, 1)
        assert 0.0 <= entry["score_percentile"] <= 100.0


def test_train_only_detectors_are_absent(scorer):
    names = {d["name"] for d in scorer.score_transaction(NORMAL_TXN)["detectors"]}
    assert names.isdisjoint({"mcd", "gmm", "kmeans", "pca_reconstruction"})


def test_explanation_block_has_the_documented_keys(scorer):
    explanation = scorer.score_transaction(NORMAL_TXN)["explanation"]
    assert set(explanation) == {
        "top_features",
        "plain_english",
        "surrogate_probability",
    }
    assert explanation["plain_english"]


def test_votes_for_matches_the_detector_flags(scorer):
    result = scorer.score_transaction(WEIRD_TXN)
    assert result["ensemble"]["votes_for"] == sum(
        d["flag"] for d in result["detectors"]
    )


def test_an_obviously_weird_transaction_is_flagged(scorer):
    """A 96% account drain with five login attempts must be caught."""
    result = scorer.score_transaction(WEIRD_TXN)
    assert result["ensemble"]["is_anomaly"] is True
    assert result["ensemble"]["votes_for"] >= 2


def test_a_normal_transaction_is_not_flagged(scorer):
    """The clean result is what makes the flagged ones credible."""
    result = scorer.score_transaction(NORMAL_TXN)
    assert result["ensemble"]["is_anomaly"] is False


def test_features_block_carries_all_nineteen(scorer):
    features = scorer.score_transaction(NORMAL_TXN)["features"]
    assert len(features) == 19
    assert features["UtilizationRatio"] == pytest.approx(120.0 / 8000.0)


def test_unseen_account_is_scored_with_a_warning(scorer):
    txn = {**NORMAL_TXN, "AccountID": "AC99999"}
    result = scorer.score_transaction(txn)
    assert result["ensemble"]["votes_total"] == 4
    assert any("unseen account" in w for w in result["warnings"])


def test_profile_store_updates_between_calls(scorer):
    txn = {
        **NORMAL_TXN,
        "AccountID": "AC77777",
        "DeviceID": "D77777",
        "TransactionDate": "2023-07-01 09:00:00",
    }
    first = scorer.score_transaction(txn)
    second = scorer.score_transaction(
        {**txn, "TransactionDate": "2023-07-01 09:05:00"}
    )
    assert first["features"]["DailyAccountVolume"] == 1
    assert second["features"]["DailyAccountVolume"] == 2
    assert second["features"]["TimeSinceLastTx_Hours"] == pytest.approx(5 / 60, rel=1e-6)


def test_scored_at_is_iso_8601(scorer):
    from datetime import datetime

    scored_at = scorer.score_transaction(NORMAL_TXN)["scored_at"]
    assert datetime.fromisoformat(scored_at)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.pipeline.score'`

- [ ] **Step 3: Write `ml/pipeline/score.py`**

```python
"""The single scoring path.

Feature engineering, the live detectors, the ensemble vote, the profile update and
the SHAP explanation all happen here, so training and serving never duplicate
logic. Artifacts load once into a module-level singleton, not per call.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.config import Config
from ml.ensemble.voting import combine_one
from ml.explain.shap_explainer import ShapExplainer
from ml.features.engineer import transform_one
from ml.storage.artifacts import ArtifactBundle, load_bundle


class Scorer:
    """Holds the loaded bundle and scores one transaction at a time."""

    def __init__(self, bundle: ArtifactBundle, threshold: float) -> None:
        self.bundle = bundle
        self.threshold = float(threshold)
        self.artifacts = bundle.feature_artifacts
        self.profiles = bundle.profile_store

        # Only detectors that can score an unseen row take part. load_bundle
        # returns them alphabetically, so impose the reporting order here.
        order = ["isolation_forest", "lof", "one_class_svm", "dbscan"]
        self.live = sorted(
            (d for d in bundle.detectors.values() if d.live_scorable),
            key=lambda d: order.index(d.name),
        )

        self.explainer = ShapExplainer(
            bundle.surrogate,
            self.artifacts.feature_columns,
            bundle.explainer_state,
        )

    def score_transaction(self, raw_txn: dict) -> dict:
        """Engineer, score, vote, explain, and record the transaction."""
        frame, warnings = transform_one(raw_txn, self.artifacts, self.profiles)

        detector_entries = []
        flags: dict[str, int] = {}
        for detector in self.live:
            scaled = pd.DataFrame(
                self.bundle.scalers[detector.scaler].transform(frame),
                columns=frame.columns,
            )
            score = float(detector.score(scaled)[0])
            flag = int(detector.flag(scaled)[0])
            flags[detector.name] = flag
            detector_entries.append(
                {
                    "name": detector.name,
                    "flag": flag,
                    "score": score,
                    "score_percentile": float(detector.score_percentile(score)),
                    "live_scored": True,
                }
            )

        ensemble = combine_one(flags, self.threshold)
        explanation = self.explainer.explain(frame, ensemble.is_anomaly)

        # Record the transaction so subsequent scores see it as history.
        self.profiles.observe(
            str(raw_txn["AccountID"]),
            str(raw_txn["DeviceID"]),
            pd.Timestamp(raw_txn["TransactionDate"]),
        )

        return {
            "transaction_id": str(raw_txn.get("TransactionID", "")),
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "ensemble": ensemble.as_dict(),
            "detectors": detector_entries,
            "explanation": explanation,
            "features": {
                col: float(frame.iloc[0][col]) for col in frame.columns
            },
            "raw": dict(raw_txn),
            "warnings": warnings,
        }


_SCORER: Scorer | None = None


def get_scorer(artifact_dir: str | Path | None = None) -> Scorer:
    """Return the process-wide scorer, loading the bundle on first use."""
    global _SCORER
    if _SCORER is None:
        cfg = Config.load()
        source = Path(artifact_dir or cfg.get("storage.local_dir"))
        _SCORER = Scorer(load_bundle(source), cfg.get("ensemble.threshold"))
    return _SCORER


def reset_scorer() -> None:
    """Drop the singleton. Used by tests."""
    global _SCORER
    _SCORER = None


def score_transaction(raw_txn: dict) -> dict:
    """Score one transaction using the process-wide scorer."""
    return get_scorer().score_transaction(raw_txn)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_score.py -v`
Expected: 12 passed

If `test_a_normal_transaction_is_not_flagged` fails, do not weaken the test. Print the four detector scores and percentiles for `NORMAL_TXN`, and check the scaler being applied matches the one each detector was fitted with — a mismatched scaler is the most likely cause.

- [ ] **Step 5: Commit**

```bash
git add ml/pipeline/score.py tests/test_score.py
git commit -m "feat: add score_transaction as the single scoring path"
```

---

### Task 16: Acceptance — full suite and the §6 tables

**Files:**
- Create: `tests/test_protocol_conformance.py`
- Create: `tests/test_acceptance.py`
- Create: `ml/README.md`

**Interfaces:**
- Consumes: everything
- Produces: nothing new

- [ ] **Step 1: Write the Protocol conformance test**

Spec §14 requires this parametrised across all eight detectors rather than checked
per detector. It runs against the detectors actually persisted in the bundle.

`tests/test_protocol_conformance.py`:

```python
"""Spec 14: Protocol conformance, parametrised across all eight detectors."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

from ml.config import Config
from ml.data.loader import load_raw
from ml.detectors.base import AnomalyDetector
from ml.detectors.registry import DETECTOR_ORDER, build_detectors
from ml.features.engineer import build_training_frame


@pytest.fixture(scope="module")
def fitted_detectors():
    cfg = Config.load()
    X, artifacts, _ = build_training_frame(load_raw(cfg.get("data.csv_path")))
    frames = {
        "standard": pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        ),
        "robust": pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns),
        "continuous": pd.DataFrame(
            StandardScaler().fit_transform(X[artifacts.continuous_columns]),
            columns=artifacts.continuous_columns,
        ),
    }
    fitted = {}
    for detector in build_detectors(cfg):
        frame = frames[detector.scaler]
        fitted[detector.name] = (detector.fit(frame), frame)
    return fitted


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_satisfies_the_protocol(fitted_detectors, name):
    detector, _ = fitted_detectors[name]
    assert isinstance(detector, AnomalyDetector)
    assert detector.name == name
    assert detector.view in {"full", "continuous"}
    assert isinstance(detector.live_scorable, bool)


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_fit_returns_self(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    assert detector.fit(frame) is detector


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_score_and_flag_shapes(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    scores = detector.score(frame)
    flags = detector.flag(frame)
    assert scores.shape == (len(frame),)
    assert flags.shape == (len(frame),)
    assert set(np.unique(flags)).issubset({0, 1})


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_flag_rate_matches_contamination_within_one_row(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    expected = detector.contamination * len(frame)
    assert abs(detector.fit_flags_.sum() - expected) <= 1.0


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_flag_agrees_with_thresholding_score(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    expected = (detector.score(frame) >= detector.live_threshold_).astype(int)
    assert np.array_equal(detector.flag(frame), expected)


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_scores_a_single_row(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    assert detector.score(frame.iloc[[0]]).shape == (1,)


@pytest.mark.parametrize("name", DETECTOR_ORDER)
def test_score_percentile_is_bounded(fitted_detectors, name):
    detector, frame = fitted_detectors[name]
    value = float(detector.score(frame.iloc[[0]])[0])
    assert 0.0 <= detector.score_percentile(value) <= 100.0
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_protocol_conformance.py -v`
Expected: 56 passed (7 tests across 8 detectors)

- [ ] **Step 3: Write the acceptance test**

`tests/test_acceptance.py`:

```python
"""Plan A acceptance: the spec 6 tables and a standalone score_transaction call."""
import pytest

from ml.config import Config
from ml.pipeline.score import Scorer
from ml.pipeline.train import format_report, run_training
from ml.storage.artifacts import load_bundle

LIVE = ["isolation_forest", "lof", "one_class_svm", "dbscan"]
TRAIN_ONLY = ["mcd", "gmm", "kmeans", "pca_reconstruction"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    dest = tmp_path_factory.mktemp("acceptance")
    report = run_training(Config.load(), dest=dest)
    print("\n" + format_report(report))
    return report, dest


def test_spec_6_1_all_eight_detectors_flag_126_rows(trained):
    report, _ = trained
    assert sorted(report.rate_table) == sorted(LIVE + TRAIN_ONLY)
    for name in LIVE + TRAIN_ONLY:
        assert report.flag_counts[name] == 126, f"{name} flagged {report.flag_counts[name]}"


def test_spec_6_2_live_pairwise_agreement_is_moderate(trained):
    report, _ = trained
    for pair, jaccard in report.agreement.items():
        assert 0.10 <= jaccard <= 0.60, f"{pair} Jaccard {jaccard:.3f}"


def test_spec_6_3_vote_histogram_is_dominated_by_zero_votes(trained):
    report, _ = trained
    assert report.vote_histogram[0] / report.n_rows > 0.80
    assert report.vote_histogram[4] > 0, "some rows should be unanimous"


def test_spec_6_3_ensemble_rate_is_in_the_target_band(trained):
    report, _ = trained
    assert 100 <= report.ensemble_flagged <= 160
    assert 0.03 <= report.ensemble_rate <= 0.07


def test_spec_6_4_flagged_rows_have_higher_utilization(trained):
    """The ensemble must be finding drained accounts, not noise."""
    import pandas as pd

    from ml.data.loader import load_raw
    from ml.detectors.registry import live_detectors
    from ml.ensemble.voting import combine_matrix
    from ml.features.engineer import build_training_frame

    _, dest = trained
    bundle = load_bundle(dest)
    raw = load_raw(Config.load().get("data.csv_path"))
    X, _, _ = build_training_frame(raw)

    flags = {
        d.name: d.fit_flags_
        for d in live_detectors(list(bundle.detectors.values()))
    }
    _, labels = combine_matrix(flags, Config.load().get("ensemble.threshold"))

    flagged = X.loc[labels == 1, "UtilizationRatio"].mean()
    normal = X.loc[labels == 0, "UtilizationRatio"].mean()
    assert flagged > normal * 3, f"flagged {flagged:.3f} vs normal {normal:.3f}"


def test_surrogate_meets_the_configured_fidelity_floor(trained):
    report, _ = trained
    assert report.surrogate_auc >= Config.load().get("surrogate.min_auc")


def test_score_transaction_works_standalone(trained):
    """Plan A's terminating condition: no API, just the function."""
    _, dest = trained
    scorer = Scorer(load_bundle(dest), Config.load().get("ensemble.threshold"))
    result = scorer.score_transaction(
        {
            "TransactionID": "TX999999",
            "AccountID": "AC00128",
            "DeviceID": "D000380",
            "Location": "San Diego",
            "TransactionDate": "2023-08-01 03:14:00",
            "TransactionAmount": 4800.00,
            "AccountBalance": 5000.00,
            "CustomerAge": 24,
            "TransactionDuration": 12,
            "LoginAttempts": 5,
            "TransactionType": "Debit",
            "Channel": "Online",
            "CustomerOccupation": "Student",
        }
    )
    assert result["ensemble"]["is_anomaly"] is True
    assert result["ensemble"]["votes_total"] == 4
    assert len(result["detectors"]) == 4
    assert result["explanation"]["plain_english"].startswith("Flagged primarily due to")
    print("\nplain english:", result["explanation"]["plain_english"])
```

- [ ] **Step 4: Run the acceptance test**

Run: `python -m pytest tests/test_acceptance.py -v -s`
Expected: 7 passed, with the validation tables and the plain-English sentence printed.

- [ ] **Step 5: Run the entire suite**

Run: `python -m pytest -v`
Expected: all tests pass across every `tests/test_*.py` file.

- [ ] **Step 6: Confirm the notebook was never touched**

Run: `git status --porcelain final.ipynb`
Expected: empty output. If anything is listed, revert it with `git checkout -- final.ipynb`.

- [ ] **Step 7: Write `ml/README.md`**

```markdown
# ml

Anomaly detection over the transaction dataset.

## Train

    python -m ml.pipeline.train

Fits eight detectors, builds the ensemble and the surrogate, prints the
validation tables, and writes the artifact bundle to the directory named by
`storage.local_dir` in `config.yaml`.

## Score

    from ml.pipeline.score import score_transaction

    result = score_transaction({
        "TransactionID": "TX000001",
        "AccountID": "AC00128",
        "DeviceID": "D000380",
        "Location": "San Diego",
        "TransactionDate": "2023-08-01 03:14:00",
        "TransactionAmount": 4800.00,
        "AccountBalance": 5000.00,
        "CustomerAge": 24,
        "TransactionDuration": 12,
        "LoginAttempts": 5,
        "TransactionType": "Debit",
        "Channel": "Online",
        "CustomerOccupation": "Student",
    })

`score_transaction()` is the single scoring path. Feature engineering, the four
live detectors, the ensemble vote and the SHAP explanation all run inside it.

## Detectors

Four vote: Isolation Forest, LOF, One-Class SVM, DBSCAN.
Four are fitted and reported but do not vote: MCD, GMM, K-Means,
PCA-reconstruction.

## Test

    python -m pytest
```

- [ ] **Step 8: Commit**

```bash
git add tests/test_protocol_conformance.py tests/test_acceptance.py ml/README.md
git commit -m "test: add protocol conformance and Plan A acceptance suites"
```

---

## Acceptance Criteria

Plan A is complete when all of the following hold:

1. `python -m pytest` passes with no failures, including the eight-way Protocol conformance suite.
2. `python -m ml.pipeline.train` prints all four validation tables with no warnings.
3. All eight detectors flag exactly **126 rows (5.02%)**.
4. The ≥2-of-4 ensemble flags **between 100 and 160 rows (4.0%–6.4%)**. Spec §6.3 records 126; the running-count change to the daily-count features (see Design Decisions) may move this by a few rows, so the acceptance test uses a band rather than an exact figure.
5. Live pairwise Jaccard falls between 0.10 and 0.60.
6. Surrogate held-out AUC ≥ `surrogate.min_auc` (0.95).
7. `score_transaction()` returns every key in the spec §8.1 contract, with exactly four detector entries.
8. An obviously weird transaction is flagged; a normal one is not.
9. `git status --porcelain final.ipynb` is empty.
