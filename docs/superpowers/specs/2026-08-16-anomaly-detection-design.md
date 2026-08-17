# Transaction Anomaly Detection — Design

**Date:** 2026-08-16
**Status:** Approved for implementation planning

## 1. Scope

A modular anomaly detection system over a 2,512-row synthetic banking transaction
dataset, comprising three deliverables:

1. A Python package (`/ml`) that trains eight anomaly detectors, combines four of
   them into a voting ensemble, and explains any single transaction via a
   SHAP-explainable surrogate model.
2. A FastAPI backend (`/backend`) deployed to Cloud Run, serving scoring endpoints
   and a scored-transaction feed.
3. A React demo dashboard (`/dashboard`) with a live transaction table and a
   transaction injection form.

The dashboard is the only frontend. There is no marketing site.

## 2. Source data

`original.csv`, 2,512 rows, 16 columns, no missing values, no duplicate rows.

### 2.1 Known defects in the source data

Three properties of the dataset materially affect the design and are handled
explicitly rather than absorbed silently.

**`PreviousTransactionDate` is not a previous-transaction timestamp.** Its full
range spans six minutes on 2024-11-04, while `TransactionDate` spans all of 2023.
It is an export or ingest timestamp. The derived
`TimeSinceLastTx_Hours = TransactionDate - PreviousTransactionDate` is negative for
all 2,512 rows (range -16,120h to -7,382h) and is a deterministic linear function
of `TransactionDate`, carrying no velocity signal.

Two consequences make this fatal rather than cosmetic. A plain-English explanation
citing "a short gap since the account's last transaction" would be describing a
quantity the feature cannot express. And at serving time a transaction dated `now()`
produces a *positive* value around +15,000h — far outside the training distribution —
so every injected demo transaction would be flagged substantially because of its
timestamp.

**Resolution:** `PreviousTransactionDate` is dropped as a feature source.
`TimeSinceLastTx_Hours` is recomputed as the gap to that account's own previous
transaction, by sorting on `(AccountID, TransactionDate)` and differencing. This
yields a genuine velocity feature: median 936.22h, minimum 0.01h, with 495 nulls
for the first transaction of each account.

**`DailyAccountVolume` and `DailyDeviceVelocity` are near-constant.** Modal share is
97.1% and 98.1% respectively; both take only the values 1 and 2. They are retained —
Isolation Forest's flagged rows do show a real if small lift (1.246 vs 1.018) — but
their weakness is recorded here so that no downstream analysis overstates them.

**`LoginAttempts` has 95.1% modal share.** Retained; it is one of the strongest
signals in the flagged population (1.87 vs 1.09) despite the skew.

### 2.2 Feature ledger

The engineered matrix is 19 columns, derived as follows:

| Stage | Columns | Operation |
|---|---|---|
| 0 | 16 | Raw CSV |
| 1 | 21 | Add 4 engineered features and the `TransactionDay` helper |
| 2 | 13 | Drop 8 columns |
| 3 | 13 | `Location` (43 cities) replaced by `Location_Freq` |
| 4 | **19** | One-hot 3 categorical columns into 9 dummies (13 − 3 + 9) |

**Eight raw columns survive as features:** `TransactionAmount`, `TransactionType`,
`Channel`, `CustomerAge`, `CustomerOccupation`, `TransactionDuration`,
`LoginAttempts`, `AccountBalance`.

**Five engineered features:** `TimeSinceLastTx_Hours`, `DailyAccountVolume`,
`UtilizationRatio`, `DailyDeviceVelocity`, `Location_Freq`.

**Eight dropped columns.** Most are consumed rather than discarded:

| Column | Fate |
|---|---|
| `AccountID` | Groupby key for `DailyAccountVolume` and the per-account gap |
| `DeviceID` | Groupby key for `DailyDeviceVelocity` |
| `TransactionDate` | Consumed into `TimeSinceLastTx_Hours` and the day key |
| `PreviousTransactionDate` | Dropped entirely (see 2.1) |
| `Location` | Replaced by `Location_Freq` |
| `TransactionID` | Discarded — row identifier |
| `IP Address` | Discarded — high cardinality, unused |
| `MerchantID` | Discarded — 100 distinct values, unused |

The resulting matrix has rank 16 of 19, because the one-hot groups each sum to 1
(`Credit + Debit = 1`; 3 channels sum to 1; 4 occupations sum to 1), giving a
condition number around 2×10^16.

This is harmless for the tree-based, distance-based and reconstruction-based
detectors in the roster, which are insensitive to collinear columns. GMM is the one
parametric estimator among them, so it was verified directly on the full 19-column
matrix rather than assumed: it reports `converged=True` with zero warnings and a
5.02% flag rate. Full dummies are therefore retained across every detector on this
view,
because a dropped-baseline encoding would make SHAP output less legible — a Credit
transaction would have to be read as "Debit = 0".

Rank deficiency is not universally harmless, and it is the reason MCD cannot be
fitted on this matrix. `EllipticEnvelope` on the full 19 columns warns that the
covariance is not full rank and then fails to converge across roughly sixty
iterations with "Determinant has increased; this should not happen". Restoring full
rank does not fix it either, because `LoginAttempts`, `DailyAccountVolume` and
`DailyDeviceVelocity` are 95–98% single-valued and become exactly constant inside
MCD's central support subset, making that subset's covariance singular regardless of
the rank of the whole matrix. MCD is therefore fitted on the continuous view defined
in §3.2, where it converges cleanly. This is recorded so the decision is not
revisited without the evidence.

## 3. Feature layer

`ml/features/engineer.py` exposes exactly two entry points, both producing the same
19 columns in the same frozen order:

```python
build_training_frame(df_raw) -> (X, FeatureArtifacts, ProfileStore)
transform_one(raw_txn: dict, artifacts, profiles) -> pd.DataFrame  # shape (1, 19)
```

Column order is itself a persisted artifact, so drift between training and serving
fails loudly instead of silently misaligning values.

### 3.1 Persisted feature state

```python
FeatureArtifacts:
    location_freq: dict[str, int]        # city -> training frequency
    location_freq_default: int           # for a city unseen in training
    feature_columns: list[str]           # the full view: the 19, ordered
    continuous_columns: list[str]        # the continuous view: the 7, ordered
    categorical_levels: dict[str, list[str]]

ProfileStore:
    account_last_tx: dict[str, datetime]
    account_day_counts: dict[tuple[str, date], int]
    device_day_counts: dict[tuple[str, date], int]
```

`ProfileStore` exists because three of the five engineered features require history
that a single incoming transaction does not carry. It is built during training,
saved with the model artifacts, loaded into memory at serving time, and updated
in-process after each score.

### 3.2 Column views

Detectors consume one of two views of the same engineered frame. Each detector
declares which via a `view` attribute (§4.1), so `train.py` and `score.py` stay
generic and never branch per model.

| View | Columns | Consumers |
|---|---|---|
| `full` | 19 — the matrix in §2.2 | Isolation Forest, LOF, One-Class SVM, DBSCAN, GMM, K-Means, PCA-reconstruction |
| `continuous` | 7 — `TransactionAmount`, `CustomerAge`, `TransactionDuration`, `AccountBalance`, `TimeSinceLastTx_Hours`, `UtilizationRatio`, `Location_Freq` | MCD |

The continuous view exists solely because MCD is a Gaussian elliptical estimator and
cannot be fitted on binary dummies and 95–98% single-valued counters (§2.2). It is
rank 7 of 7 with a condition number of 1.90.

The continuous view is a strict column subset of the full view, so it needs no
separate engineering step — `build_training_frame` returns the 19 columns and MCD
selects its 7 by name. There remains one feature-engineering code path, and
`transform_one` is unaffected.

### 3.3 Serving-time feature rules

| Feature | Live computation |
|---|---|
| `TimeSinceLastTx_Hours` | `txn_date - profiles.account_last_tx[acct]`, then bounded to the training range (see below); training median (936.22h) if the account is unseen |
| `DailyAccountVolume` | `account_day_counts.get((acct, day), 0) + 1` |
| `DailyDeviceVelocity` | `device_day_counts.get((device, day), 0) + 1` |
| `UtilizationRatio` | `amount / balance` — computable from the row alone |
| `Location_Freq` | `location_freq.get(city, location_freq_default)` |
| One-hots | Fixed from `categorical_levels`; an unseen level yields all-zeros for its group |

Daily counts are self-inclusive, so a lone injected transaction receives 1 —
identical to how a lone training row receives 1. Unseen categorical levels produce
an all-zero group rather than an error, and are reported in the result payload so
the condition is visible rather than silent.

### 3.4 `TimeSinceLastTx_Hours` must be bounded on both sides

The raw serving-time subtraction can produce values on either side of the training
range, and in both cases the effect is the failure §2.1 exists to prevent: the
transaction is flagged for its **timestamp** rather than its **content**. Training
gaps span 0.01h to 7,512.07h, so `FeatureArtifacts` persists both
`time_since_last_tx_median` and `time_since_last_tx_max`, and `transform_one`
clamps outside that range, appending a warning in each case.

**Below zero** — the transaction predates the account's last known activity, i.e.
out-of-order arrival. Clamp to the **median**. There is no meaningful gap to report,
and the median is the neutral value already used for an unseen account.

**Above the training maximum** — most commonly a transaction dated *now* against an
account whose history ends when the dataset does. A transaction dated today produces
roughly 24,000h against a 7,512h maximum: 3.2× beyond anything the detectors have
seen. Left unclamped this pins LOF, One-Class SVM and DBSCAN to the 100th percentile
and flags an entirely ordinary transaction. Clamp to the **maximum**, not the median:
a long dormancy is a genuine signal — dormant-account reactivation is real fraud
behaviour — and clamping to the median would erase it. Clamping to the observed
maximum preserves the meaning "as dormant as anything in training" without letting
an artefact of the dataset's age dominate every distance-based detector.

Both bounds matter for the demonstration specifically: §12 requires a preset that
returns clean, and without the upper bound no transaction dated today against a known
account can return clean.

## 4. Detector layer

### 4.1 Protocol

```python
class AnomalyDetector(Protocol):
    name: str
    view: Literal["full", "continuous"]   # which column view it consumes (§3.2)
    live_scorable: bool
    def fit(self, X: pd.DataFrame) -> "AnomalyDetector": ...
    def score(self, X: pd.DataFrame) -> np.ndarray:   # higher = more anomalous
    def flag(self, X: pd.DataFrame) -> np.ndarray:    # 0/1 per row
```

### 4.2 Threshold transfer

`flag()` is implemented once in a shared base class. At fit time each detector scores
its own training data and stores:

- `threshold_` — the `(1 - contamination)` percentile of those scores
- `train_scores_` — the sorted training scores, for percentile reporting

Then `flag(X) = (score(X) >= threshold_).astype(int)`.

This is the mechanism that makes the ensemble code generic. Four detectors expose a
native `predict()` (Isolation Forest, LOF, One-Class SVM, MCD) and four do not
(DBSCAN, GMM, K-Means, PCA-reconstruction); threshold transfer erases that
distinction. It
also guarantees each detector flags exactly the contamination rate on the training
set, and judges new rows against a *frozen* threshold, which is the correct
live-scoring semantics.

### 4.3 Roster and sign conventions

All scores are normalised so that higher always means more anomalous.

| Detector | Live | View | Underlying quantity | `score()` |
|---|---|---|---|---|
| `isolation_forest` | yes | full | `decision_function` (higher = normal) | `-decision_function` |
| `lof` | yes | full | `decision_function` (higher = normal) | `-decision_function` |
| `one_class_svm` | yes | full | `decision_function` (higher = normal) | `-decision_function` |
| `dbscan` | yes | full | distance to nearest core sample | as-is |
| `mcd` | no | continuous | `mahalanobis` (higher = anomalous) | as-is |
| `gmm` | no | full | `score_samples` log-likelihood (higher = normal) | `-score_samples` |
| `kmeans` | no | full | `transform().min(axis=1)` | as-is |
| `pca_reconstruction` | no | full | MSE of `inverse_transform(transform(X))` vs `X` | as-is |

Eight detectors are implemented and fitted. Four are live-scorable and form the
voting ensemble. MCD, GMM, K-Means and PCA-reconstruction are fitted, reported in the
training rate table, and carry `live_scorable = False`; they are excluded from the
live scoring path and from the vote.

MCD is the only detector on the `continuous` view, for the reason given in §2.2. It
is not live-scorable, so its view never has to be constructed on the serving path —
`transform_one` builds the full view only.

### 4.4 Scaling

`StandardScaler` for Isolation Forest, One-Class SVM, DBSCAN, GMM, K-Means and
PCA-reconstruction. `RobustScaler` for LOF. A separate `StandardScaler` is fitted on
the continuous view for MCD, since a scaler fitted on 19 columns cannot transform 7.
All three scalers are fitted during training and persisted.

The training matrix must be frozen before any detector runs. In the source notebook,
LOF was fitted on a frame that already contained Isolation Forest's output columns,
so LOF was training on another detector's predictions. The package builds the feature
matrix once and passes it read-only to every detector.

### 4.5 LOF is two fitted objects

`novelty=False` and `novelty=True` are not interchangeable in scikit-learn: the
former exposes `negative_outlier_factor_` but has no `predict()`; the latter has
`predict()` but its training-data scores are a different quantity. Both are fitted
on the same data and both are persisted:

- The `novelty=False` fit produces training labels, the rate-table entry, and the
  training score distribution.
- The `novelty=True` fit serves live scoring, and is calibrated independently by
  running its own `score_samples` over the training set to derive its threshold.

Two thresholds, both persisted, and the reason is documented in `ml/detectors/lof.py`.

### 4.6 DBSCAN live scoring

scikit-learn's DBSCAN exposes only `fit_predict` and cannot classify unseen points.
It is made live-scorable by storing `components_` (the core samples) at fit time and
scoring a new row by its distance to the nearest core sample, via a `NearestNeighbors`
index. This is the natural extension of DBSCAN's own rule: a point within reach of a
core sample joins that cluster, and a point far from every core sample is noise.

Validated at `eps=3.0, min_samples=5`: 2,269 core samples, native noise rate 6.45%.
Threshold transfer at the 95th percentile of nearest-core distance flags 126 rows
(5.02%), and **all 126 are a subset of DBSCAN's own 162 native-noise points** — so
the calibrated flag is a tightening of DBSCAN's noise definition, not a substitute
for it. Single-row scoring costs approximately 8ms.

`eps` requires care and is a config parameter. DBSCAN's native noise rate swings from
1.31% to 32.40% across `eps` in [4.0, 1.5], and scikit-learn's default `eps=0.5` would
classify substantially the entire dataset as noise. Threshold transfer contains this
risk — the anomaly rate is set by the percentile, not by `eps` — but `train.py`
reports the native noise rate alongside the calibrated rate so a badly chosen `eps`
is visible rather than masked.

### 4.7 Detector hyperparameters

All live in `config.yaml`.

| Detector | Parameters |
|---|---|
| Isolation Forest | `n_estimators=200`, `contamination=0.05`, `random_state=42` |
| LOF | `n_neighbors=20`, `contamination=0.05` |
| One-Class SVM | `kernel=rbf`, `gamma=scale`, `nu=0.05` |
| DBSCAN | `eps=3.0`, `min_samples=5` |
| MCD | `contamination=0.05`, `random_state=42` |
| GMM | `n_components=5`, `covariance_type=full`, `random_state=42` |
| K-Means | `n_clusters=8`, `n_init=10`, `random_state=42` |
| PCA-reconstruction | `n_components=0.95` (variance retained) |

The PCA fitted here for reconstruction error is a separate fit from any PCA used for
visualisation, and is persisted independently.

## 5. Ensemble

`ml/ensemble/voting.py` combines the live-scorable detectors generically:

```
votes_total    = number of live-scorable detectors        # 4
votes_required = ceil(votes_total * threshold)            # ceil(4 * 0.5) = 2
is_anomaly     = votes_for >= votes_required
```

`threshold` defaults to 0.5 in `config.yaml`. Nothing is hardcoded per detector; the
combiner reads the roster and computes the requirement.

Per-detector votes are retained alongside the final decision, since the dashboard
presents "N / 4 models agree" as part of the explanation.

## 6. Validation results

Measured on the full dataset with the design as specified above.

### 6.1 Per-detector anomaly rates (n = 2,512)

| Detector | Flagged | Rate | Role |
|---|---|---|---|
| Isolation Forest | 126 | 5.02% | live |
| LOF | 126 | 5.02% | live |
| One-Class SVM | 126 | 5.02% | live |
| DBSCAN | 126 | 5.02% | live |
| MCD | 126 | 5.02% | train-only |
| GMM | 126 | 5.02% | train-only |
| K-Means | 126 | 5.02% | train-only |
| PCA-reconstruction | 126 | 5.02% | train-only |

Every detector sits at the contamination rate by construction. No detector
misbehaves.

### 6.2 Pairwise agreement among the four live detectors

| Pair | Overlap | Jaccard |
|---|---|---|
| Isolation Forest / One-Class SVM | 66 | 0.355 |
| Isolation Forest / DBSCAN | 67 | 0.362 |
| Isolation Forest / LOF | 41 | 0.194 |
| LOF / One-Class SVM | 46 | 0.223 |
| LOF / DBSCAN | 44 | 0.212 |
| One-Class SVM / DBSCAN | 70 | 0.385 |

Agreement in the 0.19–0.39 range: the detectors find materially different things,
which is what makes the vote worth taking, but are not independent of each other.

### 6.3 Vote distribution and ensemble rate

| Votes | Rows | Share |
|---|---|---|
| 0 / 4 | 2,225 | 88.57% |
| 1 / 4 | 161 | 6.41% |
| 2 / 4 | 61 | 2.43% |
| 3 / 4 | 39 | 1.55% |
| 4 / 4 | 26 | 1.04% |

| Threshold | Rows | Rate |
|---|---|---|
| ≥1 of 4 (0.25) | 287 | 11.43% |
| **≥2 of 4 (0.50, default)** | **126** | **5.02%** |
| ≥3 of 4 (0.75) | 65 | 2.59% |
| ≥4 of 4 (1.00) | 26 | 1.04% |

The default threshold lands at 5.02%, within the 3–7% target band, with no tuning
required.

### 6.4 What the ensemble flags

| Feature | Normal | Flagged |
|---|---|---|
| `UtilizationRatio` | 0.16 | 0.91 |
| `TransactionAmount` | $284.25 | $550.21 |
| `LoginAttempts` | 1.09 | 1.87 |
| `AccountBalance` | $5,124.19 | $4,927.06 |
| `TimeSinceLastTx_Hours` | 1,227.27 | 1,414.27 |
| `CustomerAge` | 44.65 | 45.06 |

Drained balances, larger amounts and repeated logins. `TimeSinceLastTx_Hours` and
`CustomerAge` barely separate; the rebuilt gap feature earns its place by being
correct at serving time rather than by being a strong training-time discriminator,
and this is stated so the explanation layer does not overclaim it.

The 126 ensemble positives give the surrogate a 5.0% positive class.

## 7. Explainability

The surrogate explains; it never decides. The ensemble owns the verdict.

**`ml/explain/surrogate.py`** trains an `XGBClassifier` on the 19 features against
the ensemble's binary label, with `scale_pos_weight ≈ 18.9` for the 126/2,386 class
balance. It is evaluated on **fidelity to the ensemble** — held-out AUC and agreement
rate against the ensemble label — not on detection accuracy, which would be a
category error given the labels are themselves model output.

**`ml/explain/shap_explainer.py`** wraps `shap.TreeExplainer(xgb_model)` and exposes:

```python
explain(transaction: dict) -> {
    "top_features": [
        {"feature": str, "value": float, "shap_value": float,
         "direction": "increases" | "decreases", "percentile": float}
    ],
    "plain_english": str,
}
```

`percentile` ranks the feature's value against its training distribution. Any claim
that a value is "unusually high" is a claim about that distribution, so the sentence
is generated from a measured percentile rather than asserted.

A feature-to-phrase table renders the copy: `UtilizationRatio` becomes "share of the
account balance drained", `LoginAttempts` becomes "repeated login attempts",
`TimeSinceLastTx_Hours` becomes "gap since the account's last transaction". One-hot
features are phrased as facts ("the transaction was made online") rather than as
magnitudes, because "unusually high Channel_Online" is meaningless. The top 2–3
features by absolute SHAP value compose the sentence. The not-flagged case uses its
own template rather than a negation of the flagged one.

## 8. Scoring pipeline

`ml/pipeline/score.py` exposes the single function that the API, the dashboard and
any notebook testing all call:

```python
score_transaction(raw_txn: dict) -> dict
```

Feature engineering, the four live detectors, the ensemble vote, the profile update
and the SHAP explanation all happen inside it. There is one code path; training and
serving do not duplicate logic.

Artifacts load once into a module-level singleton, not per request.

### 8.1 Result contract

This is the only structure the API and dashboard may depend on.

```python
{
  "transaction_id": str,
  "scored_at": str,                     # ISO 8601
  "ensemble": {
      "is_anomaly": bool,
      "votes_for": int,
      "votes_total": int,               # 4
      "votes_required": int,            # 2
      "threshold": float                # 0.5
  },
  "detectors": [                        # 4 entries, live-scorable only
      {"name": str, "flag": int, "score": float,
       "score_percentile": float, "live_scored": bool}
  ],
  "explanation": {
      "top_features": [...],
      "plain_english": str,
      "surrogate_probability": float
  },
  "features": {...},                    # engineered values
  "raw": {...},                         # as submitted
  "warnings": [str]                     # e.g. unseen categorical level
}
```

Raw detector scores are mutually incomparable — Isolation Forest's is roughly ±0.2,
GMM's log-likelihood roughly −100 to 20, K-Means distance roughly 0 to 10. Each
detector therefore persists its sorted training score distribution and reports
`score_percentile` against it, so the breakdown panel can present a comparable
figure rather than four unrelated floats.

### 8.2 Training pipeline

`ml/pipeline/train.py` orchestrates the offline run: load, engineer, fit both scalers,
fit all eight detectors, compute the ensemble label, train and evaluate the surrogate,
build the SHAP explainer, and write the artifact bundle.

It reports, to stdout and into `manifest.json`:

- the per-detector rate table (§6.1)
- DBSCAN's native noise rate alongside its calibrated rate
- the pairwise agreement matrix (§6.2)
- the vote histogram and the ensemble rate at each threshold (§6.3)
- surrogate held-out AUC and agreement against the ensemble label

A detector whose rate falls outside a configurable sane band (default 3–7%) raises a
warning naming the detector and its rate.

## 9. Artifact storage

`ml/storage/` holds artifact save and load behind one interface with a local
filesystem implementation and a Cloud Storage implementation, because `train.py` and
`score.py` both need artifact I/O and duplicating it invites drift. This module is
not in the original directory sketch and is added for that reason.

Bundle layout, written to `gs://{bucket}/artifacts/{version}/`:

```
manifest.json           config hash, training timestamp, rate table, versions
scaler_standard.pkl
scaler_robust.pkl
scaler_continuous.pkl
feature_artifacts.pkl
profile_store.pkl
detectors/{name}.pkl    8 files, each carrying threshold_ and train_scores_.
                        lof.pkl carries both fitted objects (§4.5) and both
                        thresholds; dbscan.pkl carries the core samples and the
                        NearestNeighbors index built over them (§4.6)
surrogate_xgb.json
```

Artifacts are written to a single reusable directory rather than immutable
versioned ones with a `latest` pointer, which an earlier draft specified. `train.py`
writes flat, and `save_bundle` clears stale detector pickles before writing so a
changed roster cannot leave orphans that `load_bundle`'s glob picks back up. The
torn-bundle risk versioning was meant to address is real but does not arise in the
demonstration flow, where the upload completes before the service is deployed.

**Library versions are pinned exactly** in `requirements.txt`, and the container
installs from it. The detectors are joblib pickles, and a scikit-learn minor-version
difference between the training environment and the container breaks unpickling.

Expected bundle size is 2–5MB, dominated by DBSCAN's 2,269 core samples and LOF's
retained training array.

## 10. Backend

FastAPI, `/backend`. Artifacts load in a `lifespan` startup handler.

| Method | Path | Behaviour |
|---|---|---|
| POST | `/score` | Full transaction → result contract, persisted to Firestore |
| POST | `/batch-score` | List in, list of results out |
| GET | `/transactions/recent?limit=` | Firestore feed, ordered by `scored_at` descending |
| POST | `/demo/inject` | Partial transaction, remaining fields filled from a preset, delegates to `score_transaction()` |
| GET | `/demo/presets` | Serves the demo scenarios |
| GET | `/health` | Cloud Run probe |

`/demo/inject` is a convenience wrapper, not a second scoring path — it exists so the
form can submit six fields instead of sixteen. Presets are served from the backend so
scenarios can be tuned without rebuilding the React app.

CORS is configured for the dashboard origin. The service is deployed
`--allow-unauthenticated` because the dashboard calls it directly from a browser.

**Firestore** holds one collection, `scored_transactions`, one document per result,
matching the result contract. Ordering on a single field is auto-indexed, so no
composite index is required. `backend/seed.py` seeds approximately 200 scored
historical transactions so the dashboard table is populated on first load.

**Seeding deliberately does not live in `train.py`.** Putting it there would make
offline training import Firestore, breaking the training tests on any machine
without the `google-cloud` libraries and coupling training to a service it
otherwise has no need of. The seed script scores through the same
`score_transaction()` path, so the outcome is identical. It must also swap in a
fresh `ProfileStore` for the replay: scoring historical rows against the bundle's
already-populated store flags every one of them, because each row then sees its
own account's later activity as history.

## 11. Deployment

Cloud Run, container built by Cloud Build and pushed to Artifact Registry.
The deployed service is configured entirely by `ANOMALY_*` environment variables set
on the Cloud Run service (see `backend/config.py`), and the README's first step sets
the corresponding shell variables. `config.yaml` carries no GCP keys: an earlier
draft specified them, but nothing read them, and duplicating deployment facts across
a YAML file, the container environment and the build config invites drift.

**Cold start is the principal operational risk.** Container pull, importing
scikit-learn, XGBoost and SHAP, downloading and unpickling the bundle, and
initialising `TreeExplainer` together take an estimated 8–20 seconds. A first request
of that latency during a live demonstration is unacceptable, so the service is
deployed with `--min-instances=1`. The README documents the single command to enable
it before a demonstration and the command to return it to zero afterwards.

**`--max-instances=1` is also set.** The `ProfileStore` is mutated in-process, so
multiple instances would develop divergent views of recent account and device
activity. Pinning to one instance keeps the profile store coherent. This is a
deliberate constraint of the demonstration deployment and is documented as such in
the README rather than left implicit.

**Cost.** With one warm instance at 1 vCPU and 2 GiB (raised from 1 GiB: scikit-learn,
XGBoost and SHAP together need the headroom), the dominant line is idle
instance time at roughly $1.50–2.00 per week, plus a few cents for Firestore reads
under dashboard polling, Cloud Storage, and Artifact Registry image storage. Total
demo-level usage is on the order of $2–3 per week with `--min-instances=1`, and under
$1 per week at zero. The README states these as approximate and directs the reader to
verify against current published pricing.

The README contains the complete `gcloud` sequence from an empty project: enabling
APIs, creating the bucket and Artifact Registry repository, provisioning Firestore,
running training, uploading artifacts, building, deploying, and the min-instances
commands.

## 12. Dashboard

React with Vite, dark palette. Three components: the polling transaction table, the
expanded detail panel, and the injection form.

**The injection form is built first and receives the most attention.** It loads
scenarios from `/demo/presets`, and on submit optimistically prepends the returned
row rather than waiting for the next poll, so the result renders immediately.
Scoring is approximately 50ms — four detectors plus SHAP — so the round trip
dominates and the one-to-two-second target is comfortable.

The table polls `/transactions/recent` and colour-codes each row by
`votes_for / votes_total`, which is the quantity the verdict actually rests on.

**Expanding a flagged row** reveals:

- the verdict, as "N / 4 live-scored models agree"
- the per-detector breakdown for those four, each with its `score_percentile`
- a horizontal SHAP bar chart, signed, distinguishing contributions that push toward
  anomaly from those that push toward normal
- the plain-English sentence

MCD, GMM, K-Means and PCA-reconstruction are named in a muted footnote as
training-time detectors that do not vote, so the number 4 is explained on screen.

**Presets** include an account-drain scenario (amount 4,800 against a 5,000 balance,
utilization 0.96), a credential-stuffing scenario (5 login attempts, Online channel),
a rapid-fire scenario that injects twice against the same account to exercise the
profile store, and **a normal transaction that returns clean**. The last is not
optional: a demonstration in which every input is flagged demonstrates nothing, and
the clean result is what makes the flagged ones credible.

## 13. Repository structure

```
/ml
  config.yaml                     contamination, voting threshold, feature list,
                                  detector hyperparameters, GCP paths, sane band
  data/loader.py                  load and clean the raw CSV
  features/engineer.py            build_training_frame, transform_one
  detectors/
    base.py                       Protocol and threshold-transfer base class
    isolation_forest.py
    lof.py                        dual novelty=False / novelty=True fits
    one_class_svm.py
    dbscan.py                     nearest-core-sample live scoring
    mcd.py                        continuous view, live_scorable = False
    gmm.py                        live_scorable = False
    kmeans.py                     live_scorable = False
    pca_reconstruction.py         live_scorable = False
  ensemble/voting.py              generic majority-vote combiner
  explain/
    surrogate.py                  XGBoost trained on ensemble labels
    shap_explainer.py             SHAP wrapper, top features, plain English
  pipeline/
    train.py                      full offline run, writes artifacts and reports
    score.py                      score_transaction()
  storage/                        artifact save/load, local and GCS

/backend
  main.py
  routers/score.py
  routers/demo.py
  Dockerfile
  cloudbuild.yaml
  requirements.txt

/dashboard                        Vite React app

README.md                         gcloud sequence and cost estimate
final.ipynb                       read-only reference, never modified
original.csv                      source dataset
```

**`final.ipynb` is read-only for the duration of this project.** No cell is added,
edited, removed or re-executed, and its outputs are left as they are. It is retained
solely as the exploratory data analysis record and as a reference for the analysis
that informed §2.1 and §2.2.

Everything in `/ml`, `/backend` and `/dashboard` is built from scratch against this
specification. No code is copied out of the notebook — the feature engineering,
Isolation Forest and LOF implementations there are superseded by §3 and §4, which
differ from the notebook in ways that matter: `TimeSinceLastTx_Hours` is recomputed
per account (§2.1), the feature matrix is frozen before any detector runs so LOF no
longer trains on Isolation Forest's output columns (§4.4), and every detector flags
via threshold transfer rather than its native `predict()` (§4.2).

## 14. Testing

**Train/serve parity is the critical test.** `transform_one` applied to a training
row, with the profile store rewound to that row's state, must reproduce that row's
output from `build_training_frame` exactly. Every serving bug this design is
vulnerable to shows up here first.

- **Protocol conformance**, parametrised across all eight detectors: `fit` returns
  self, `score` and `flag` return the right shapes, `flag` on training data matches
  the contamination rate to within one row, and `flag` agrees with thresholding
  `score`.
- **Threshold transfer**: a detector's flag rate on its own training data equals
  `contamination` within tolerance.
- **Ensemble**: vote counting and the `ceil(n * threshold)` arithmetic, including
  boundary cases at thresholds 0, 0.5 and 1.
- **LOF dual fit**: both objects are persisted, both thresholds are distinct and
  present, and the novelty copy scores a single unseen row.
- **DBSCAN live scoring**: calibrated flags on training data are a subset of native
  noise labels, reproducing the validated property in §4.6.
- **Surrogate fidelity**: held-out AUC against the ensemble label exceeds
  `surrogate.min_auc` in config, default 0.95.
- **Result contract**: `score_transaction` on a hand-crafted transaction returns every
  documented key with the documented types.
- **Unseen inputs**: an unseen city, an unseen categorical level and an unseen account
  each produce a scored result plus a populated `warnings` entry, not an exception.
- **API**: `TestClient` coverage of every endpoint against a fixture artifact bundle.
- **Determinism**: two training runs with the same seed produce identical rate tables.

## 15. Build order

This design is implemented as three sequenced plans, each reviewable on its own. The
result contract in §8.1 is fixed here precisely so plans B and C are not guessing at
shapes produced by plan A.

**Plan A — ML core**

1. Feature layer and detectors, validated against the rates in §6.
2. Ensemble, surrogate and SHAP explanation, tested on known-anomalous rows.
3. `score_transaction()`, confirmed standalone on a hand-crafted transaction with no
   API present.

**Plan B — Backend and deployment**

4. FastAPI wrapper, containerised, deployed to Cloud Run with the README sequence.

**Plan C — Dashboard**

5. Built against the deployed API, injection form first.

One deviation from the original brief is deliberate and has been confirmed. The brief
placed "complete the missing detectors in the notebook, verify the anomaly rates, then
refactor" first. The rates in §6 have already been measured and are recorded above, so
building the detectors in the notebook and then moving them into the package would
produce a second copy of the modelling code for no additional information.

Plan A therefore builds them in the package directly, `final.ipynb` is not touched
(§13), and `train.py` reproduces the §6 tables as its own output — the same
verification gate, applied to the code that actually ships.
