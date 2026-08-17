# Transaction anomaly detection

An unsupervised anomaly-detection service for card transactions. Eight
detectors (Isolation Forest, LOF, One-Class SVM, DBSCAN, MCD, GMM, K-Means,
PCA-reconstruction) are fit offline on historical transactions; four of them
(Isolation Forest, LOF, One-Class SVM, DBSCAN) vote at serving time, and a
SHAP-explained surrogate model turns the vote into a plain-English sentence.
A FastAPI service wraps the scorer with a small HTTP surface: score a
transaction, batch-score several, read back the recent feed, and inject one
of four demo presets for a live walkthrough.

Everything — training, serving, and the demo presets — runs from a single
scoring path (`ml/pipeline/score.py`'s `Scorer.score_transaction`), so the
API can never drift from what was trained.

## Local quickstart

Install dependencies, then train, run, and score:

```bash
pip install -r requirements.txt

# Train: fits all eight detectors and writes the artifact bundle to artifacts/
python -m ml.pipeline.train

# Run: serves on http://localhost:8000 using the local artifact directory
# and an in-memory transaction log (both are the default; no env vars needed)
uvicorn backend.main:app --reload

# Score: POST a transaction and get back detector votes, an ensemble
# verdict, and a plain-English explanation
curl -s -X POST http://localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{
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
    "CustomerOccupation": "Student"
  }'
```

Or skip constructing a payload by hand and fire one of the four built-in
demo presets (`normal`, `account_drain`, `credential_stuffing`,
`rapid_fire` — see `backend/presets.py`):

```bash
curl -s -X POST http://localhost:8000/demo/inject \
  -H 'Content-Type: application/json' -d '{"preset": "account_drain"}'
```

Run the test suite (no cloud credentials or network access required):

```bash
python -m pytest
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/score` | Score one transaction |
| POST | `/batch-score` | Score up to 500 transactions |
| GET | `/transactions/recent` | Read back the scored-transaction feed, newest first |
| POST | `/demo/inject` | Score a named preset (with optional field overrides) |
| GET | `/demo/presets` | List the available demo presets |
| GET | `/health` | Liveness/readiness probe |

## Deploying to Cloud Run

The service reads its configuration from environment variables (prefixed
`ANOMALY_`; see `backend/config.py`) so the same code runs locally against
the filesystem and an in-memory log, or in Cloud Run against Cloud Storage
and Firestore, with no code changes.

Starting from an empty GCP project:

```bash
# 1. Set these once
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export BUCKET="gs://${PROJECT_ID}-anomaly-artifacts"
export REPO="anomaly"
export SERVICE="anomaly-api"

gcloud config set project "${PROJECT_ID}"

# 2. Enable the APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  firestore.googleapis.com

# 3. Create the artifact bucket
gcloud storage buckets create "${BUCKET}" --location="${REGION}"

# 4. Create the container repository
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location="${REGION}"

# 5. Provision Firestore in Native mode
gcloud firestore databases create --location="${REGION}"

# 6. Train locally and upload the bundle
python -m ml.pipeline.train
gcloud storage cp -r artifacts/* "${BUCKET}/artifacts/latest/"

# 7. Build and deploy
gcloud builds submit --config backend/cloudbuild.yaml \
  --substitutions=_REGION="${REGION}",_REPO="${REPO}",_SERVICE="${SERVICE}",_BUCKET="${BUCKET#gs://}"

# 8. Seed the feed so the dashboard is not empty
SERVICE_URL=$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')
curl -s -X POST "${SERVICE_URL}/demo/inject" -H 'Content-Type: application/json' -d '{"preset":"normal"}'
```

`backend/cloudbuild.yaml` builds the image from `backend/Dockerfile`, pushes
it to Artifact Registry, and deploys it to Cloud Run with
`ANOMALY_ARTIFACT_SOURCE=gcs`, `ANOMALY_GCS_BUCKET`, and
`ANOMALY_TRANSACTION_LOG=firestore` set, so the deployed service reads the
bundle uploaded in step 6 and writes scored transactions to Firestore.

`backend/seed.py` (`python -m backend.seed`) can also populate the feed with
a batch of real historical transactions rather than a single preset; point
it at the deployed bucket and Firestore collection with the same `ANOMALY_*`
environment variables the service itself uses.

### Min-instances toggle

Cloud Run scales to zero by default, which costs nothing while idle but
costs an 8-20s cold start on the next request — while a container starts,
downloads the ~7MB artifact bundle, and unpickles eight detectors. Before a
live demonstration, pin an instance so the first request is instant; scale
back down afterwards:

```bash
# Before a demonstration — removes the 8-20s cold start
gcloud run services update "${SERVICE}" --region "${REGION}" --min-instances=1
# Afterwards — back to scale-to-zero
gcloud run services update "${SERVICE}" --region "${REGION}" --min-instances=0
```

`backend/cloudbuild.yaml` also sets `--max-instances=1`. The profile store
(the per-account, per-device history behind `DailyAccountVolume`,
`DailyDeviceVelocity`, and `TimeSinceLastTx_Hours`) is mutated in the
serving process's memory on every scored transaction; a second concurrent
instance would build its own, divergent view of recent activity instead of
sharing one. Capping at one instance keeps that history consistent at the
cost of the service being unable to handle concurrent load beyond what a
single instance can serve — an acceptable trade for a demo-scale deployment.

### Cost estimate

Approximate, at current (2026) Cloud Run / Firestore / Storage / Artifact
Registry list prices for the `us-central1` region and demo-scale traffic —
check [cloud.google.com/pricing](https://cloud.google.com/pricing) for
current rates before relying on these numbers:

| Item | Weekly, warm | Weekly, scale-to-zero |
|---|---|---|
| Cloud Run idle instance (1 vCPU, 2 GiB) | ~$1.50–2.00 | $0 |
| Cloud Run requests (demo volume) | <$0.05 | <$0.05 |
| Firestore reads/writes (dashboard polling) | <$0.10 | <$0.10 |
| Cloud Storage (bundle ~7MB) | <$0.01 | <$0.01 |
| Artifact Registry (image ~1.5GB) | ~$0.04 | ~$0.04 |
| **Total** | **~$2–3** | **<$0.25** |

"Warm" is `--min-instances=1` left on all week; "scale-to-zero" is
`--min-instances=0`, paying only for the requests actually made.
