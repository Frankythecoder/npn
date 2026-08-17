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
    "TransactionDate": "2023-12-01 03:14:00",
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
| POST | `/score-csv` | Score one chunk of an uploaded CSV |
| GET | `/transactions/recent` | Read back the scored-transaction feed, newest first |
| POST | `/demo/inject` | Score a named preset (with optional field overrides) |
| GET | `/demo/presets` | List the available demo presets |
| GET | `/health` | Liveness/readiness probe |

### Uploading a CSV

The dashboard's **Upload CSV** tab takes a file with `original.csv`'s columns or
any subset of them, parses it in the browser, and posts it to `/score-csv` in
chunks. `/score` and `/batch-score` are unchanged and still require a complete,
validated transaction — the tolerances a file needs are the opposite ones, so
they live behind their own endpoint rather than loosening `TransactionIn`:

- **A file is refused** only if it carries none of `TransactionAmount`,
  `AccountBalance`, `CustomerAge`, `TransactionDuration` or `LoginAttempts`.
- **Absent columns are filled** from the artifact bundle — the training median
  for numerics, the most prevalent level for categoricals, the most frequent
  city for `Location` — and each fill is reported in the row's `warnings`.
  `AccountID`, `DeviceID` and `TransactionDate` are synthesised instead, which
  routes the row through `transform_one`'s unseen-account path rather than
  inventing a history for it.
- **A row is rejected on its own**, with its line number, if a column the file
  does supply is empty or unparseable, or if the ml layer refuses it. The rest
  of the chunk still scores.
- **`IP Address`, `MerchantID`, `PreviousTransactionDate` and `TransactionID`**
  are recognised and ignored, so `original.csv` uploads unchanged.

Each upload scores against **its own profile store**, discarded afterwards. The
store the bundle ships with has already observed every training row, so scoring
that data against it would make `DailyAccountVolume` and `DailyDeviceVelocity`
read one higher than they ever did in training and flag the entire file —
`backend/seed.py` hits the same wall and solves it the same way.

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

# 6b. Grant the service accounts what the deploy and the runtime need.
#     NOTE: not yet exercised against a real project — confirm on first deploy.
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
CLOUDBUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Cloud Build needs to deploy to Cloud Run and act as the runtime account
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CLOUDBUILD_SA}" --role="roles/run.admin"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CLOUDBUILD_SA}" --role="roles/iam.serviceAccountUser"

# The running service needs to read the bundle and write to Firestore
gcloud storage buckets add-iam-policy-binding "${BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/storage.objectViewer"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/datastore.user"

# 7. Build and deploy.
#    _TAG must be passed: cloudbuild.yaml defaults it to "latest", and the
#    SHORT_SHA that repository-triggered builds provide is empty here.
gcloud builds submit --config backend/cloudbuild.yaml \
  --substitutions=_REGION="${REGION}",_REPO="${REPO}",_SERVICE="${SERVICE}",_BUCKET="${BUCKET#gs://}",_TAG="$(git rev-parse --short HEAD)"

# 8. Seed the feed with ~200 real scored transactions, so the dashboard opens
#    on a populated table rather than an empty one. Run as a one-shot job
#    against the image just built: it already contains original.csv, the
#    google-cloud libraries and the code, none of which are installed locally.
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:$(git rev-parse --short HEAD)"
gcloud run jobs create anomaly-seed --image "${IMAGE}" --region "${REGION}" \
  --set-env-vars=ANOMALY_ARTIFACT_SOURCE=gcs,ANOMALY_GCS_BUCKET="${BUCKET#gs://}",ANOMALY_TRANSACTION_LOG=firestore \
  --command=python --args=-m,backend.seed,--limit,200
gcloud run jobs execute anomaly-seed --region "${REGION}" --wait
```

`backend/cloudbuild.yaml` builds the image from `backend/Dockerfile`, pushes
it to Artifact Registry, and deploys it to Cloud Run with
`ANOMALY_ARTIFACT_SOURCE=gcs`, `ANOMALY_GCS_BUCKET`, and
`ANOMALY_TRANSACTION_LOG=firestore` set, so the deployed service reads the
bundle uploaded in step 6 and writes scored transactions to Firestore.

Step 8 runs `backend/seed.py` in the deployed image. It is deliberately not
part of `python -m ml.pipeline.train`: putting it there would make offline
training import Firestore, which would break the training tests on any
machine without the `google-cloud` libraries. Running it locally is not
supported for the same reason — those libraries are intentionally absent
from `requirements.txt`.

### Min-instances toggle

**The deploy in step 7 already sets `--min-instances=1`**, so the service is
billed for one warm instance from the moment it goes up. That is deliberate:
a cold start takes 8–20 seconds while the container starts, downloads the
~7MB bundle and unpickles eight detectors, and a first request that slow
during a live demonstration is unacceptable.

It is not free, so turn it off when you are not demonstrating:

```bash
# Not demonstrating — scale to zero and stop paying for idle time
gcloud run services update "${SERVICE}" --region "${REGION}" --min-instances=0
# Before a demonstration — pin an instance so the first request is instant
gcloud run services update "${SERVICE}" --region "${REGION}" --min-instances=1
```

### Running the image locally

The image defaults to `ANOMALY_ARTIFACT_SOURCE=local` and
`ANOMALY_ARTIFACT_DIR=artifacts`, and `artifacts/` is **not** copied into it.
So `docker run` with no environment set fails at startup. Either mount a
trained bundle and point at it, or set the GCS variables the deployed service
uses:

```bash
docker run -p 8080:8080 -v "$(pwd)/artifacts:/app/artifacts" <image>
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
| Cloud Run idle instance (1 vCPU) | ~$1.50–2.00 | $0 |
| Cloud Run requests (demo volume) | <$0.05 | <$0.05 |
| Firestore reads/writes (dashboard polling) | <$0.10 | <$0.10 |
| Cloud Storage (bundle ~7MB) | <$0.01 | <$0.01 |
| Artifact Registry (image ~1.5GB) | ~$0.04 | ~$0.04 |
| **Total** | **~$2–3** | **<$0.25** |

"Warm" is `--min-instances=1` left on all week; "scale-to-zero" is
`--min-instances=0`, paying only for the requests actually made.
