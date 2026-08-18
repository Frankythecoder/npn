# ml

Anomaly detection over the transaction dataset.

## Train

    python -m ml.pipeline.train

Fits seven detectors, builds the ensemble and the surrogate, prints the
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
