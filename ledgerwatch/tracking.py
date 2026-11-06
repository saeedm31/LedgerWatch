"""Experiment tracking: MLflow when it is installed (and MLFLOW_TRACKING_URI is set or a local ./mlruns is fine),
otherwise one JSON line per run in models/runs.jsonl. Same fields either way."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

PARAMS = ("version", "train_months", "test_months", "flags_budget_per_1000", "n_train", "n_positive", "n_reviewer_labels")
METRICS = ("recall", "high_severity_recall", "precision", "flags_per_1000")


def log_run(metrics: dict, model_dir: Path) -> str:
    row = {"ts": datetime.now(UTC).isoformat(timespec="seconds"),
           **{k: metrics.get(k) for k in PARAMS}, **{k: metrics.get(k) for k in METRICS}, "passed": metrics.get("passed")}
    try:
        import mlflow

        mlflow.set_experiment("ledgerwatch")
        with mlflow.start_run(run_name=str(metrics.get("version"))):
            mlflow.log_params({k: row[k] for k in PARAMS})
            mlflow.log_metrics({k: float(row[k]) for k in METRICS if row[k] is not None})
            mlflow.log_dict(metrics, "metrics.json")
        where = "mlflow"
    except Exception:  # not installed or no tracking server: keep a local record
        where = "jsonl"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "runs.jsonl", "a") as fh:
        fh.write(json.dumps({**row, "tracked_in": where}, default=str) + "\n")
    return where
