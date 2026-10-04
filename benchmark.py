#!/usr/bin/env python3
"""LightGBM benchmark on Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud).

Measures data load time, training time (early stopping on a validation split),
test-set metrics (AUC-ROC, Accuracy, F1, Precision, Recall), single-row inference
latency and 1000-row batch throughput, then writes everything to a JSON file.

Usage:
    python3 benchmark.py --data ~/ml-benchmark/creditcard.csv --out benchmark_result.json
"""
import argparse
import inspect
import json
import os
import platform
import socket
import statistics
import time
import urllib.request
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

SEED = 42
TARGET = "Class"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.expanduser("~/ml-benchmark/creditcard.csv"))
    parser.add_argument("--out", default="benchmark_result.json")
    parser.add_argument("--threshold", type=float, default=0.5, help="probability cut-off for the fraud class")
    parser.add_argument("--latency-runs", type=int, default=200, help="timed single-row predictions")
    parser.add_argument("--throughput-runs", type=int, default=20, help="timed 1000-row batch predictions")
    return parser.parse_args()


def imds(path):
    """Read EC2 instance metadata (IMDSv2). Returns None when not running on EC2."""
    try:
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(token_req, timeout=1).read().decode()
        req = urllib.request.Request(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(req, timeout=1).read().decode()
    except Exception:
        return None


def total_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    return None


def environment_info():
    return {
        "hostname": socket.gethostname(),
        "ec2_instance_type": imds("instance-type"),
        "ec2_region": imds("placement/region"),
        "ec2_availability_zone": imds("placement/availability-zone"),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": total_ram_gb(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def percentile(values, q):
    return float(np.percentile(values, q))


def measure_latency(model, X, runs):
    """Predict one row at a time (cycling through test rows), return per-call ms stats."""
    for i in range(10):  # warm-up
        model.predict_proba(X.iloc[[i]])
    timings_ms = []
    for i in range(runs):
        row = X.iloc[[i % len(X)]]
        start = time.perf_counter()
        model.predict_proba(row)
        timings_ms.append((time.perf_counter() - start) * 1000)
    return {
        "runs": runs,
        "mean_ms": round(statistics.mean(timings_ms), 4),
        "median_ms": round(statistics.median(timings_ms), 4),
        "p95_ms": round(percentile(timings_ms, 95), 4),
        "p99_ms": round(percentile(timings_ms, 99), 4),
    }


def measure_throughput(model, X, runs, batch_size=1000):
    """Predict a fixed 1000-row batch repeatedly, return batch time and rows/second."""
    batch = X.iloc[:batch_size]
    for _ in range(3):  # warm-up
        model.predict_proba(batch)
    timings_s = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict_proba(batch)
        timings_s.append(time.perf_counter() - start)
    median_s = statistics.median(timings_s)
    return {
        "batch_size": len(batch),
        "runs": runs,
        "batch_median_ms": round(median_s * 1000, 4),
        "batch_mean_ms": round(statistics.mean(timings_s) * 1000, 4),
        "rows_per_second": round(len(batch) / median_s, 1),
    }


def main():
    args = parse_args()
    started = datetime.now(timezone.utc)

    print(f"[1/5] Loading {args.data}")
    start = time.perf_counter()
    df = pd.read_csv(args.data)
    load_s = time.perf_counter() - start
    print(f"      {df.shape[0]:,} rows x {df.shape[1]} cols in {load_s:.3f} s")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    # 80% train+val / 20% test, then 10% of train+val as validation for early stopping.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=SEED
    )
    print(f"[2/5] Split: train {len(X_train):,} / val {len(X_val):,} / test {len(X_test):,} (stratified, seed {SEED})")

    params = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    # Early stopping watches validation AUPRC only (first_metric_only): with ~0.17% fraud the
    # validation split holds ~40 positives, and logloss/ROC-AUC there peak after a few dozen
    # trees and stop training far too early. Kaggle recommends AUPRC for this dataset.
    early_stopping = {"metric": "average_precision", "patience": 100}
    model = lgb.LGBMClassifier(**params)
    print(f"[3/5] Training LGBMClassifier (early stopping {early_stopping['patience']} rounds on validation AUPRC)")
    # LightGBM >= 4.7 deprecates eval_set in favour of eval_X/eval_y; older versions only know eval_set.
    if "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters:
        eval_kwargs = {"eval_X": (X_val,), "eval_y": (y_val,)}
    else:
        eval_kwargs = {"eval_set": [(X_val, y_val)]}
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_metric=early_stopping["metric"],
        callbacks=[lgb.early_stopping(early_stopping["patience"], first_metric_only=True, verbose=False)],
        **eval_kwargs,
    )
    train_s = time.perf_counter() - start
    best_iteration = int(model.best_iteration_ or params["n_estimators"])
    print(f"      done in {train_s:.3f} s, best iteration {best_iteration}")

    print("[4/5] Evaluating on test set")
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= args.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    metrics = {
        "auc_roc": round(float(roc_auc_score(y_test, proba)), 6),
        "auprc": round(float(average_precision_score(y_test, proba)), 6),
        "accuracy": round(float(accuracy_score(y_test, pred)), 6),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 6),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 6),
        "threshold": args.threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    print("[5/5] Measuring inference latency (1 row) and throughput (1000 rows)")
    latency = measure_latency(model, X_test, args.latency_runs)
    throughput = measure_throughput(model, X_test, args.throughput_runs)

    result = {
        "timestamp_utc": started.isoformat(timespec="seconds"),
        "environment": environment_info(),
        "dataset": {
            "path": os.path.abspath(args.data),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "fraud_rows": int(y.sum()),
            "fraud_ratio": round(float(y.mean()), 6),
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
        },
        "model": {
            "type": "lightgbm.LGBMClassifier",
            "params": params,
            "early_stopping": early_stopping,
            "best_iteration": best_iteration,
        },
        "timing": {"load_data_s": round(load_s, 4), "training_s": round(train_s, 4)},
        "metrics": metrics,
        "inference": {"latency_1_row": latency, "throughput_1000_rows": throughput},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    env = result["environment"]
    rows = [
        ("Thời gian load data", f"{load_s:.3f} s"),
        ("Thời gian training", f"{train_s:.3f} s"),
        ("Best iteration", f"{best_iteration}"),
        ("AUC-ROC", f"{metrics['auc_roc']:.4f}"),
        ("AUPRC (average precision)", f"{metrics['auprc']:.4f}"),
        ("Accuracy", f"{metrics['accuracy']:.4f}"),
        ("F1-Score", f"{metrics['f1']:.4f}"),
        ("Precision", f"{metrics['precision']:.4f}"),
        ("Recall", f"{metrics['recall']:.4f}"),
        ("Inference latency (1 row)", f"{latency['median_ms']:.3f} ms (median, p95 {latency['p95_ms']:.3f} ms)"),
        (
            "Inference throughput (1000 rows)",
            f"{throughput['batch_median_ms']:.3f} ms/batch = {throughput['rows_per_second']:,.0f} rows/s",
        ),
    ]
    width = max(len(name) for name, _ in rows)
    print()
    print(f"Host: {env['hostname']} | instance: {env['ec2_instance_type'] or 'n/a'} | "
          f"region: {env['ec2_region'] or 'n/a'} | CPUs: {env['cpu_count']} | RAM: {env['ram_total_gb']} GB")
    print(f"Confusion matrix (test, threshold {args.threshold}): TN={tn} FP={fp} FN={fn} TP={tp}")
    print("-" * (width + 40))
    for name, value in rows:
        print(f"{name:<{width}}  {value}")
    print("-" * (width + 40))
    print(f"Saved -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
