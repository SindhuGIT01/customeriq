"""Reusable evaluation utilities for classification models."""

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Compute standard classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def to_json_safe(value):
    """Recursively convert numpy/pandas scalar and array types to plain
    Python types so a metrics dict can be dumped with the stdlib json module."""
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def save_metrics_json(data: dict, path: str) -> None:
    """Save a metrics dict to disk as JSON, coercing numpy types first."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(data), f, indent=2)
