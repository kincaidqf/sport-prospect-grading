"""Evaluation metrics for regression, classification, and ranking tasks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, and R² for a regression model."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Return accuracy and ROC-AUC for a binary classification model."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> dict:
    """Return top-k precision (how many of the predicted top-k were actually top-k)."""
    # TODO: implement precision@k and NDCG@k for draft slot evaluation
    raise NotImplementedError


def print_metrics(metrics: dict, prefix: str = "") -> None:
    label = f"[{prefix}] " if prefix else ""
    for k, v in metrics.items():
        print(f"{label}{k}: {v:.4f}")
