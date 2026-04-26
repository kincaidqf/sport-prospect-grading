"""Evaluation metrics for regression, classification, and ranking tasks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, and R² for a regression model."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Return accuracy and AUC for binary or multi-class classification.

    For multi-class (y_prob is 2-D with >2 columns): returns macro F1 and
    macro OvR ROC-AUC under the key 'auc'.  For binary (y_prob is 1-D or
    2-column): returns standard ROC-AUC under the same 'auc' key so callers
    never need to branch on the metric name.
    """
    y_prob = np.asarray(y_prob)
    is_multiclass = y_prob.ndim == 2 and y_prob.shape[1] > 2

    out: dict = {"accuracy": float(accuracy_score(y_true, y_pred))}
    if is_multiclass:
        out["auc"]      = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> dict:
    """Return top-k precision (how many of the predicted top-k were actually top-k)."""
    # TODO: implement precision@k and NDCG@k for draft slot evaluation
    raise NotImplementedError


def mean_cost(y_true: np.ndarray, y_pred: np.ndarray, cost_matrix: np.ndarray) -> float:
    """Return average cost incurred per prediction under the given cost matrix.

    cost_matrix[i, j] = penalty when true class is i and predicted class is j.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    costs = cost_matrix[y_true, y_pred]
    return float(np.mean(costs))


def cost_sensitive_predict(proba: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """Return the class that minimises expected cost given predicted probabilities.

    For each sample, picks argmin_c sum_i P(true=i|x) * cost_matrix[i, c].
    """
    return np.argmin(np.asarray(proba) @ cost_matrix, axis=1)


def print_metrics(metrics: dict, prefix: str = "") -> None:
    label = f"[{prefix}] " if prefix else ""
    for k, v in metrics.items():
        print(f"{label}{k}: {v:.4f}")
