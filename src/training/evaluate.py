"""Evaluation metrics for regression, classification, and ranking tasks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
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


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Return accuracy, balanced_accuracy, AUC, macro F1, and per-class metrics.

    For multi-class (y_prob is 2-D with >2 columns): returns macro OvR ROC-AUC
    and per-class precision/recall/f1 keyed by class name.  For binary: returns
    standard ROC-AUC.  The 'auc' key is always present so callers never branch.
    """
    y_prob = np.asarray(y_prob)
    is_multiclass = y_prob.ndim == 2 and y_prob.shape[1] > 2

    out: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if is_multiclass:
        out["auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        n_classes = len(precision)
        names = (class_names or [f"class_{i}" for i in range(n_classes)])[:n_classes]
        for i, name in enumerate(names):
            out[f"precision_{name}"] = float(precision[i])
            out[f"recall_{name}"] = float(recall[i])
            out[f"f1_{name}"] = float(fscore[i])
    else:
        out["auc"] = float(roc_auc_score(y_true, y_prob))

    return out


def ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> dict:
    """Return top-k precision (how many of the predicted top-k were actually top-k)."""
    # TODO: implement precision@k and NDCG@k for draft slot evaluation
    raise NotImplementedError


def print_metrics(metrics: dict, prefix: str = "") -> None:
    label = f"[{prefix}] " if prefix else ""
    for k, v in metrics.items():
        print(f"{label}{k}: {v:.4f}")
